import { promises as fs } from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// If PYTHON_EXE is not set, fall back to `conda run -n argus python`
// so no path needs to be hardcoded regardless of where conda is installed.
const [PYTHON_EXE, PYTHON_PREFIX_ARGS]: [string, string[]] = process.env.PYTHON_EXE
  ? [process.env.PYTHON_EXE, []]
  : ["conda", ["run", "--no-capture-output", "-n", "argus", "python"]];
const WORKSPACE_ROOT = path.resolve(process.cwd(), "..");
const RUNNER = path.join(process.cwd(), "scripts", "query_runner.py");
const CACHE_DIR = path.join(WORKSPACE_ROOT, "outputs", "query_cache");
// Live DMD fetches N snapshots sequentially from remote data — give it more time.
// Regular queries (single PNG, depth sweep, animation) keep the tighter limit.
const TIMEOUT_MS = 300_000; // 5 min: depth sweep / multi-face fetch can be slow even parallelized
const TIMEOUT_MS_LIVE_DMD = 1_800_000; // 30 min: fetch + Tucker HOSVD + DMD + render (599 ECCO frames took ~7 min)

const ALLOWED_DATASETS = new Set(["dyamond", "ecco", "cmip6", "dmd", "dmd_live"]);
const ALLOWED_RENDER_MODES = new Set(["grid", "globe"]);

function sanitizeRenderMode(value: unknown): string {
  if (typeof value === "string" && ALLOWED_RENDER_MODES.has(value)) return value;
  return "globe";
}

type AnyRecord = Record<string, unknown>;

function clampNum(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function sanitizeString(value: unknown, allowed: RegExp, fallback: string): string {
  if (typeof value !== "string") return fallback;
  return allowed.test(value) ? value : fallback;
}

function validate(body: AnyRecord): AnyRecord | null {
  const dataset = body.dataset;
  if (typeof dataset !== "string" || !ALLOWED_DATASETS.has(dataset)) return null;
  const render_mode = sanitizeRenderMode(body.render_mode);

  // ── DMD / Tucker analysis viewer (uses pre-computed .npz files) ────────────
  if (dataset === "dmd") {
    const tech = sanitizeString(body.tech, /^(dmd|mrdmd|tucker)$/, "dmd");
    const task = sanitizeString(body.task, /^(wind|ocean|coupled)$/, "wind");
    const mode_idx = clampNum(body.mode_idx, -1, 19, 0);
    const n_frames = clampNum(body.n_frames, 1, 60, 30);
    const show_all = body.show_all === true;
    const viz = sanitizeString(body.viz, /^(animation|eigenvalues|reconstruction)$/, "animation");
    const n_modes_use = clampNum(body.n_modes_use, 1, 20, 20);
    return { dataset, tech, task, mode_idx, n_frames, show_all, viz, n_modes_use };
  }

  if (dataset === "dyamond") {
    // Ocean (LLC2160 / ECCO fallback) path — matches contest reference notebook.
    if (body.ocean === true) {
      const variable = sanitizeString(body.variable, /^(theta|salt|u|v|w)$/, "theta");
      const t_start = clampNum(body.t_start, 0, 10312, 0);
      const t_end   = clampNum(body.t_end, t_start, 10312, Math.min(t_start + 3, 10312));
      const depth_level = clampNum(body.depth_level, 0, 89, 0);
      const quality = clampNum(body.quality, -15, 0, -6);
      const animate = body.animate === true;
      const n_frames = clampNum(body.n_frames, 1, 30, 20);
      return { dataset, ocean: true, variable, t_start, t_end, depth_level, quality, render_mode, animate, n_frames };
    }
    // Coupled atm + ocean composite globe path.
    if (body.coupled === true) {
      const atm_variable = sanitizeString(body.atm_variable, /^(u|v|t|fcld|p|w|h|co|co2|qi|ql|ri|rl|delp|dthdt|dthdtcn)$/, "u");
      const ocn_variable = sanitizeString(body.ocn_variable, /^(theta|salt|u|v|w)$/, "theta");
      const t_atm = clampNum(body.t_atm, 0, 10269, 10);
      const t_ocn = clampNum(body.t_ocn, 0, 10000, 200);
      const z_level_atm = clampNum(body.z_level_atm, 0, 71, 50);
      const depth_level_ocn = clampNum(body.depth_level_ocn, 0, 89, 10);
      const quality = clampNum(body.quality, -15, 0, -4);
      return {
        dataset,
        coupled: true,
        atm_variable,
        ocn_variable,
        t_atm,
        t_ocn,
        z_level_atm,
        depth_level_ocn,
        quality,
        render_mode: "globe",
      };
    }
    const variable = sanitizeString(body.variable, /^(u|v|t|fcld|p|w|h|co|co2|qi|ql|ri|rl|delp|dthdt|dthdtcn)$/, "u");
    const face = clampNum(body.face, 0, 5, 0);
    const t_start = clampNum(body.t_start, 0, 10269, 0);
    const t_end = clampNum(body.t_end, t_start, 10269, Math.min(t_start + 4, 10269));
    const z_level = clampNum(body.z_level, 0, 51, 50); // DYAMOND GEOS depth_52 → levels 0-51
    const quality = clampNum(body.quality, -9, 0, -4); // cap at -9 to stay within IDX pyramid
    const all_faces = body.all_faces === true;
    const animate = body.animate === true;
    const n_frames = clampNum(body.n_frames, 1, 30, 20);
    const depth_level = clampNum(body.depth_level, 0, 51, 0); // animation depth_level same range
    // Face Explorer params
    const z_sweep = body.z_sweep === true;
    const cross_section = body.cross_section === true;
    const all_panel = body.all_panel === true;
    const z_start = clampNum(body.z_start, 0, 51, 0);
    const z_end = clampNum(body.z_end, z_start, 51, Math.min(z_start + 20, 51));
    const y_row = clampNum(body.y_row, 0, 1439, 720);
    const t_snap = clampNum(body.t_snap, 0, 10269, 5000);
    return { dataset, variable, face, t_start, t_end, z_level, quality, render_mode, all_faces, animate, n_frames, depth_level, z_sweep, cross_section, all_panel, z_start, z_end, y_row, t_snap };
  }
  if (dataset === "ecco") {
    const variable = sanitizeString(body.variable, /^(theta|salt)$/, "theta");
    const t_start = clampNum(body.t_start, 0, 10000, 0);
    const t_end = clampNum(body.t_end, t_start, 10000, Math.min(t_start + 3, 10000));
    const depth_start = clampNum(body.depth_start, 0, 89, 0);
    const depth_end = clampNum(body.depth_end, depth_start, 89, Math.max(depth_start, 5));
    const quality = clampNum(body.quality, -15, 0, -6);
    const animate = body.animate === true;
    const n_frames = clampNum(body.n_frames, 1, 30, 20);
    const depth_level = clampNum(body.depth_level, 0, 89, 0);
    return { dataset, variable, t_start, t_end, depth_start, depth_end, depth_level, quality, render_mode, animate, n_frames };
  }
  // dmd_live — live DMD / rDMD / HOSVD on fetched snapshots
  if (dataset === "dmd_live") {
    const source      = sanitizeString(body.source, /^(dyamond|ecco|cmip6)$/, "dyamond");
    const tech        = sanitizeString(body.tech, /^(dmd|rdmd|hosvd|rank_sweep)$/, "dmd");
    const n_frames = clampNum(body.n_frames, 4, 30, 20);
    const n_modes  = clampNum(body.n_modes, 4, 12, 8);
    // No sampling parameter — backend streams ALL steps in the range at render quality.
    if (source === "dyamond") {
      const variable = sanitizeString(body.variable, /^(u|v|t|fcld|p|w|h|co|co2|qi|ql|ri|rl|delp|dthdt|dthdtcn)$/, "u");
      const face     = clampNum(body.face, 0, 5, 0);
      const t_start  = clampNum(body.t_start, 0, 10269, 0);
      const t_end    = clampNum(body.t_end, t_start, 10269, Math.min(t_start + 160, 10269));
      const z_level  = clampNum(body.z_level, 0, 51, 50);
      const quality  = clampNum(body.quality, -9, 0, -6);
      return { dataset, source, tech, variable, face, t_start, t_end, z_level, quality, n_frames, n_modes };
    } else if (source === "cmip6") {
      const variable = sanitizeString(body.variable, /^(tas|tasmin|tasmax|pr|rsds|rlds|sfcWind|hurs)$/, "tas");
      const model    = sanitizeString(body.model, /^[A-Za-z0-9._-]+$/, "ACCESS-CM2");
      const scenario = sanitizeString(body.scenario, /^(historical|ssp245|ssp585)$/, "historical");
      const t_start  = clampNum(body.t_start, 0, 100000, 0);
      const t_end    = clampNum(body.t_end, t_start, 100000, Math.min(t_start + 200, 100000));
      return { dataset, source, tech, variable, model, scenario, t_start, t_end, n_frames, n_modes };
    } else {
      const variable    = sanitizeString(body.variable, /^(theta|salt)$/, "theta");
      const t_start     = clampNum(body.t_start, 0, 10000, 0);
      const t_end       = clampNum(body.t_end, t_start, 10000, Math.min(t_start + 200, 10000));
      const depth_level = clampNum(body.depth_level, 0, 89, 0);
      const quality     = clampNum(body.quality, -15, 0, -8);
      return { dataset, source, tech, variable, t_start, t_end, depth_level, quality, n_frames, n_modes };
    }
  }

  // cmip6
  const variable = sanitizeString(body.variable, /^(tas|tasmin|tasmax|pr|rsds|rlds|sfcWind|hurs)$/, "tas");
  const model = sanitizeString(
    body.model,
    /^[A-Za-z0-9._-]+$/,
    "ACCESS-CM2",
  );
  const scenario = sanitizeString(body.scenario, /^(historical|ssp245|ssp585)$/, "historical");
  const t_start = clampNum(body.t_start, 0, 100000, 0);
  const t_end = clampNum(body.t_end, t_start, 100000, Math.min(t_start + 10, 100000));
  const animate = body.animate === true;
  const n_frames = clampNum(body.n_frames, 1, 30, 20);
  return { dataset, variable, model, scenario, t_start, t_end, render_mode, animate, n_frames };
}

function hashParams(params: AnyRecord): string {
  const stable = JSON.stringify(params, Object.keys(params).sort());
  return crypto.createHash("sha256").update(stable).digest("hex").slice(0, 24);
}

async function ensureCacheDir() {
  await fs.mkdir(CACHE_DIR, { recursive: true });
}

function runPython(params: AnyRecord): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_EXE, [...PYTHON_PREFIX_ARGS, RUNNER], {
      cwd: WORKSPACE_ROOT,
      windowsHide: true,
    });

    const chunks: Buffer[] = [];
    const errChunks: Buffer[] = [];
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, TIMEOUT_MS);

    child.stdout.on("data", (d: Buffer) => chunks.push(d));
    child.stderr.on("data", (d: Buffer) => errChunks.push(d));

    child.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) {
        return reject(new Error(`Python runner timed out after ${TIMEOUT_MS}ms`));
      }
      if (code !== 0) {
        const err = Buffer.concat(errChunks).toString("utf8").trim();
        return reject(new Error(err || `Python exited with code ${code}`));
      }
      resolve(Buffer.concat(chunks));
    });

    child.stdin.write(JSON.stringify(params));
    child.stdin.end();
  });
}

function streamPython(params: AnyRecord, cachePath: string, timeoutMs = TIMEOUT_MS): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const t0 = Date.now();

  const write = (controller: ReadableStreamDefaultController<Uint8Array>, obj: unknown) => {
    try {
      controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));
    } catch {
      /* stream closed */
    }
  };

  const elapsed = () => Number(((Date.now() - t0) / 1000).toFixed(2));

  return new ReadableStream<Uint8Array>({
    start(controller) {
      const child = spawn(PYTHON_EXE, [...PYTHON_PREFIX_ARGS, RUNNER], {
        cwd: WORKSPACE_ROOT,
        windowsHide: true,
      });

      const stdoutChunks: Buffer[] = [];
      let stderrBuf = "";
      const errLines: string[] = [];
      let timedOut = false;
      let eventsEmitted = 0;

      const timer = setTimeout(() => {
        timedOut = true;
        child.kill("SIGKILL");
      }, timeoutMs);

      child.stdout.on("data", (d: Buffer) => stdoutChunks.push(d));

      child.stderr.on("data", (d: Buffer) => {
        stderrBuf += d.toString("utf8");
        let idx;
        while ((idx = stderrBuf.indexOf("\n")) !== -1) {
          const line = stderrBuf.slice(0, idx).replace(/\r$/, "");
          stderrBuf = stderrBuf.slice(idx + 1);
          if (line.startsWith("PROGRESS ")) {
            write(controller, {
              type: "progress",
              msg: line.slice("PROGRESS ".length),
              t: elapsed(),
            });
          } else if (line.startsWith("EVENT ")) {
            try {
              const obj = JSON.parse(line.slice("EVENT ".length));
              write(controller, { ...obj, t: elapsed() });
              eventsEmitted += 1;
            } catch {
              // ignore malformed event
            }
          } else if (line.length > 0) {
            errLines.push(line);
          }
        }
      });

      child.on("error", (e) => {
        clearTimeout(timer);
        write(controller, { type: "error", msg: e.message, t: elapsed() });
        try { controller.close(); } catch { /* noop */ }
      });

      child.on("close", async (code) => {
        clearTimeout(timer);
        if (stderrBuf.length > 0) {
          const line = stderrBuf.replace(/\r$/, "");
          if (line.startsWith("PROGRESS ")) {
            write(controller, {
              type: "progress",
              msg: line.slice("PROGRESS ".length),
              t: elapsed(),
            });
          } else {
            errLines.push(line);
          }
          stderrBuf = "";
        }

        if (timedOut) {
          write(controller, {
            type: "error",
            msg: `Python runner timed out after ${timeoutMs}ms`,
            t: elapsed(),
          });
          try { controller.close(); } catch { /* noop */ }
          return;
        }

        if (code !== 0) {
          const msg = errLines.join("\n").trim() || `Python exited with code ${code}`;
          write(controller, { type: "error", msg: msg.slice(0, 2000), t: elapsed() });
          try { controller.close(); } catch { /* noop */ }
          return;
        }

        // EVENTs are authoritative: if the runner streamed any events, the
        // payload is fully delivered and stdout holds only the b"" return (or
        // stray debug noise) — never treat it as an image.
        if (eventsEmitted > 0) {
          try { controller.close(); } catch { /* noop */ }
          return;
        }

        const png = Buffer.concat(stdoutChunks);
        if (png.length === 0) {
          write(controller, { type: "error", msg: "Empty PNG from runner.", t: elapsed() });
          try { controller.close(); } catch { /* noop */ }
          return;
        }

        await fs.writeFile(cachePath, new Uint8Array(png)).catch(() => {});
        const dataUrl = `data:image/png;base64,${png.toString("base64")}`;
        write(controller, { type: "image", dataUrl, cached: false, t: elapsed() });
        try { controller.close(); } catch { /* noop */ }
      });

      child.stdin.write(JSON.stringify(params));
      child.stdin.end();
    },
  });
}

export async function POST(request: NextRequest) {
  let body: AnyRecord;
  try {
    body = (await request.json()) as AnyRecord;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const params = validate(body);
  if (!params) {
    return NextResponse.json({ error: "Invalid or missing parameters." }, { status: 400 });
  }

  await ensureCacheDir();
  const hash = hashParams(params);
  const cachePath = path.join(CACHE_DIR, `${hash}.png`);

  const acceptsNdjson = (request.headers.get("accept") ?? "").includes("application/x-ndjson");
  const debug = body.debug === true || acceptsNdjson;

  if (debug) {
    // Streaming NDJSON path
    const encoder = new TextEncoder();

    // Multi-event queries (animation, cube faces, coupled layers, dmd) emit
    // multiple events and never write a single PNG to cache, so cache is useless
    // for them and can even serve stale single-image results from a previous run.
    const p = params as AnyRecord;
    const isMultiEvent =
      p.animate ||
      p.coupled ||
      p.dataset === "dmd" ||
      p.dataset === "dmd_live" ||
      // ECCO depth-stack sweep streams one image_depth event per level and
      // never writes a single PNG — must skip the single-image cache or a stale
      // hit gets served as a globe texture instead of the depth slices.
      (p.dataset === "ecco" &&
        p.render_mode === "globe" &&
        Number(p.depth_end) > Number(p.depth_start)) ||
      // Face Explorer z-sweep emits image_frame events and returns empty bytes.
      (p.dataset === "dyamond" && p.z_sweep);

    if (!isMultiEvent) {
      // Cache hit: short stream with progress + image
      try {
        const cached = await fs.readFile(cachePath);
        const dataUrl = `data:image/png;base64,${Buffer.from(cached).toString("base64")}`;
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(
              encoder.encode(
                JSON.stringify({ type: "progress", msg: "cache hit", t: 0 }) + "\n",
              ),
            );
            controller.enqueue(
              encoder.encode(
                JSON.stringify({ type: "image", dataUrl, cached: true, t: 0 }) + "\n",
              ),
            );
            controller.close();
          },
        });
        return new Response(stream, {
          headers: {
            "Content-Type": "application/x-ndjson",
            "X-Cache": "HIT",
            "Cache-Control": "no-store",
          },
        });
      } catch {
        // cache miss — fall through
      }
    }

    const streamTimeout = (params as AnyRecord).dataset === "dmd_live" ? TIMEOUT_MS_LIVE_DMD : TIMEOUT_MS;
    const stream = streamPython(params, cachePath, streamTimeout);
    return new Response(stream, {
      headers: {
        "Content-Type": "application/x-ndjson",
        "X-Cache": "MISS",
        "Cache-Control": "no-store",
      },
    });
  }

  // Legacy binary path (kept for simplicity/tests)
  try {
    const cached = await fs.readFile(cachePath);
    return new NextResponse(new Uint8Array(cached), {
      headers: {
        "Content-Type": "image/png",
        "X-Cache": "HIT",
        "Cache-Control": "private, max-age=300",
      },
    });
  } catch {
    // cache miss — fall through
  }

  try {
    const png = await runPython(params);
    if (png.length === 0) {
      return NextResponse.json({ error: "Empty PNG from runner." }, { status: 500 });
    }
    await fs.writeFile(cachePath, new Uint8Array(png)).catch(() => {});
    return new NextResponse(new Uint8Array(png), {
      headers: {
        "Content-Type": "image/png",
        "X-Cache": "MISS",
        "Cache-Control": "private, max-age=300",
      },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: msg.slice(0, 2000) }, { status: 500 });
  }
}
