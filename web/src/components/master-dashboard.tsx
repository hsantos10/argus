"use client";

import { useEffect, useRef, useState } from "react";
import { GlobeViewer } from "@/components/globe-viewer";
import { CubeViewer } from "@/components/cube-viewer";
import { DepthStackViewer, type DepthSlice } from "@/components/depth-stack-viewer";
import { OceanGlobeViewer, GlobeControlsPanel, useGlobeData, type GlobeLogLine } from "@/components/ocean-globe-viewer";
import { VolumeViewer } from "@/components/volume-viewer";
import { cn } from "@/lib/utils";

// ── Variable metadata ──────────────────────────────────────────────────────────

type VarMeta = { unit: string; cmap: string; vmin: number; vmax: number };

const VAR_META: Record<string, VarMeta> = {
  u:       { unit: "m/s",  cmap: "RdBu_r", vmin: -50,  vmax: 50   },
  v:       { unit: "m/s",  cmap: "RdBu_r", vmin: -50,  vmax: 50   },
  w:       { unit: "Pa/s", cmap: "RdBu_r", vmin: -0.5, vmax: 0.5  },
  t:       { unit: "K",    cmap: "plasma",  vmin: 200,  vmax: 310  },
  theta:   { unit: "°C",   cmap: "plasma",  vmin: -2,   vmax: 32   },
  salt:    { unit: "PSU",  cmap: "viridis", vmin: 32,   vmax: 37   },
  tas:     { unit: "K",    cmap: "plasma",  vmin: 230,  vmax: 310  },
  tasmax:  { unit: "K",    cmap: "Reds",    vmin: 250,  vmax: 330  },
  pr:      { unit: "mm/d", cmap: "Blues",   vmin: 0,    vmax: 15   },
  hurs:    { unit: "%",    cmap: "BrBG",    vmin: 0,    vmax: 100  },
  sfcWind: { unit: "m/s",  cmap: "YlOrRd",  vmin: 0,    vmax: 15   },
};

const CMAP_CSS: Record<string, string[]> = {
  plasma:  ["#0d0887","#6a00a8","#b12a90","#e16462","#fca636","#f0f921"],
  RdBu_r:  ["#053061","#2166ac","#4393c3","#d1e5f0","#fddbc7","#ef8a62","#b2182b","#67001f"],
  viridis: ["#440154","#414487","#2a788e","#22a884","#7ad151","#fde725"],
  Blues:   ["#08306b","#2171b5","#6baed6","#bdd7e7","#eff3ff"],
  Reds:    ["#67000d","#cb181d","#fb6a4a","#fc9272","#fee5d9"],
  BrBG:    ["#543005","#8c510a","#d8b365","#f5f5f5","#80cdc1","#01665e","#003c30"],
  YlOrRd:  ["#ffffb2","#fed976","#feb24c","#fd8d3c","#f03b20","#bd0026"],
};

function cmapGradient(cmap: string) {
  const stops = CMAP_CSS[cmap] ?? CMAP_CSS.plasma;
  return `linear-gradient(to top, ${stops.join(",")})`;
}

// ── Dataset constants ──────────────────────────────────────────────────────────

const DATASETS = [
  { name: "CMIP6",   pts: 259_200,    res: "~25 km", color: "#00c4a8", vars: "Daily climate projections to 2100" },
  { name: "DYAMOND", pts: 8_294_400,   res: "4–7 km", color: "#4a9eff", vars: "Storm-resolving ocean + atmosphere" },
  { name: "ECCO",    pts: 55_987_200,  res: "~2 km",  color: "#f4a836", vars: "Global ocean circulation, hourly" },
] as const;

const PRESETS = [
  {
    id: "cmip6_warming",
    label: "Global Warming",
    desc: "Projected surface air temperature · 2090, worst-case",
    badge: "CMIP6",
    color: "#00c4a8",
    body: { dataset: "cmip6", variable: "tas", model: "ACCESS-CM2", scenario: "ssp585",
            t_start: 23725, t_end: 25567, animate: true, n_frames: 12, render_mode: "globe" },
  },
  {
    id: "dyamond_wind",
    label: "Jet Streams",
    desc: "Upper-atmosphere wind speed at ~10 km",
    badge: "DYAMOND",
    color: "#4a9eff",
    body: { dataset: "dyamond", variable: "u", face: 0, t_start: 6000, t_end: 6000,
            z_level: 40, quality: -7, render_mode: "globe" },
  },
  {
    id: "ecco_theta",
    label: "Ocean SST",
    desc: "Sea-surface temperature (top layer only)",
    badge: "ECCO",
    color: "#f4a836",
    body: { dataset: "ecco", variable: "theta", t_start: 0, t_end: 4380,
            depth_level: 0, quality: -12, animate: true, n_frames: 10, render_mode: "globe" },
  },
  {
    id: "depth_stack",
    label: "Ocean Depths",
    desc: "Temperature down through the water column",
    badge: "ECCO",
    color: "#a06ef4",
    body: { dataset: "ecco", variable: "theta", t_start: 0, t_end: 0,
            depth_start: 0, depth_end: 89, quality: -12, render_mode: "globe" },
  },
] as const;

type ViewMode = "3d" | "2d";

// ── Archived modes (2026-07-08) ────────────────────────────────────────────────
// The Python-pipeline "Analysis" tab and the "3D Slab" volume view are hidden
// but fully intact — flip these flags to restore their toggles. All presets now
// drive the GPU globe directly instead of switching into these modes.
const SHOW_ANALYSIS = false;
const SHOW_SLAB = false;

// ── Dataset scale bars ─────────────────────────────────────────────────────────

function DatasetBars() {
  const logMax = Math.log10(DATASETS[2].pts);
  return (
    <div className="space-y-3">
      {DATASETS.map((d) => {
        const pct = (Math.log10(d.pts) / logMax) * 100;
        return (
          <div key={d.name} className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[12px] font-semibold" style={{ color: d.color + "cc" }}>{d.name}</span>
              <span className="font-mono text-[11px] text-white/35">{d.res} · {(d.pts / 1e6).toFixed(1)}M</span>
            </div>
            <div className="h-[3px] w-full overflow-hidden rounded-full bg-white/[0.06]">
              <div className="h-full rounded-full" style={{ width: `${pct}%`, background: d.color + "99" }} />
            </div>
            <p className="font-mono text-[10px] text-white/25 tracking-wide">{d.vars}</p>
          </div>
        );
      })}
    </div>
  );
}

// ── Query runner ───────────────────────────────────────────────────────────────

type AnimFrame = { dataUrl: string; label: string };
type LogLine = { t: number; msg: string };

type QState = {
  loading: boolean;
  textureUrl: string | null;
  timeFrames: AnimFrame[];
  totalExpected: number;
  faces: (string | null)[];
  depthSlices: DepthSlice[];
  currentStep: string;
  log: LogLine[];
  ms: number | null;
  error: string | null;
};

const Q0: QState = {
  loading: false, textureUrl: null, timeFrames: [], totalExpected: 0,
  faces: [null, null, null, null, null, null], depthSlices: [],
  currentStep: "", log: [], ms: null, error: null,
};

function useQueryRunner() {
  const [state, setState] = useState<QState>(Q0);
  const abortRef = useRef<AbortController | null>(null);

  function cancel() {
    abortRef.current?.abort();
    setState({ ...Q0 });
  }

  async function run(body: Record<string, unknown>) {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const t0 = performance.now();
    setState({ ...Q0, loading: true });
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
        body: JSON.stringify({ ...body, debug: true }),
        signal: abortRef.current?.signal,
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          try {
            const evt = JSON.parse(line) as Record<string, unknown>;
            if (evt.type === "progress") {
              const msg = String(evt.msg);
              setState((s) => ({ ...s, currentStep: msg, log: [...s.log, { t: (performance.now() - t0) / 1000, msg }] }));
            } else if (evt.type === "image") {
              setState((s) => ({ ...s, loading: false, textureUrl: String(evt.dataUrl), ms: Math.round(performance.now() - t0), currentStep: "" }));
            } else if (evt.type === "image_face") {
              const fi = evt.face as number;
              setState((s) => {
                const faces = [...s.faces];
                if (fi >= 0 && fi < 6) faces[fi] = String(evt.dataUrl);
                const allLoaded = faces.filter(Boolean).length >= 6;
                return { ...s, faces, loading: !allLoaded, ms: allLoaded ? Math.round(performance.now() - t0) : s.ms, currentStep: allLoaded ? "" : s.currentStep };
              });
            } else if (evt.type === "image_depth") {
              const slice: DepthSlice = { idx: evt.idx as number, level: evt.level as number, meters: evt.meters as number, dataUrl: String(evt.dataUrl) };
              setState((s) => {
                const merged = [...s.depthSlices.filter((d) => d.idx !== slice.idx), slice].sort((a, b) => a.idx - b.idx);
                return { ...s, depthSlices: merged };
              });
            } else if (evt.type === "image_frame") {
              const fi = evt.frame_idx as number;
              const tot = evt.total_frames as number;
              setState((s) => {
                const frames = [...s.timeFrames];
                frames[fi] = { dataUrl: String(evt.dataUrl), label: String(evt.t_label) };
                const good = frames.filter(Boolean) as AnimFrame[];
                const done = good.length >= tot;
                return { ...s, timeFrames: good, totalExpected: tot, loading: !done, ms: done ? Math.round(performance.now() - t0) : s.ms, currentStep: done ? "" : `Frame ${fi + 1}/${tot}` };
              });
            } else if (evt.type === "error") {
              setState((s) => ({ ...s, loading: false, error: String(evt.msg), currentStep: "" }));
            }
          } catch { /* ignore malformed lines */ }
        }
      }
      setState((s) => ({ ...s, loading: false }));
    } catch (e) {
      setState((s) => ({ ...s, loading: false, error: e instanceof Error ? e.message : String(e), currentStep: "" }));
    }
  }

  return { state, run, cancel };
}

// ── Color legend ───────────────────────────────────────────────────────────────

function ColorLegend({ variable }: { variable: string | null }) {
  const meta = variable ? VAR_META[variable] : null;
  if (!meta) return null;

  const { vmin, vmax, unit, cmap } = meta;
  const BAR_H = 320;
  const N = 6;
  const ticks = Array.from({ length: N }, (_, i) => vmax - (vmax - vmin) * (i / (N - 1)));

  function fmt(v: number) {
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
    return v % 1 === 0 ? `${v}` : v.toFixed(1);
  }

  return (
    <div className="absolute left-5 bottom-[88px] z-10 flex gap-2.5 items-start select-none pointer-events-none">
      <div className="w-5 rounded-sm flex-shrink-0 ring-1 ring-white/10" style={{ height: BAR_H, background: cmapGradient(cmap) }} />
      <div className="relative flex-shrink-0" style={{ height: BAR_H + 26 }}>
        {ticks.map((v, i) => (
          <div key={i} className="absolute flex items-center gap-1.5" style={{ top: Math.round((i / (N - 1)) * BAR_H) - 7 }}>
            <div className="w-2 h-[1px] bg-white/30" />
            <span className="font-mono text-[13px] text-white/75 leading-none tabular-nums">{fmt(v)}</span>
          </div>
        ))}
        <div className="absolute flex items-center" style={{ top: BAR_H + 8 }}>
          <span className="font-mono text-[13px] font-semibold text-white/55">{unit}</span>
        </div>
      </div>
    </div>
  );
}

// ── Timeline label helpers ─────────────────────────────────────────────────────
// Backend emits labels like "2020-Jan-20 00:00Z" (DYAMOND/ECCO) or "2064-Mar-15" (CMIP6).

function parseLabel(raw: string): { tick: string; full: string } {
  const m = raw.match(/^(\d{4})-([A-Za-z]{3})-(\d{2})(?:\s+(\d{2}:\d{2}))?/);
  if (m) {
    const [, year, mon, day, time] = m;
    return {
      tick: `${mon} '${year.slice(2)}`,
      full: time ? `${day} ${mon} ${year} ${time}` : `${day} ${mon} ${year}`,
    };
  }
  return { tick: raw, full: raw };
}

// ── Timeline bar ───────────────────────────────────────────────────────────────

interface TimelineBarProps {
  frames: AnimFrame[];
  loading: boolean;
  totalExpected: number;
  onCurrentUrl: (url: string) => void;
  viewMode: ViewMode;
  setViewMode: (m: ViewMode) => void;
  right: number;
}

function TimelineBar({ frames, loading, totalExpected, onCurrentUrl, viewMode, setViewMode, right }: TimelineBarProps) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const didAutoStart = useRef(false);
  const trackRef = useRef<HTMLDivElement>(null);

  const n = frames.length;
  const safeIdx = n > 0 ? Math.min(idx, n - 1) : 0;
  const current = frames[safeIdx];

  useEffect(() => {
    if (!didAutoStart.current && n >= 3) {
      didAutoStart.current = true;
      setPlaying(true);
    }
  }, [n]);

  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (!playing || n < 2) return;
    timerRef.current = setInterval(() => setIdx((i) => (i + 1) % n), 200);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, n]);

  useEffect(() => {
    if (current?.dataUrl) onCurrentUrl(current.dataUrl);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.dataUrl]);

  function handleTrackClick(e: React.MouseEvent<HTMLDivElement>) {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect || n < 2) return;
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setPlaying(false);
    setIdx(Math.round(ratio * (n - 1)));
  }

  const pct = n > 1 ? (safeIdx / (n - 1)) * 100 : 0;
  const tickStep = Math.max(1, Math.ceil(n / 12));

  const ctrlBtn = "flex h-7 w-7 items-center justify-center rounded-full border border-white/[0.12] bg-white/[0.05] font-mono text-[11px] text-white/55 hover:bg-white/[0.12] disabled:opacity-30 transition-colors";
  const modeBtn = (active: boolean) =>
    cn("px-2 py-[3px] font-mono text-[10px] font-bold leading-none border transition-colors",
      active ? "bg-white/15 border-white/35 text-white/80" : "bg-transparent border-white/10 text-white/28 hover:border-white/25 hover:text-white/50");

  return (
    <div
      className="absolute bottom-4 left-4 z-10 flex items-center gap-3 rounded-xl border border-white/[0.08] bg-black/60 px-4 py-2.5 backdrop-blur-md"
      style={{ right: right + 16 }}
    >
      {/* Playback controls */}
      <div className="flex items-center gap-1 shrink-0">
        <button type="button" onClick={() => { setPlaying(false); setIdx(0); }} disabled={n < 2} className={ctrlBtn}>⏮</button>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          disabled={n < 2}
          className="flex h-8 w-8 items-center justify-center rounded-full border border-white/[0.14] bg-white/[0.08] font-mono text-[13px] text-white/65 hover:bg-white/[0.16] disabled:opacity-30 transition-colors"
        >
          {playing ? "⏸" : "▶"}
        </button>
        <button type="button" onClick={() => { setPlaying(false); setIdx(n - 1); }} disabled={n < 2} className={ctrlBtn}>⏭</button>
      </div>

      {/* Track */}
      {n >= 2 ? (
        <div ref={trackRef} className="flex-1 relative py-4 cursor-pointer select-none min-w-0" onClick={handleTrackClick}>
          <div className="absolute inset-x-0 top-1/2 h-[2px] -translate-y-1/2 rounded-full bg-white/[0.10]" />
          <div
            className="absolute top-1/2 left-0 h-[2px] -translate-y-1/2 rounded-full bg-[#00c4a8]/65 transition-[width]"
            style={{ width: `${pct}%` }}
          />
          {loading && n < (totalExpected || 1) && (
            <div
              className="absolute top-1/2 h-[2px] -translate-y-1/2 rounded-full bg-amber-400/30 animate-pulse"
              style={{ left: `${pct}%`, right: 0 }}
            />
          )}
          {frames.map((f, i) => {
            if (i % tickStep !== 0) return null;
            const x = (i / (n - 1)) * 100;
            return (
              <div
                key={i}
                className="absolute flex flex-col items-center pointer-events-none"
                style={{ left: `${x}%`, transform: "translateX(-50%)", bottom: 2 }}
              >
                <div className="w-[1px] h-1.5 bg-white/20" />
                <span className="font-mono text-[8px] text-white/30 mt-0.5 whitespace-nowrap">
                  {parseLabel(f.label).tick}
                </span>
              </div>
            );
          })}
          <div
            className="absolute top-1/2 -translate-y-1/2 h-3.5 w-3.5 rounded-full bg-white border-[1.5px] border-[#00c4a8] shadow pointer-events-none"
            style={{ left: `calc(${pct}% - 7px)` }}
          />
        </div>
      ) : (
        <div className="flex-1 flex items-center px-1">
          <div className="w-full h-[2px] rounded-full bg-white/[0.06]" />
        </div>
      )}

      {/* Current frame label */}
      <span className="font-mono text-[11px] text-white/55 shrink-0 text-right tabular-nums" style={{ minWidth: 88 }}>
        {current ? parseLabel(current.label).full : (loading ? "…" : "—")}
      </span>

      <div className="h-4 w-[1px] bg-white/[0.08] shrink-0" />

      {/* 3D / 2D */}
      <div className="flex shrink-0">
        <button onClick={() => setViewMode("3d")} className={cn(modeBtn(viewMode === "3d"), "rounded-l-sm rounded-r-none border-r-0")}>3D</button>
        <button onClick={() => setViewMode("2d")} className={cn(modeBtn(viewMode === "2d"), "rounded-r-sm rounded-l-none")}>2D</button>
      </div>
    </div>
  );
}

// ── Progress log panel ───────────────────────────────────────────────────────────

// Pull the most informative line for a compact one-line summary: the backend
// emits "  Steps : N / M total (..)" and "  Est. time : ~Ns" during planning.
function logSummary(log: LogLine[]): string | null {
  for (let i = log.length - 1; i >= 0; i--) {
    const m = log[i].msg;
    if (/Steps\s*:/.test(m) || /Subsampled/.test(m) || /Est\. time/.test(m)
        || /^Done/.test(m) || /^Requesting/.test(m) || /^Fetching/.test(m)) return m.trim();
  }
  return null;
}

function ProgressLogPanel({ log, loading }: { log: LogLine[]; loading: boolean }) {
  const [collapsed, setCollapsed] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!collapsed && scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [log.length, collapsed]);

  if (log.length === 0 && !loading) return null;
  const summary = logSummary(log);

  return (
    <div className="absolute top-[5.25rem] left-4 z-10 w-[360px] max-w-[calc(100vw-2rem)] rounded-xl border border-white/[0.08] bg-black/60 backdrop-blur-md">
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <div className="flex items-center gap-2 min-w-0">
          {loading && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-amber-400" />}
          <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-white/45">Progress</span>
          {summary && collapsed && (
            <span className="truncate font-mono text-[10px] text-white/30">{summary}</span>
          )}
        </div>
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          className="shrink-0 font-mono text-[10px] text-white/35 hover:text-white/70"
        >
          {collapsed ? "Show" : "Hide"} ({log.length})
        </button>
      </div>
      {!collapsed && (
        <div
          ref={scrollRef}
          className="max-h-56 overflow-auto border-t border-white/[0.06] px-3 py-2 font-mono text-[10.5px] leading-relaxed text-white/45"
        >
          {log.map((l, i) => (
            <div key={i} className="whitespace-pre-wrap">
              <span className="text-[#00c4a8]/55">[+{l.t.toFixed(2)}s]</span> {l.msg}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Vertical query form ────────────────────────────────────────────────────────

type DatasetTab = "dyamond" | "ecco" | "cmip6";
type DyamondMode = "atmosphere" | "ocean" | "coupled";

function VerticalQueryForm({ run, loading, onActiveVar }: {
  run: (b: Record<string, unknown>) => void;
  loading: boolean;
  onActiveVar: (variable: string) => void;
}) {
  const [ds, setDs] = useState<DatasetTab>("dyamond");
  const [dMode, setDMode] = useState<DyamondMode>("atmosphere");

  // DYAMOND atmosphere — 30-min steps from 2020-01-20; date→index: (date-epoch)/1800000
  const [dVar,       setDVar]       = useState("u");
  const [dFace,      setDFace]      = useState(0);
  const [dZ,         setDZ]         = useState(40);
  const [dDateStart, setDDateStart] = useState("2020-03-01");
  const [dDateEnd,   setDDateEnd]   = useState("2020-03-15");
  const [dFrames,    setDFrames]    = useState(8);
  const [dQual,      setDQual]      = useState(-7);

  // DYAMOND ocean — hourly steps, same calendar as ECCO (Sep 2011–Nov 2012)
  const [doVar,       setDoVar]       = useState("theta");
  const [doDepth,     setDoDepth]     = useState(0);
  const [doDateStart, setDoDateStart] = useState("2011-09-13");
  const [doDateEnd,   setDoDateEnd]   = useState("2012-01-01");
  const [doFrames,    setDoFrames]    = useState(8);
  const [doQual,      setDoQual]      = useState(-12);

  // ECCO — hourly steps from 2011-09-13
  const [eVar,       setEVar]       = useState("theta");
  const [eDepth,     setEDepth]     = useState(0);
  const [eDateStart, setEDateStart] = useState("2011-09-13");
  const [eDateEnd,   setEDateEnd]   = useState("2012-01-01");
  const [eFrames,    setEFrames]    = useState(8);
  const [eQual,      setEQual]      = useState(-12);

  // CMIP6 — daily steps from 1950-01-01
  const [cVar,      setCVar]      = useState("tas");
  const [cModel,    setCModel]    = useState("ACCESS-CM2");
  const [cScenario, setCScenario] = useState("ssp585");
  const [cYearStart, setCYearStart] = useState(2015);
  const [cYearEnd,   setCYearEnd]   = useState(2020);
  const [cFrames,   setCFrames]   = useState(8);

  // Convert human dates/years → positional array indices (matching _subsample)
  const DYM_EPOCH  = Date.UTC(2020, 0, 20); // Jan 20, 2020
  const ECCO_EPOCH = Date.UTC(2011, 8, 13); // Sep 13, 2011
  const dymDateToT  = (d: string) => Math.max(0, Math.round((new Date(d).getTime() - DYM_EPOCH)  / 1_800_000));
  const eccoDateToT = (d: string) => Math.max(0, Math.round((new Date(d).getTime() - ECCO_EPOCH) / 3_600_000));
  const yearToT     = (y: number) => Math.max(0, Math.round((y - 1950) * 365.25));

  function handleRender() {
    if (ds === "dyamond") {
      if (dMode === "ocean") {
        onActiveVar(doVar);
        run({ dataset: "dyamond", ocean: true, variable: doVar, depth_level: doDepth,
              t_start: eccoDateToT(doDateStart), t_end: eccoDateToT(doDateEnd),
              quality: doQual, animate: true, n_frames: doFrames, render_mode: "globe" });
      } else if (dMode === "coupled") {
        onActiveVar(dVar);
        run({ dataset: "dyamond", coupled: true,
              atm_variable: dVar, ocn_variable: doVar,
              t_atm: dymDateToT(dDateStart), t_ocn: eccoDateToT(doDateStart),
              z_level_atm: dZ, depth_level_ocn: doDepth,
              quality: dQual });
      } else {
        onActiveVar(dVar);
        const t0 = dymDateToT(dDateStart), t1 = dymDateToT(dDateEnd);
        run({ dataset: "dyamond", variable: dVar, face: dFace, t_start: t0, t_end: t1,
              z_level: dZ, quality: dQual, animate: t1 > t0, n_frames: dFrames, render_mode: "globe" });
      }
    } else if (ds === "ecco") {
      onActiveVar(eVar);
      run({ dataset: "ecco", variable: eVar, depth_level: eDepth,
            t_start: eccoDateToT(eDateStart), t_end: eccoDateToT(eDateEnd),
            quality: eQual, animate: true, n_frames: eFrames, render_mode: "globe" });
    } else {
      onActiveVar(cVar);
      run({ dataset: "cmip6", variable: cVar, model: cModel, scenario: cScenario,
            t_start: yearToT(cYearStart), t_end: yearToT(cYearEnd),
            animate: true, n_frames: cFrames, render_mode: "globe" });
    }
  }

  const inp = "w-full bg-[#080f1c] text-white/65 border border-white/[0.08] rounded px-2.5 py-2 text-[13px] font-mono outline-none focus:border-white/20";
  const lbl = "font-mono text-[11px] uppercase tracking-widest text-white/35 mb-0.5";
  const field = "space-y-1";

  // Quality select — more-negative = COARSER (less data, faster).
  // q=0 is full resolution; each -1 step halves each spatial axis.
  // DYAMOND atm is capped at -9 by the backend IDX pyramid.
  function QualitySelect({ value, onChange, min = -15 }: { value: number; onChange: (v: number) => void; min?: number }) {
    const opts: { v: number; label: string }[] = [
      { v: -1,  label: "-1  Finest  (most data)" },
      { v: -3,  label: "-3  Very high" },
      { v: -5,  label: "-5  High" },
      { v: -7,  label: "-7  Medium" },
      { v: -9,  label: "-9  Low" },
      { v: -12, label: "-12  Very low" },
      { v: -15, label: "-15  Coarsest  (least data)" },
    ].filter(o => o.v >= min);
    return (
      <select
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className={inp}
      >
        {opts.map(o => <option key={o.v} value={o.v}>{o.label}</option>)}
      </select>
    );
  }

  const dsBtnCls = (active: boolean) =>
    cn("flex-1 py-1.5 text-[11px] font-mono font-bold uppercase tracking-widest transition-colors border-b-2",
      active ? "border-[#00c4a8] text-white/70" : "border-transparent text-white/22 hover:text-white/45");

  return (
    <div className="space-y-3 px-3 pb-3">
      <div className="flex border-b border-white/[0.06]">
        {(["dyamond", "ecco", "cmip6"] as DatasetTab[]).map((d) => (
          <button key={d} type="button" onClick={() => setDs(d)} className={dsBtnCls(ds === d)}>
            {d === "dyamond" ? "DYAMOND" : d === "ecco" ? "ECCO" : "CMIP6"}
          </button>
        ))}
      </div>

      {ds === "dyamond" && (
        <div className="space-y-2">
          {/* Atmosphere / Ocean / Coupled selector */}
          <div className={field}>
            <p className={lbl}>Component</p>
            <div className="flex rounded overflow-hidden border border-white/[0.08]">
              {(["atmosphere", "ocean", "coupled"] as DyamondMode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setDMode(m)}
                  className={cn(
                    "flex-1 py-1.5 font-mono text-[10px] font-bold uppercase tracking-wider transition-colors",
                    dMode === m ? "bg-[#4a9eff]/20 text-[#4a9eff]" : "text-white/28 hover:text-white/50",
                  )}
                >
                  {m === "atmosphere" ? "Atm" : m === "ocean" ? "Ocn" : "Coupled"}
                </button>
              ))}
            </div>
          </div>

          {/* Atmosphere / Coupled: GEOS params */}
          {(dMode === "atmosphere" || dMode === "coupled") && (
            <>
              {dMode === "coupled" && (
                <p className="font-mono text-[10px] text-white/30 leading-tight">
                  Single snapshot: ATM (GEOS) + OCN (LLC2160). Ocean variable/depth from OCN tab.
                </p>
              )}
              <div className={field}>
                <p className={lbl}>{dMode === "coupled" ? "ATM variable" : "Variable"}</p>
                <select value={dVar} onChange={(e) => setDVar(e.target.value)} className={inp}>
                  <option value="u">u – zonal wind</option>
                  <option value="v">v – meridional wind</option>
                  <option value="w">w – vertical velocity</option>
                  <option value="t">t – air temperature</option>
                  <option value="p">p – pressure</option>
                  <option value="h">h – layer heights</option>
                  <option value="fcld">fcld – cloud fraction</option>
                  <option value="qi">qi – cloud ice water</option>
                  <option value="ql">ql – cloud liquid water</option>
                  <option value="ri">ri – ice particle radius</option>
                  <option value="rl">rl – liquid particle radius</option>
                  <option value="co">co – carbon monoxide</option>
                  <option value="co2">co2 – carbon dioxide</option>
                  <option value="delp">delp – pressure thickness</option>
                  <option value="dthdt">dthdt – θ tendency (moist)</option>
                  <option value="dthdtcn">dthdtcn – θ tendency (convection)</option>
                </select>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className={field}>
                  <p className={lbl}>Face</p>
                  <input type="number" min={0} max={5} value={dFace} onChange={(e) => setDFace(Number(e.target.value))} className={inp} />
                </div>
                <div className={field}>
                  <p className={lbl}>Z level</p>
                  <input type="number" min={0} max={51} value={dZ} onChange={(e) => setDZ(Number(e.target.value))} className={inp} />
                </div>
                <div className={field}>
                  <p className={lbl}>Quality</p>
                  <QualitySelect value={dQual} onChange={setDQual} min={-9} />
                </div>
              </div>
              {dMode === "atmosphere" ? (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <div className={field}>
                      <p className={lbl}>Start date</p>
                      <input type="date" min="2020-01-20" max="2020-08-19" value={dDateStart} onChange={(e) => setDDateStart(e.target.value)} className={inp} />
                    </div>
                    <div className={field}>
                      <p className={lbl}>End date</p>
                      <input type="date" min="2020-01-20" max="2020-08-19" value={dDateEnd} onChange={(e) => setDDateEnd(e.target.value)} className={inp} />
                    </div>
                  </div>
                  <div className={field}>
                    <p className={lbl}>Frames</p>
                    <input type="number" min={1} max={30} value={dFrames} onChange={(e) => setDFrames(Number(e.target.value))} className={inp} />
                  </div>
                </>
              ) : (
                <div className={field}>
                  <p className={lbl}>ATM date</p>
                  <input type="date" min="2020-01-20" max="2020-08-19" value={dDateStart} onChange={(e) => setDDateStart(e.target.value)} className={inp} />
                </div>
              )}
            </>
          )}

          {/* Ocean: LLC2160 params */}
          {dMode === "ocean" && (
            <>
              <div className={field}>
                <p className={lbl}>Variable</p>
                <select value={doVar} onChange={(e) => setDoVar(e.target.value)} className={inp}>
                  <option value="theta">theta – temperature</option>
                  <option value="salt">salt – salinity</option>
                  <option value="u">u – zonal vel</option>
                  <option value="v">v – merid vel</option>
                  <option value="w">w – vert vel</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className={field}>
                  <p className={lbl}>Depth lvl</p>
                  <input type="number" min={0} max={89} value={doDepth} onChange={(e) => setDoDepth(Number(e.target.value))} className={inp} />
                </div>
                <div className={field}>
                  <p className={lbl}>Quality</p>
                  <QualitySelect value={doQual} onChange={setDoQual} min={-15} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className={field}>
                  <p className={lbl}>Start date</p>
                  <input type="date" min="2011-09-13" max="2012-11-22" value={doDateStart} onChange={(e) => setDoDateStart(e.target.value)} className={inp} />
                </div>
                <div className={field}>
                  <p className={lbl}>End date</p>
                  <input type="date" min="2011-09-13" max="2012-11-22" value={doDateEnd} onChange={(e) => setDoDateEnd(e.target.value)} className={inp} />
                </div>
              </div>
              <div className={field}>
                <p className={lbl}>Frames</p>
                <input type="number" min={1} max={30} value={doFrames} onChange={(e) => setDoFrames(Number(e.target.value))} className={inp} />
              </div>
            </>
          )}
        </div>
      )}

      {ds === "ecco" && (
        <div className="space-y-2">
          <div className={field}>
            <p className={lbl}>Variable</p>
            <select value={eVar} onChange={(e) => setEVar(e.target.value)} className={inp}>
              <option value="theta">theta – potential temp</option>
              <option value="salt">salt – salinity</option>
            </select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className={field}>
              <p className={lbl}>Depth lvl</p>
              <input type="number" min={0} max={89} value={eDepth} onChange={(e) => setEDepth(Number(e.target.value))} className={inp} />
            </div>
            <div className={field}>
              <p className={lbl}>Quality</p>
              <QualitySelect value={eQual} onChange={setEQual} min={-15} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className={field}>
              <p className={lbl}>Start date</p>
              <input type="date" min="2011-09-13" max="2012-11-22" value={eDateStart} onChange={(e) => setEDateStart(e.target.value)} className={inp} />
            </div>
            <div className={field}>
              <p className={lbl}>End date</p>
              <input type="date" min="2011-09-13" max="2012-11-22" value={eDateEnd} onChange={(e) => setEDateEnd(e.target.value)} className={inp} />
            </div>
          </div>
          <div className={field}>
            <p className={lbl}>Frames</p>
            <input type="number" min={1} max={30} value={eFrames} onChange={(e) => setEFrames(Number(e.target.value))} className={inp} />
          </div>
        </div>
      )}

      {ds === "cmip6" && (
        <div className="space-y-2">
          <div className={field}>
            <p className={lbl}>Variable</p>
            <select value={cVar} onChange={(e) => setCVar(e.target.value)} className={inp}>
              <option value="tas">tas – air temperature</option>
              <option value="tasmax">tasmax – daily max temp</option>
              <option value="pr">pr – precipitation</option>
              <option value="hurs">hurs – relative humidity</option>
              <option value="sfcWind">sfcWind – surface wind</option>
            </select>
          </div>
          <div className={field}>
            <p className={lbl}>Model</p>
            <select value={cModel} onChange={(e) => setCModel(e.target.value)} className={inp}>
              {["ACCESS-CM2","CanESM5","CESM2","CMCC-CM2-SR5","EC-Earth3","GFDL-ESM4","INM-CM5-0","IPSL-CM6A-LR","MIROC6","MPI-ESM1-2-HR","MRI-ESM2-0"].map((m) =>
                <option key={m} value={m}>{m}</option>
              )}
            </select>
          </div>
          <div className={field}>
            <p className={lbl}>Scenario</p>
            <select value={cScenario} onChange={(e) => setCScenario(e.target.value)} className={inp}>
              <option value="historical">historical (1950–2014)</option>
              <option value="ssp245">ssp245 – moderate</option>
              <option value="ssp585">ssp585 – high emissions</option>
            </select>
          </div>
          <p className="font-mono text-[10px] text-white/25 leading-tight">
            Quality N/A — fixed 1440×600 grid, no resolution pyramid.
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div className={field}>
              <p className={lbl}>Year start</p>
              <input type="number" min={1950} max={2099} value={cYearStart} onChange={(e) => setCYearStart(Number(e.target.value))} className={inp} />
            </div>
            <div className={field}>
              <p className={lbl}>Year end</p>
              <input type="number" min={1950} max={2099} value={cYearEnd} onChange={(e) => setCYearEnd(Number(e.target.value))} className={inp} />
            </div>
          </div>
          <div className={field}>
            <p className={lbl}>Frames</p>
            <input type="number" min={1} max={30} value={cFrames} onChange={(e) => setCFrames(Number(e.target.value))} className={inp} />
          </div>
        </div>
      )}

      {/* ── Fetch size estimate ─────────────────────────────────────── */}
      {(() => {
        // frame_kb(q) = ref_kb × 4^(q − ref_q)  (more-negative q → coarser → less data)
        // References are empirical measurements of actual network bytes.
        let frameKb: number;
        let nFetch: number;
        let detail: string;

        if (ds === "dyamond") {
          if (dMode === "ocean") {
            // ECCO fallback or LLC2160 — same grid reference as ECCO
            frameKb = 3417 * Math.pow(4, doQual - (-9));
            nFetch = doFrames;
            detail = `q=${doQual}`;
          } else {
            // GEOS cubed-sphere: 6 faces × ~480 KB/face at q=−7
            frameKb = 6 * 480 * Math.pow(4, dQual - (-7));
            nFetch = dFrames;
            detail = `q=${dQual} · 6 faces`;
          }
        } else if (ds === "ecco") {
          frameKb = 3417 * Math.pow(4, eQual - (-9));
          nFetch = eFrames;
          detail = `q=${eQual}`;
        } else {
          // CMIP6: fixed 1440×600 float32 grid, no quality scaling
          frameKb = 3400;
          nFetch = cFrames;
          detail = "1440×600 · no quality";
        }

        const totalMb = (nFetch * frameKb) / 1024;
        const isLarge = totalMb > 500;
        const isHuge  = totalMb > 5000;

        const fmtSize = (mb: number) =>
          mb < 1    ? `${(mb * 1024).toFixed(0)} KB`
          : mb < 1024 ? `${mb.toFixed(mb < 10 ? 1 : 0)} MB`
          : `${(mb / 1024).toFixed((mb / 1024) < 10 ? 1 : 0)} GB`;

        return (
          <div className={cn(
            "rounded border px-3 py-2 font-mono text-[10.5px]",
            isHuge  ? "border-red-500/40 bg-red-500/5"
            : isLarge ? "border-amber-500/30 bg-amber-500/5"
            : "border-white/[0.07] bg-white/[0.02]",
          )}>
            <p className="text-[9px] font-bold uppercase tracking-widest text-white/28 mb-1.5">Est. fetch</p>
            <p className="text-white/45">
              {nFetch} frames × {fmtSize(frameKb / 1024)}/frame
              <span className="text-white/22 ml-1.5">({detail})</span>
            </p>
            <p className={cn(
              "mt-0.5 font-semibold text-[12px]",
              isHuge ? "text-red-400" : isLarge ? "text-amber-400" : "text-white/65",
            )}>
              ≈ {fmtSize(totalMb)}
              {isHuge  && <span className="ml-1.5 text-[10px] font-normal">⚠ very large</span>}
              {!isHuge && isLarge && <span className="ml-1.5 text-[10px] font-normal text-amber-400/70">— large fetch</span>}
            </p>
          </div>
        );
      })()}

      <button
        onClick={handleRender}
        disabled={loading}
        className="w-full rounded-lg border border-[#00c4a8]/30 bg-[#00c4a8]/15 py-2.5 font-mono text-[13px] font-bold uppercase tracking-widest text-[#00c4a8] transition-all hover:bg-[#00c4a8]/25 disabled:opacity-40"
      >
        {loading ? "Rendering…" : "▶ Render"}
      </button>
    </div>
  );
}

// ── Master Dashboard ───────────────────────────────────────────────────────────

const PANEL_W = 300;
// Clears the bottom-4 playback/timeline bar (~54px tall) that overlaps the globe's bottom edge.
const VR_BUTTON_BOTTOM = 84;

export function MasterDashboard() {
  const { state, run, cancel } = useQueryRunner();
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [globeUrl, setGlobeUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<"presets" | "custom">("presets");
  const [panelOpen, setPanelOpen] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("3d");
  const [activeVar, setActiveVar] = useState<string | null>(null);
  // GPU ocean mode: bypass the Python analysis pipeline for ECCO presets
  const [gpuMode, setGpuMode] = useState(true);
  const [gpuView, setGpuView] = useState<"globe" | "slab">("globe");
  const globeState = useGlobeData(gpuMode && gpuView === "globe");

  const hasFrames  = state.timeFrames.length > 0;
  const hasFaces   = state.faces.some(Boolean);
  const hasDepth   = state.depthSlices.length > 0;
  const hasAnyData = hasFrames || hasFaces || hasDepth || !!state.textureUrl;
  const globeTexture = hasFrames ? globeUrl : state.textureUrl;
  const panelRight = panelOpen ? PANEL_W : 0;

  const activeP = PRESETS.find((p) => p.id === activePreset);

  return (
    <div className="relative overflow-hidden bg-[#060b16]" style={{ height: "100vh", marginTop: "-3rem" }}>

      {/* ── Full-screen globe ─────────────────────────────────────────── */}
      <div className="absolute inset-0">
        {gpuMode && viewMode === "3d" ? (
          gpuView === "slab" ? <VolumeViewer /> : (
            <OceanGlobeViewer state={globeState} vrButtonRight={panelRight + 12} vrButtonBottom={VR_BUTTON_BOTTOM} />
          )
        ) : viewMode === "2d" ? (
          <div className="absolute inset-0 flex items-center justify-center bg-[#060b16]">
            {globeTexture ? (
              <img src={globeTexture} alt="2D projection" className="w-full h-full object-contain" />
            ) : hasFaces ? (
              <img src={state.faces.find((f) => f !== null) ?? ""} alt="face" className="w-full h-full object-contain" />
            ) : (
              <span className="font-mono text-[13px] text-white/15">No data loaded</span>
            )}
          </div>
        ) : hasDepth ? (
          <DepthStackViewer
            slices={state.depthSlices}
            altText="Stacked ocean depth slabs"
            vrButtonRight={panelRight + 12}
            vrButtonBottom={VR_BUTTON_BOTTOM}
          />
        ) : hasFaces ? (
          <CubeViewer faces={state.faces} altText="DYAMOND cubed-sphere globe" />
        ) : (
          <GlobeViewer
            textureUrl={globeTexture}
            altText="Master dashboard globe"
            vrButtonRight={panelRight + 12}
            vrButtonBottom={VR_BUTTON_BOTTOM}
          />
        )}
      </div>

      {/* ── Top gradient ──────────────────────────────────────────────── */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-black/80 via-black/25 to-transparent" />

      {/* ── Bottom gradient ───────────────────────────────────────────── */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/60 to-transparent" />

      {/* ── Top-left: logo + status ───────────────────────────────────── */}
      <div className="absolute top-12 left-4 z-10 flex items-center gap-3 flex-wrap">
        <span className="font-mono text-[13px] font-bold uppercase tracking-[0.2em] text-white/55">SciViz 2026</span>
        <div className="flex items-center gap-1.5 rounded-full border border-white/[0.07] bg-black/40 px-3 py-1.5 backdrop-blur-sm">
          <span className={cn(
            "h-2 w-2 rounded-full",
            state.loading ? "animate-pulse bg-amber-400" : state.error ? "bg-red-500" : "bg-[#00c4a8]"
          )} />
          <span className="font-mono text-[11px] font-bold uppercase tracking-widest text-white/50">
            {state.loading ? (state.currentStep || "Rendering") : state.error ? "Error" : "Live"}
          </span>
          {state.ms != null && !state.loading && (
            <span className="font-mono text-[10px] text-white/28 ml-0.5">{state.ms} ms</span>
          )}
        </div>
        {state.loading && (
          <button
            type="button"
            onClick={cancel}
            className="rounded-full border border-white/[0.10] bg-black/40 px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-widest text-white/40 backdrop-blur-sm hover:text-white/70 hover:border-white/25 transition-colors"
          >
            ✕ Cancel
          </button>
        )}
        {/* GPU / Analysis toggle (Analysis archived — see SHOW_ANALYSIS) */}
        {SHOW_ANALYSIS && (
        <div className="flex rounded-full overflow-hidden border border-white/[0.08] bg-black/40 backdrop-blur-sm">
          {([["gpu", "GPU Ocean"] as const, ["analysis", "Analysis"] as const]).map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => setGpuMode(key === "gpu")}
              className={cn(
                "px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-widest transition-colors",
                (key === "gpu") === gpuMode
                  ? "bg-[#f4a836]/20 text-[#f4a836]"
                  : "text-white/28 hover:text-white/50",
              )}
            >
              {label}
            </button>
          ))}
        </div>
        )}
        {/* Globe / Slab toggle (3D Slab archived — see SHOW_SLAB) */}
        {SHOW_SLAB && gpuMode && (
          <div className="flex rounded-full overflow-hidden border border-white/[0.08] bg-black/40 backdrop-blur-sm">
            {([["globe", "Globe"] as const, ["slab", "3D Slab"] as const]).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setGpuView(key)}
                className={cn(
                  "px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-widest transition-colors",
                  gpuView === key
                    ? "bg-[#f4a836]/20 text-[#f4a836]"
                    : "text-white/28 hover:text-white/50",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {hasAnyData && (activeP || activeVar) && (
          <div className="flex items-center gap-2 rounded-full border border-white/[0.07] bg-black/40 px-3 py-1.5 backdrop-blur-sm font-mono text-[10px] uppercase tracking-widest text-white/40">
            {activeP && <span className="text-white/55">{activeP.label}</span>}
            {activeP && <span className="text-white/20">·</span>}
            {activeP && <span>{activeP.badge}</span>}
            {activeVar && <span className="text-white/20">·</span>}
            {activeVar && <span className="text-[#00c4a8]/70">{activeVar}</span>}
          </div>
        )}
      </div>

      {/* ── Progress log ──────────────────────────────────────────────── */}
      <ProgressLogPanel
        log={gpuMode ? globeState.log : state.log}
        loading={gpuMode ? globeState.loading : state.loading}
      />

      {/* ── Color legend ──────────────────────────────────────────────── */}
      {hasAnyData && <ColorLegend variable={activeVar} />}

      {/* ── Timeline bar ──────────────────────────────────────────────── */}
      {!(gpuMode && gpuView === "globe" && (globeState.frames.length > 0 || globeState.fetchingFrames)) && (
        <TimelineBar
          frames={state.timeFrames}
          loading={state.loading}
          totalExpected={state.totalExpected}
          onCurrentUrl={setGlobeUrl}
          viewMode={viewMode}
          setViewMode={setViewMode}
          right={panelRight}
        />
      )}

      {/* ── Loading overlay ───────────────────────────────────────────── */}
      {gpuMode && globeState.loading && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="rounded-full border border-white/[0.06] bg-black/65 px-6 py-2 backdrop-blur-sm flex items-center gap-2.5">
            <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400 flex-shrink-0" />
            <span className="font-mono text-[11px] text-white/50">Fetching ocean data…</span>
          </div>
        </div>
      )}
      {!gpuMode && state.loading && state.currentStep && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="rounded-full border border-white/[0.06] bg-black/65 px-6 py-2 backdrop-blur-sm">
            <span className="font-mono text-[11px] text-white/50">{state.currentStep}</span>
          </div>
        </div>
      )}

      {/* ── Error overlay ─────────────────────────────────────────────── */}
      {(gpuMode ? globeState.error : state.error) && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 w-80 -translate-x-1/2 -translate-y-1/2 rounded-lg border border-red-900/40 bg-red-950/70 p-4 text-center backdrop-blur-sm">
          <p className="font-mono text-[11px] text-red-400 leading-relaxed">
            {gpuMode ? "Ocean backend offline" : state.error}
          </p>
          {gpuMode && (
            <p className="mt-1 font-mono text-[10px] text-red-400/60">
              Start the backend: <code>uvicorn main:app --port 8000</code> (see README)
            </p>
          )}
        </div>
      )}

      {/* ── Right side panel ──────────────────────────────────────────── */}
      <div className="absolute top-0 bottom-0 right-0 z-10 flex" style={{ width: panelOpen ? PANEL_W : 0 }}>
        <button
          onClick={() => setPanelOpen((v) => !v)}
          className="absolute -left-6 top-1/2 -translate-y-1/2 flex h-14 w-6 items-center justify-center rounded-l-lg border border-r-0 border-white/[0.08] bg-black/55 text-white/35 backdrop-blur-sm transition-colors hover:text-white/65"
        >
          <span className="font-mono text-[11px]">{panelOpen ? "›" : "‹"}</span>
        </button>

        {panelOpen && (
          <div className="flex w-full flex-col overflow-hidden border-l border-white/[0.07] bg-black/60 backdrop-blur-md">

            <div className="shrink-0 border-b border-white/[0.06] px-3 pt-14 pb-2">
              <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-white/28">Parameters</p>
              {activeP && (
                <div className="mt-2 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full" style={{ background: activeP.color }} />
                  <span className="font-mono text-[12px] text-white/50 truncate">{activeP.label}</span>
                  <span className="ml-auto font-mono text-[10px] text-white/28">{activeP.badge}</span>
                </div>
              )}
            </div>

            <div className="flex shrink-0 border-b border-white/[0.06]">
              {(["presets", "custom"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={cn(
                    "flex-1 border-b-2 py-2.5 font-mono text-[11px] font-bold uppercase tracking-widest transition-colors",
                    mode === m ? "border-[#00c4a8] text-white/65" : "border-transparent text-white/20 hover:text-white/40",
                  )}
                >
                  {m === "presets" ? "Presets" : "Custom"}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto">
              {mode === "presets" && (
                <div className="space-y-1.5 p-3">
                  {PRESETS.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setActivePreset(p.id);
                        setActiveVar(String(p.body.variable ?? ""));
                        // All presets drive the live GPU globe (the Python
                        // Analysis pipeline and the 3D Slab view are archived).
                        setGpuMode(true);
                        setGpuView("globe");
                        if (p.id === "cmip6_warming") {
                          globeState.setGroup("cmip6");
                          globeState.setVariable("cmip6_tas");
                          globeState.setCmip6Scenario("ssp585");
                          // 1 Aug 2090 in the noleap calendar: (2090−1950)·365 + 212
                          globeState.setTime((2090 - 1950) * 365 + 212);
                        } else if (p.id === "dyamond_wind") {
                          // Jet-stream altitudes (~8–15 km): GEOS levels 36–43.
                          // Passed through setGroup — a separate setLoadedRange
                          // call would clamp against the previous group's
                          // depthLevels (stale state).
                          globeState.setGroup("dyamond_atm", [36, 43]);
                          globeState.setVariable("geos_u");
                        } else if (p.id === "ecco_theta") {
                          // Sea-surface temperature — just the top layer.
                          globeState.setGroup("ecco", [0, 1]);
                          globeState.setVariable("theta");
                        } else {
                          // depth_stack "Ocean Depths" — a stack of layers so the
                          // volumetric temperature structure is visible with depth
                          // (this is what makes it distinct from Ocean SST now that
                          // the flat 3D Slab view is archived).
                          globeState.setGroup("ecco", [0, 15]);
                          globeState.setVariable("theta");
                        }
                      }}
                      disabled={state.loading}
                      className={cn(
                        "flex w-full items-center gap-2.5 rounded-lg border px-3 py-2.5 text-left transition-all",
                        activePreset === p.id
                          ? "border-white/20 bg-white/[0.09]"
                          : "border-white/[0.07] bg-white/[0.02] hover:border-white/14 hover:bg-white/[0.05]",
                        state.loading && "cursor-not-allowed opacity-40",
                      )}
                    >
                      <span className="mt-1 h-2 w-2 shrink-0 self-start rounded-full" style={{ background: p.color }} />
                      <div className="min-w-0 flex-1">
                        <p className="text-[13px] font-semibold text-white/78">{p.label}</p>
                        <p className="text-[11px] leading-snug text-white/45">{p.desc}</p>
                        <p className="mt-0.5 font-mono text-[9px] uppercase tracking-widest text-white/28">{p.badge}</p>
                      </div>
                      {activePreset === p.id && (
                        <span className="font-mono text-[8px]" style={{ color: p.color }}>●</span>
                      )}
                    </button>
                  ))}

                  <div className="mt-4 border-t border-white/[0.06] pt-3">
                    <p className="mb-2.5 font-mono text-[11px] font-bold uppercase tracking-widest text-white/25">Datasets</p>
                    <DatasetBars />
                  </div>
                </div>
              )}

              {mode === "custom" && (
                <div className="pt-3">
                  {gpuMode ? (
                    <GlobeControlsPanel state={globeState} />
                  ) : (
                    <VerticalQueryForm
                      run={run}
                      loading={state.loading}
                      onActiveVar={setActiveVar}
                    />
                  )}
                </div>
              )}
            </div>

            <div className="shrink-0 border-t border-white/[0.06] px-3 py-2">
              {gpuMode ? (
                <div className="flex items-center gap-2">
                  <span className={cn(
                    "h-1.5 w-1.5 rounded-full flex-shrink-0",
                    globeState.loading ? "animate-pulse bg-amber-400" : globeState.error ? "bg-red-500" : "bg-[#00c4a8]",
                  )} />
                  <span className="font-mono text-[11px] text-white/40 truncate">
                    {globeState.loading ? "Fetching ocean data…" : globeState.error ? "Backend offline — start port 8000" : "Live GPU rendering"}
                  </span>
                </div>
              ) : state.error ? (
                <p className="font-mono text-[11px] text-red-400 leading-relaxed truncate">{state.error}</p>
              ) : state.loading ? (
                <div className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
                  <span className="font-mono text-[11px] text-white/40 truncate">{state.currentStep || "Processing…"}</span>
                </div>
              ) : state.ms != null ? (
                <div className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#00c4a8]" />
                  <span className="font-mono text-[11px] text-white/40">{state.ms} ms</span>
                  {hasFrames && <span className="font-mono text-[11px] text-white/28 ml-1">· {state.timeFrames.length} frames</span>}
                </div>
              ) : (
                <p className="font-mono text-[11px] text-white/25">Select a preset to render</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
