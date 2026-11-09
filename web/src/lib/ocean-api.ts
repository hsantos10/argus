/**
 * API client for the ocean FastAPI backend (port 8000).
 * This is the other team's backend that serves raw binary LLC4320 data
 * for GPU-side rendering — distinct from our /api/query analysis pipeline.
 */

// Same-origin path, proxied server-side to the backend by the Next.js rewrite
// in next.config.ts. This avoids CORS and mixed-content issues entirely — a
// VR headset browsing https://<lan-ip>:3000 never talks to port 8000 itself;
// Next.js's own server forwards the request to http://127.0.0.1:8000.
export const OCEAN_BASE = "/ocean-api";

export type OceanDatasetInfo = {
  key: string;
  label: string;
  url: string;
  group?: string;        // "ecco" | "dyamond_ocean" | "dyamond_atm"
  grid?: string;         // "llc" | "cubed"
  depth_levels?: number;
};

export type OceanDatasetMetadata = {
  key: string;
  label: string;
  url: string;
  dims: [number, number, number];
  timesteps: number[];
  default_quality: number;
  time_info?: {
    t0_iso: string;
    dt_hours: number;
    n_timesteps: number;
  };
};

export type VolumePayload = {
  data: Float32Array;
  shape: [number, number, number];
  min: number;
  max: number;
};

export type GlobeVolumePayload = {
  data: Uint8Array;
  depth: number;
  height: number;
  width: number;
  min: number;
  max: number;
  z0: number;
  z1: number;
};

export type GlobeVolumeQuery = {
  time?: number;
  z0?: number;
  z1?: number;
  quality?: number;
  norm_min?: number;
  norm_max?: number;
  // CMIP6 experiment (historical | ssp126 | ssp245 | ssp370 | ssp585);
  // ignored by non-CMIP6 datasets.
  scenario?: string;
};

export type VolumeQuery = {
  time?: number;
  quality?: number;
  x0?: number; x1?: number;
  y0?: number; y1?: number;
  z0?: number; z1?: number;
};

function buildParams(q: Record<string, unknown>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) {
    if (v !== undefined && v !== null) p.set(k, String(v));
  }
  return p.toString();
}

export async function oceanListDatasets(): Promise<OceanDatasetInfo[]> {
  const r = await fetch(`${OCEAN_BASE}/datasets`);
  if (!r.ok) throw new Error(`ocean /datasets ${r.status}`);
  return r.json();
}

export async function oceanGetMetadata(key: string): Promise<OceanDatasetMetadata> {
  const r = await fetch(`${OCEAN_BASE}/metadata/${key}`);
  if (!r.ok) throw new Error(`ocean /metadata/${key} ${r.status}`);
  return r.json();
}

export async function oceanFetchLatLUT(
  grid: string = "llc4320",
  latBounds?: [number, number], // partial-coverage grids (e.g. CMIP6: −60…90)
): Promise<Float32Array> {
  const bounds = latBounds ? `&lat0=${latBounds[0]}&lat1=${latBounds[1]}` : "";
  const r = await fetch(`${OCEAN_BASE}/latlut?grid=${encodeURIComponent(grid)}${bounds}`);
  if (!r.ok) throw new Error(`ocean /latlut ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}

export async function oceanFetchGlobeVolume(
  key: string,
  q: GlobeVolumeQuery = {},
): Promise<GlobeVolumePayload> {
  const r = await fetch(`${OCEAN_BASE}/globe_volume/${key}?${buildParams(q)}`);
  if (!r.ok) throw new Error(`ocean /globe_volume/${key} ${r.status}`);

  const sh = (r.headers.get("x-shape") ?? "").split(",").map(Number);
  if (sh.length !== 3 || sh.some((n) => !Number.isFinite(n)))
    throw new Error(`bad X-Shape from /globe_volume: ${r.headers.get("x-shape")}`);

  return {
    data: new Uint8Array(await r.arrayBuffer()),
    depth: sh[0], height: sh[1], width: sh[2],
    min: parseFloat(r.headers.get("x-min") ?? "0"),
    max: parseFloat(r.headers.get("x-max") ?? "1"),
    z0: parseInt(r.headers.get("x-z0") ?? "0", 10),
    z1: parseInt(r.headers.get("x-z1") ?? "0", 10),
  };
}

export async function oceanFetchVolume(
  key: string,
  q: VolumeQuery = {},
): Promise<VolumePayload> {
  const r = await fetch(`${OCEAN_BASE}/volume/${key}?${buildParams(q)}`);
  if (!r.ok) throw new Error(`ocean /volume/${key} ${r.status}`);

  const sh = (r.headers.get("x-shape") ?? "").split(",").map(Number);
  if (sh.length !== 3 || sh.some((n) => !Number.isFinite(n)))
    throw new Error(`bad X-Shape from /volume: ${r.headers.get("x-shape")}`);

  const data = new Float32Array(await r.arrayBuffer());
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isFinite(v) && Math.abs(v) > 1e-6) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (min === Infinity) min = 0;
  if (max === -Infinity) max = 1;

  return { data, shape: [sh[0], sh[1], sh[2]], min, max };
}
