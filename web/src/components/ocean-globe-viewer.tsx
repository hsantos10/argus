"use client";

/**
 * GPU-accelerated ocean globe viewer for LLC4320 data.
 * Adapted from GlobeViewer.tsx by TODO-ATTRIBUTION.
 * Fetches raw uint8 depth blocks from the FastAPI backend (port 8000) and
 * ray-marches through them on the GPU — no Python rendering round-trip.
 */

import { useEffect, useRef, useState, useMemo } from "react";
import * as THREE from "three";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import { createXRStore, XR, useXR } from "@react-three/xr";
import { cn } from "@/lib/utils";
import { makeColormapTexture, COLORMAP_CSS_STOPS, type ColormapName } from "@/lib/transfer-functions";
import { oceanFetchGlobeVolume, oceanGetMetadata, oceanFetchLatLUT, oceanListDatasets, type GlobeVolumePayload, type OceanDatasetInfo } from "@/lib/ocean-api";
import { useXRSupport } from "@/lib/use-xr-support";
import { VRButton } from "@/components/vr-button";
import { formatLevelPhysicalOnly } from "@/lib/depth-levels";

// Single shared store — only one OceanGlobeViewer is ever mounted at a time.
const oceanXRStore = createXRStore();

// ─── Constants ────────────────────────────────────────────────────────────────

const GLOBE_R_OUT = 1.0;
const GLOBE_R_IN = 0.85;
// The ray-march mesh's own bounding geometry must be large enough to contain
// whichever shell is active — ocean groups stay within GLOBE_R_OUT, but the
// atmosphere group renders outward from the surface up to GLOBE_R_OUT plus
// the same shell thickness (see the uROut/uRIn per-group effect below).
const GLOBE_R_BOUND = GLOBE_R_OUT + (GLOBE_R_OUT - GLOBE_R_IN);

// ─── Inline GLSL (spherical-shell ray-marching) ───────────────────────────────

const GLOBE_VERT = /* glsl */ `
  out vec3 vPos;
  out vec3 vCamObj;
  void main() {
    vPos = position;
    vCamObj = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const GLOBE_FRAG = /* glsl */ `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uVolume;
  uniform sampler2D uColormap;
  uniform sampler2D uLatLUT;
  uniform vec2 uBlockZ;
  uniform vec2 uRenderZ;
  uniform float uSteps;
  uniform float uDensity;
  uniform bool uHasData;
  // Per-dataset-group shell radii (uniforms, not compile-time consts): ocean
  // groups have layer 0 = surface (uROut) deepening inward to uRIn; the
  // atmosphere group is the opposite sense — layer 0 is the top of the
  // atmosphere, which should render OUTWARD from the planet's surface, not
  // toward its core, so for that group uROut/uRIn are set to radii both
  // >= the surface (see GlobeScene's per-group effect).
  uniform float uROut;
  uniform float uRIn;
  // Level range mapped across the [uROut, uRIn] shell. Ocean groups use the
  // full level range (0 .. nLevels-1) so depth stays physically proportional;
  // the atmosphere group instead stretches the LOADED level block across the
  // whole shell — otherwise its ~16 near-surface levels out of 52 would all
  // sit within a few % of the surface radius and never visibly grow beyond
  // the globe.
  uniform vec2 uShellZ;
  // When true (all datasets, by user preference), any ray that hits real
  // data renders fully opaque (color un-premultiplied back to full
  // brightness) instead of a soft volumetric haze — reads as a solid
  // weather/ocean surface rather than a translucent wash over the land layer.
  uniform bool uOpaque;
  // True while the land relief mesh is visible — the planet is then opaque,
  // so the march must not include the far-side shell segment (see below).
  uniform bool uLandShown;

  in vec3 vPos;
  in vec3 vCamObj;
  out vec4 outColor;

  vec2 sphereHit(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float h = b * b - c;
    if (h < 0.0) return vec2(1.0, -1.0);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
  }

  float layerToR(float layer) {
    float span = max(uShellZ.y - uShellZ.x, 1e-3);
    return uROut - (layer - uShellZ.x) / span * (uROut - uRIn);
  }

  void main() {
    if (!uHasData) discard;

    vec3 ro = vCamObj;
    vec3 rd = normalize(vPos - vCamObj);

    float lo = max(uRenderZ.x, uBlockZ.x) - 0.5;
    float hi = min(uRenderZ.y, uBlockZ.y) + 0.5;
    if (hi <= lo) discard;
    float rMax = min(layerToR(lo), uROut);
    float rMin = max(layerToR(hi), 0.001);

    vec2 tOut = sphereHit(ro, rd, rMax);
    if (tOut.y < 0.0) discard;
    float a0 = max(tOut.x, 0.0);
    float a1 = tOut.y;

    float seg1a = a0, seg1b = a1;
    float seg2a = 0.0, seg2b = -1.0;
    vec2 tIn = sphereHit(ro, rd, rMin);
    if (tIn.y >= tIn.x) {
      seg1b = min(a1, tIn.x);
      seg2a = max(a0, tIn.y);
      seg2b = a1;
    }
    // seg2 is the shell on the FAR side of the planet — visible only because
    // the globe is otherwise hollow. With the opaque land surface shown, rays
    // must stop at the planet instead of passing through its core; otherwise
    // far-side ocean colors fill in the continents (which have no near-side
    // data) and land looks "translucent".
    if (uLandShown) seg2b = -1.0;
    float len1 = max(seg1b - seg1a, 0.0);
    float len2 = max(seg2b - seg2a, 0.0);
    float L = len1 + len2;
    if (L <= 0.0) discard;

    float nSteps = min(uSteps, max(32.0, 2.0 * (hi - lo)));
    float dt = L / nSteps;
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    vec4 acc = vec4(0.0);

    for (int i = 0; i < 1024; i++) {
      if (float(i) >= nSteps || acc.a > 0.98) break;
      float marchDist = (float(i) + jitter) * dt;
      float t = (marchDist < len1) ? (seg1a + marchDist) : (seg2a + (marchDist - len1));
      vec3 p = ro + rd * t;
      float r = length(p);

      float layer = uShellZ.x + (uROut - r) / (uROut - uRIn) * max(uShellZ.y - uShellZ.x, 1e-3);
      if (layer < lo || layer > hi) continue;

      float sinLat = clamp(p.y / r, -1.0, 1.0);
      float v = texture(uLatLUT, vec2((sinLat + 1.0) * 0.5, 0.5)).r;
      if (v < 0.0) continue;
      float u = atan(-p.z, p.x) * 0.15915494309 + 0.5;
      float w = (layer - uBlockZ.x + 0.5) / (uBlockZ.y - uBlockZ.x + 1.0);

      float s = texture(uVolume, vec3(u, v, w)).r;
      if (s < 0.5 / 255.0) continue;

      float n = clamp((s * 255.0 - 1.0) / 254.0, 0.0, 1.0);
      vec3 col = texture(uColormap, vec2(n, 0.5)).rgb;
      float a = 1.0 - exp(-uDensity * dt);

      acc.rgb += (1.0 - acc.a) * a * col;
      acc.a   += (1.0 - acc.a) * a;
    }

    if (acc.a < 0.01) discard;
    outColor = uOpaque ? vec4(acc.rgb / acc.a, 1.0) : vec4(acc.rgb, acc.a);
  }
`;

// ─── Types ────────────────────────────────────────────────────────────────────

// ─── Dataset catalog ──────────────────────────────────────────────────────────
// Static mirror of the backend registry (/api/datasets); the hook re-fetches the
// live list at runtime and falls back to this when the backend is offline.

export type DatasetGroupId = "ecco" | "dyamond_ocean" | "dyamond_atm" | "cmip6";

export const DATASET_GROUPS: { id: DatasetGroupId; label: string }[] = [
  { id: "ecco", label: "ECCO LLC4320 · Ocean" },
  { id: "dyamond_ocean", label: "DYAMOND LLC2160 · Ocean" },
  { id: "dyamond_atm", label: "DYAMOND GEOS · Atmosphere" },
  { id: "cmip6", label: "CMIP6 NEX-GDDP · Climate" },
];

// CMIP6 experiments served by the backend (one model: ACCESS-CM2). historical
// covers 1950–2014; the SSP scenarios cover 2015–2100.
export const CMIP6_SCENARIOS = [
  { id: "historical", label: "Historical" },
  { id: "ssp126", label: "SSP1-2.6" },
  { id: "ssp245", label: "SSP2-4.5" },
  { id: "ssp370", label: "SSP3-7.0" },
  { id: "ssp585", label: "SSP5-8.5" },
] as const;
export type Cmip6Scenario = (typeof CMIP6_SCENARIOS)[number]["id"];

export type DatasetDef = {
  key: string;
  label: string; // short label for the variable pills
  cmap: ColormapName;
  unit: string;
  group: DatasetGroupId;
  grid: "llc" | "cubed" | "latlon";
};

export const DATASET_DEFS: DatasetDef[] = [
  { key: "theta",         label: "Temp (°C)",       cmap: "sst",      unit: "°C",   group: "ecco",          grid: "llc" },
  { key: "salt",          label: "Salinity (PSU)",  cmap: "viridis",  unit: "PSU",  group: "ecco",          grid: "llc" },
  { key: "w",             label: "Vert. velocity",  cmap: "puor",     unit: "m/s",  group: "ecco",          grid: "llc" },
  { key: "dyamond_theta", label: "Temp (°C)",       cmap: "sst",      unit: "°C",   group: "dyamond_ocean", grid: "llc" },
  { key: "dyamond_salt",  label: "Salinity (PSU)",  cmap: "viridis",  unit: "PSU",  group: "dyamond_ocean", grid: "llc" },
  { key: "dyamond_w",     label: "Vert. velocity",  cmap: "puor",     unit: "m/s",  group: "dyamond_ocean", grid: "llc" },
  { key: "geos_t",        label: "Air temp (K)",    cmap: "coolwarm", unit: "K",    group: "dyamond_atm",   grid: "cubed" },
  { key: "geos_u",        label: "Zonal wind (u)",  cmap: "puor",     unit: "m/s",  group: "dyamond_atm",   grid: "cubed" },
  { key: "geos_v",        label: "Merid. wind (v)", cmap: "puor",     unit: "m/s",  group: "dyamond_atm",   grid: "cubed" },
  { key: "geos_fcld",     label: "Cloud fraction",  cmap: "viridis",  unit: "frac", group: "dyamond_atm",   grid: "cubed" },
  { key: "geos_qv",       label: "Humidity (qv)",   cmap: "viridis",  unit: "kg/kg", group: "dyamond_atm",  grid: "cubed" },
  { key: "geos_w",        label: "Vert. wind (w)",  cmap: "puor",     unit: "m/s",  group: "dyamond_atm",   grid: "cubed" },
  { key: "geos_ql",       label: "Cloud liquid",    cmap: "viridis",  unit: "kg/kg", group: "dyamond_atm",  grid: "cubed" },
  { key: "geos_qi",       label: "Cloud ice",       cmap: "viridis",  unit: "kg/kg", group: "dyamond_atm",  grid: "cubed" },
  { key: "geos_co2",      label: "CO₂",             cmap: "viridis",  unit: "mol/mol", group: "dyamond_atm", grid: "cubed" },
  { key: "cmip6_tas",     label: "Air temp (K)",    cmap: "coolwarm", unit: "K",    group: "cmip6",         grid: "latlon" },
  { key: "cmip6_tasmax",  label: "Max temp (K)",    cmap: "coolwarm", unit: "K",    group: "cmip6",         grid: "latlon" },
  { key: "cmip6_tasmin",  label: "Min temp (K)",    cmap: "coolwarm", unit: "K",    group: "cmip6",         grid: "latlon" },
  { key: "cmip6_pr",      label: "Precipitation",   cmap: "viridis",  unit: "kg/m²/s", group: "cmip6",      grid: "latlon" },
  { key: "cmip6_sfcwind", label: "Surface wind",    cmap: "viridis",  unit: "m/s",  group: "cmip6",         grid: "latlon" },
  { key: "cmip6_hurs",    label: "Rel. humidity",   cmap: "viridis",  unit: "%",    group: "cmip6",         grid: "latlon" },
  { key: "cmip6_huss",    label: "Spec. humidity",  cmap: "viridis",  unit: "kg/kg", group: "cmip6",        grid: "latlon" },
  { key: "cmip6_rlds",    label: "Longwave down",   cmap: "sst",      unit: "W/m²", group: "cmip6",         grid: "latlon" },
  { key: "cmip6_rsds",    label: "Shortwave down",  cmap: "sst",      unit: "W/m²", group: "cmip6",         grid: "latlon" },
];

const DATASET_BY_KEY: Record<string, DatasetDef> = Object.fromEntries(
  DATASET_DEFS.map((d) => [d.key, d]),
);

export function datasetDef(key: string): DatasetDef {
  return DATASET_BY_KEY[key] ?? { key, label: key, cmap: "viridis", unit: "", group: "ecco", grid: "llc" };
}

// Per-group time axis (timestep counts verified against the IDX servers) and
// vertical extent. dtMs is the step between snapshots (hourly for the runs,
// daily for CMIP6). CMIP6's time axis is a NOLEAP (365-day) calendar —
// 55115 steps = 151 × 365 covering 1950-01-01…2100-12-31 with no Feb 29.
const GROUP_TIME: Record<DatasetGroupId, {
  t0Iso: string; nSteps: number; depthLevels: number; depthName: string;
  dtMs: number; calendar?: "noleap";
}> = {
  ecco:          { t0Iso: "2011-09-10T00:00:00Z", nSteps: 10312, depthLevels: 90, depthName: "Depth", dtMs: 3600_000 },
  dyamond_ocean: { t0Iso: "2020-01-20T00:00:00Z", nSteps: 10366, depthLevels: 90, depthName: "Depth", dtMs: 3600_000 },
  dyamond_atm:   { t0Iso: "2020-01-20T00:00:00Z", nSteps: 10269, depthLevels: 52, depthName: "Level", dtMs: 3600_000 },
  cmip6:         { t0Iso: "1950-01-01T00:00:00Z", nSteps: 55115, depthLevels: 1,  depthName: "Surface", dtMs: 86_400_000, calendar: "noleap" },
};

// ── Noleap (365-day) calendar helpers — CMIP6's time axis has no leap days ──
const NOLEAP_CUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const NOLEAP_Y0 = 1950;

function noleapIndexFromDate(d: Date): number {
  const m = d.getUTCMonth();
  // Feb 29 (absent in this calendar) maps to Feb 28.
  const day = Math.min(d.getUTCDate(), m === 1 ? 28 : 31);
  return (d.getUTCFullYear() - NOLEAP_Y0) * 365 + NOLEAP_CUM[m] + day - 1;
}

function noleapDateFromIndex(i: number): Date {
  const y = NOLEAP_Y0 + Math.floor(i / 365);
  const doy = ((i % 365) + 365) % 365;
  let m = 11;
  while (m > 0 && NOLEAP_CUM[m] > doy) m--;
  return new Date(Date.UTC(y, m, doy - NOLEAP_CUM[m] + 1));
}

// Group-aware timestep ⇄ Date conversion (uses the group's dtMs / calendar).
function groupTimeToDate(g: DatasetGroupId, t: number): Date {
  const gt = GROUP_TIME[g];
  if (gt.calendar === "noleap") return noleapDateFromIndex(t);
  return new Date(new Date(gt.t0Iso).getTime() + t * gt.dtMs);
}

function groupDateToTime(g: DatasetGroupId, d: Date): number {
  const gt = GROUP_TIME[g];
  const raw = gt.calendar === "noleap"
    ? noleapIndexFromDate(d)
    : Math.round((d.getTime() - new Date(gt.t0Iso).getTime()) / gt.dtMs);
  return Math.max(0, Math.min(gt.nSteps - 1, raw));
}

type BlockCache = {
  variable: string;
  time: number;
  quality: number;
  z0: number;
  z1: number;
  data: Uint8Array;
  width: number;
  height: number;
  min: number;
  max: number;
  // Actual layer count in `data`. OpenVisus quality downsampling can reduce
  // the z axis too (small grids like GEOS faces), so this may be less than
  // z1 - z0 + 1 — in that case per-layer slicing/merging is impossible.
  layers: number;
};

export type GlobeLogLine = { t: number; msg: string };

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function timeToDateStr(g: DatasetGroupId, t: number): string {
  const d = groupTimeToDate(g, t);
  const dd = d.getUTCDate().toString().padStart(2, "0");
  const mon = MONTHS[d.getUTCMonth()];
  const yyyy = d.getUTCFullYear();
  if (GROUP_TIME[g].dtMs >= 86_400_000) return `${dd} ${mon} ${yyyy}`;
  const hh = d.getUTCHours().toString().padStart(2, "0");
  return `${dd} ${mon} ${yyyy}  ${hh}:00 UTC`;
}

function dateStrToTimestep(g: DatasetGroupId, dateStr: string): number {
  return groupDateToTime(g, new Date(dateStr + "T00:00:00Z"));
}

function timestepToIsoDate(g: DatasetGroupId, t: number): string {
  return groupTimeToDate(g, t).toISOString().slice(0, 10);
}

function groupDateMin(g: DatasetGroupId): string {
  return GROUP_TIME[g].t0Iso.slice(0, 10);
}

function groupDateMax(g: DatasetGroupId): string {
  const gt = GROUP_TIME[g];
  return groupTimeToDate(g, gt.nSteps - 1).toISOString().slice(0, 10);
}

export type GlobeFrame = {
  time: number;
  dateStr: string;
  volumeTex: THREE.Data3DTexture;
  dataRange: [number, number];
  blockZ: [number, number];
};

export type GlobeDataState = {
  group: DatasetGroupId;
  // Optional initialRange overrides the group's default loaded/render layer
  // range in the same state update — callers (e.g. dashboard presets) can't
  // call setLoadedRange right after, because it would clamp against the OLD
  // group's depthLevels (stale closure).
  setGroup: (g: DatasetGroupId, initialRange?: [number, number]) => void;
  variables: string[]; // dataset keys available in the active group
  variable: string;
  setVariable: (v: string) => void;
  depthLevels: number;
  depthName: string;
  dateMin: string; // "YYYY-MM-DD" bounds of the active group's time axis
  dateMax: string;
  currentIsoDate: string;
  timestepForDate: (dateStr: string) => number;
  time: number;
  setTime: (t: number) => void;
  currentDateStr: string;
  loadedRange: [number, number];
  setLoadedRange: (r: [number, number]) => void;
  renderRange: [number, number];
  setRenderRange: (r: [number, number]) => void;
  quality: number;
  setQuality: (q: number) => void;
  volumeTex: THREE.Data3DTexture | null;
  latTex: THREE.DataTexture | null;
  blockZ: [number, number];
  colormap: THREE.DataTexture;
  dataRange: [number, number];
  // active (multi-frame aware) values used by the scene + legend
  activeTexture: THREE.Data3DTexture | null;
  activeBlockZ: [number, number];
  activeDataRange: [number, number];
  // variable actually backing activeTexture — lags `variable` until a fetch
  // completes, so colormap/legend/units never flash the wrong dataset's look
  // while new data is in flight.
  loadedVariable: string;
  loading: boolean;
  error: string | null;
  log: GlobeLogLine[];
  fetchedBytes: number | null;
  // force a refetch of the current variable/time/quality/range even if none
  // of them changed since the last fetch (VR "LOAD" button).
  fetchSingle: () => void;
  // multi-frame
  startDate: string;        // "YYYY-MM-DD"
  endDate: string;
  setStartDate: (d: string) => void;
  setEndDate: (d: string) => void;
  frames: GlobeFrame[];
  activeFrame: number;
  setActiveFrame: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean | ((prev: boolean) => boolean)) => void;
  nFramesTarget: number;
  setNFramesTarget: (n: number) => void;
  fetchFrames: () => void;
  fetchingFrames: boolean;
  // full-precision current time, for VR's Y/M/D/H spinner controls
  maxTimestep: number;
  currentDate: Date;
  dateToTime: (d: Date) => number;
  // toggleable Earth relief/land layer, shown wherever the data shell has no
  // data (land, over ocean views; see GlobeScene's land overlay mesh)
  showLand: boolean;
  setShowLand: (v: boolean) => void;
  // CMIP6 experiment (historical 1950–2014, SSPs 2015–2100); ignored by the
  // other groups.
  cmip6Scenario: Cmip6Scenario;
  setCmip6Scenario: (s: Cmip6Scenario) => void;
  // Coupled wind vector overlay (DYAMOND only): draw GEOS surface/level winds
  // as arrows or flowing particles on top of the ocean-temperature globe.
  windOverlay: boolean;
  setWindOverlay: (v: boolean) => void;
  windLevel: number;              // GEOS level 0..51 (51 = surface)
  setWindLevel: (n: number) => void;
  windDensity: number;            // particle-pool size (see WIND_DENSITIES)
  setWindDensity: (n: number) => void;
  windField: WindField | null;
  windLoading: boolean;
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeVolumeTex(data: Uint8Array, w: number, h: number, d: number): THREE.Data3DTexture {
  const tex = new THREE.Data3DTexture(data as unknown as Uint8Array<ArrayBuffer>, w, h, d);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

function sliceBlock(cache: BlockCache, z0: number, z1: number): Uint8Array {
  const layerSize = cache.width * cache.height;
  return cache.data.slice((z0 - cache.z0) * layerSize, (z1 - cache.z0 + 1) * layerSize);
}

function concatBlocks(...parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) { out.set(p, offset); offset += p.length; }
  return out;
}

// ─── Wind vector field (coupled overlay) ─────────────────────────────────────
// One GEOS atmosphere level of zonal (u) and meridional (v) wind, decoded from
// the uint8 globe_volume blocks back to m/s and sampled on the CPU so glyphs /
// particles can be placed by true geography over the ocean SST globe.

export type WindField = {
  u: Float32Array;   // eastward m/s, row-major [row*width + col]
  v: Float32Array;   // northward m/s
  width: number;
  height: number;
  level: number;     // GEOS level (0 = TOA … 51 = surface) actually shown
  time: number;
  speedMax: number;  // robust (98th-pct) speed, for length/colour normalisation
};

// Decode one z-layer of a uint8 globe_volume block into m/s. The backend maps
// each block's [min,max] onto bytes 1…255, reserving 0 for missing (see
// backend/main.py globe_volume: `1 + round(norm*254)`), so the inverse is
// min + (byte-1)/254*(max-min); byte 0 → 0 (missing / no-data).
function decodeWindLayer(p: GlobeVolumePayload, level: number): Float32Array {
  const layerSize = p.width * p.height;
  // globe_volume may quality-downsample the z axis, so the returned depth can
  // be fewer layers than (z1-z0+1). Map the requested level into whatever
  // layers we actually got.
  const span = Math.max(1, p.z1 - p.z0);
  const idx = p.depth === (p.z1 - p.z0 + 1)
    ? level - p.z0
    : Math.round(((level - p.z0) / span) * (p.depth - 1));
  const layer = Math.max(0, Math.min(p.depth - 1, idx));
  const base = layer * layerSize;
  const out = new Float32Array(layerSize);
  const scale = (p.max - p.min) / 254;
  for (let i = 0; i < layerSize; i++) {
    const b = p.data[base + i];
    out[i] = b === 0 ? 0 : p.min + (b - 1) * scale;
  }
  return out;
}

async function fetchWindField(time: number, level: number, quality: number): Promise<WindField> {
  const lvl = Math.max(0, Math.min(51, Math.round(level)));
  // Read an 8-layer block containing `lvl`. A single-layer read can fail on the
  // GEOS faces ("cannot find a good end_resolution"), so we always fetch a band
  // and slice the wanted level out.
  const z1 = Math.min(51, lvl + 4);
  const z0 = Math.max(0, z1 - 7);
  const [pu, pv] = await Promise.all([
    oceanFetchGlobeVolume("geos_u", { time, z0, z1, quality }),
    oceanFetchGlobeVolume("geos_v", { time, z0, z1, quality }),
  ]);
  if (pu.width !== pv.width || pu.height !== pv.height) {
    throw new Error(`wind u/v grid mismatch: ${pu.width}×${pu.height} vs ${pv.width}×${pv.height}`);
  }
  const u = decodeWindLayer(pu, lvl);
  const v = decodeWindLayer(pv, lvl);

  // Robust max speed (98th percentile) so a few outliers don't crush the ramp.
  const n = u.length;
  const speeds = new Float32Array(n);
  for (let i = 0; i < n; i++) speeds[i] = Math.hypot(u[i], v[i]);
  const sorted = Float32Array.from(speeds).sort();
  const speedMax = Math.max(1e-3, sorted[Math.floor(0.98 * (n - 1))]);

  return { u, v, width: pu.width, height: pu.height, level: lvl, time, speedMax };
}

// Bilinear sample of the wind field at true geographic (lon°, lat°). Longitude
// wraps; the equirect grid runs col 0 = lon −180 (no LLC roll on cubed data)
// and row 0 = lat −90 (backend linspace(-90,90)), matching the display's
// texture-u = (lon+180)/360 convention used by the land mesh.
function sampleWindField(f: WindField, lonDeg: number, latDeg: number): { u: number; v: number; speed: number } {
  const { u, v, width: W, height: H } = f;
  let fx = ((((lonDeg + 180) % 360) + 360) % 360) / 360 * W;   // 0..W (wraps)
  const fy = Math.max(0, Math.min(H - 1, (latDeg + 90) / 180 * (H - 1)));
  const x0 = Math.floor(fx) % W;
  const x1 = (x0 + 1) % W;
  const y0 = Math.floor(fy);
  const y1 = Math.min(H - 1, y0 + 1);
  const tx = fx - Math.floor(fx);
  const ty = fy - y0;
  const at = (arr: Float32Array, x: number, y: number) => arr[y * W + x];
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const su = lerp(lerp(at(u, x0, y0), at(u, x1, y0), tx), lerp(at(u, x0, y1), at(u, x1, y1), tx), ty);
  const sv = lerp(lerp(at(v, x0, y0), at(v, x1, y0), tx), lerp(at(v, x0, y1), at(v, x1, y1), tx), ty);
  return { u: su, v: sv, speed: Math.hypot(su, sv) };
}

// ─── Data hook ────────────────────────────────────────────────────────────────

export function useGlobeData(enabled = false): GlobeDataState {
  const [group, setGroupState] = useState<DatasetGroupId>("ecco");
  const [variable, setVariable] = useState<string>("theta");
  const [remoteDatasets, setRemoteDatasets] = useState<OceanDatasetInfo[] | null>(null);
  const [time, setTime] = useState(0);
  const [quality, setQuality] = useState(-8);
  const [loadedRange, setLoadedRangeState] = useState<[number, number]>([0, 15]);
  const [renderRange, setRenderRangeState] = useState<[number, number]>([0, 15]);
  const [volumeTex, setVolumeTex] = useState<THREE.Data3DTexture | null>(null);
  const [latTex, setLatTex] = useState<THREE.DataTexture | null>(null);
  const [blockZ, setBlockZ] = useState<[number, number]>([0, 15]);
  const [dataRange, setDataRange] = useState<[number, number]>([0, 1]);
  const [loadedVariable, setLoadedVariable] = useState<string>("theta");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<GlobeLogLine[]>([]);
  const [fetchedBytes, setFetchedBytes] = useState<number | null>(null);
  const cacheRef = useRef<BlockCache | null>(null);
  const latTexCacheRef = useRef<Record<string, THREE.DataTexture>>({});
  const fetchT0 = useRef<number>(0);
  const [fetchTrigger, setFetchTrigger] = useState(0);
  const fetchSingle = () => setFetchTrigger((t) => t + 1);
  const [showLand, setShowLand] = useState(true);

  // Coupled wind overlay state
  const [windOverlay, setWindOverlay] = useState(false);
  const [windLevel, setWindLevel] = useState(51); // GEOS surface
  const [windDensity, setWindDensity] = useState<number>(2500);
  const [windField, setWindField] = useState<WindField | null>(null);
  const [windLoading, setWindLoading] = useState(false);
  const windCacheRef = useRef<Record<string, WindField>>({});

  // multi-frame state
  const [startDate, setStartDate] = useState(groupDateMin("ecco"));
  const [endDate, setEndDate] = useState(groupDateMax("ecco"));
  const [frames, setFrames] = useState<GlobeFrame[]>([]);
  const [activeFrame, setActiveFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [nFramesTarget, setNFramesTarget] = useState(8);
  const [fetchingFrames, setFetchingFrames] = useState(false);
  const fetchFramesAbortRef = useRef<boolean>(false);

  // CMIP6 experiment. historical covers 1950-01-01…2014-12-31 (noleap indices
  // 0…23724); the SSP scenarios cover 2015-01-01…2100-12-31 (23725…55114).
  const [cmip6Scenario, setCmip6ScenarioState] = useState<Cmip6Scenario>("ssp585");
  const CMIP6_SSP_START = (2015 - 1950) * 365; // first valid SSP index
  const cmip6TimeWindow = (s: Cmip6Scenario): [number, number] =>
    s === "historical" ? [0, CMIP6_SSP_START - 1] : [CMIP6_SSP_START, GROUP_TIME.cmip6.nSteps - 1];
  const setCmip6Scenario = (s: Cmip6Scenario) => {
    setCmip6ScenarioState(s);
    // Keep the current date inside the experiment's valid window — reads
    // outside it fail on the server (the field simply has no data there).
    const [lo, hi] = cmip6TimeWindow(s);
    setTime((t) => Math.max(lo, Math.min(hi, t)));
  };

  const timeInfo = GROUP_TIME[group];
  const depthLevels = timeInfo.depthLevels;
  // For CMIP6 the date pickers are bounded by the active scenario's window,
  // not the whole 1950–2100 axis.
  const dateMin = group === "cmip6"
    ? timestepToIsoDate("cmip6", cmip6TimeWindow(cmip6Scenario)[0])
    : groupDateMin(group);
  const dateMax = group === "cmip6"
    ? timestepToIsoDate("cmip6", cmip6TimeWindow(cmip6Scenario)[1])
    : groupDateMax(group);

  // Live dataset list from the backend; the static catalog is the offline fallback.
  useEffect(() => {
    if (!enabled || remoteDatasets) return;
    let cancelled = false;
    oceanListDatasets()
      .then((ds) => { if (!cancelled && Array.isArray(ds) && ds.length > 0) setRemoteDatasets(ds); })
      .catch(() => { /* backend offline — DATASET_DEFS covers the UI */ });
    return () => { cancelled = true; };
  }, [enabled, remoteDatasets]);

  const variables = useMemo(() => {
    const remote = remoteDatasets
      ?.filter((d) => (d.group ?? datasetDef(d.key).group) === group)
      .map((d) => d.key);
    if (remote && remote.length > 0) return remote;
    return DATASET_DEFS.filter((d) => d.group === group).map((d) => d.key);
  }, [remoteDatasets, group]);

  const setGroup = (g: DatasetGroupId, initialRange?: [number, number]) => {
    if (g === group && !initialRange) return;
    const gt = GROUP_TIME[g];
    setGroupState(g);
    setVariable(DATASET_DEFS.find((d) => d.group === g)?.key ?? "theta");
    // CMIP6 starts inside the active scenario's valid window (t=0 is 1950,
    // which only the "historical" experiment covers).
    setTime(g === "cmip6" ? cmip6TimeWindow(cmip6Scenario)[0] : 0);
    setStartDate(g === "cmip6" ? timestepToIsoDate("cmip6", cmip6TimeWindow(cmip6Scenario)[0]) : groupDateMin(g));
    setEndDate(g === "cmip6" ? timestepToIsoDate("cmip6", cmip6TimeWindow(cmip6Scenario)[1]) : groupDateMax(g));
    // GEOS levels run top-of-atmosphere (0) → surface (51), so the atmosphere
    // group defaults to the near-surface band where weather is visible; ocean
    // depth runs surface (0) → abyss, so it defaults to the top layers. CMIP6
    // is a single 2D surface layer.
    const clampG = (v: number) => Math.max(0, Math.min(gt.depthLevels - 1, Math.round(v)));
    const range: [number, number] = initialRange
      ? [clampG(Math.min(...initialRange)), clampG(Math.max(...initialRange))]
      : g === "cmip6"
        ? [0, 0]
        : g === "dyamond_atm"
          ? [gt.depthLevels - 16, gt.depthLevels - 1]
          : [0, 15];
    setLoadedRangeState(range);
    setRenderRangeState(range);
  };

  // clear frames when variable/quality/loadedRange/scenario changes
  useEffect(() => { setFrames([]); setActiveFrame(0); setPlaying(false); }, [variable, quality, loadedRange, cmip6Scenario]);

  const fetchFrames = async () => {
    if (fetchingFrames) return;
    fetchFramesAbortRef.current = false;
    setFetchingFrames(true);
    setFrames([]);

    const t0 = dateStrToTimestep(group, startDate);
    const t1 = dateStrToTimestep(group, endDate);
    const n = Math.max(2, Math.min(nFramesTarget, 20));
    const selectedTimes = Array.from({ length: n }, (_, i) =>
      Math.round(t0 + i * (t1 - t0) / (n - 1))
    );

    // use first frame's normalization for consistent colour scale across all frames
    let normMin: number | undefined;
    let normMax: number | undefined;
    const newFrames: GlobeFrame[] = [];

    for (let i = 0; i < selectedTimes.length; i++) {
      if (fetchFramesAbortRef.current) break;
      const t = selectedTimes[i];
      try {
        const q: Parameters<typeof oceanFetchGlobeVolume>[1] = {
          time: t, z0: loadedRange[0], z1: loadedRange[1], quality,
          ...(datasetDef(variable).group === "cmip6" ? { scenario: cmip6Scenario } : {}),
          ...(normMin !== undefined ? { norm_min: normMin, norm_max: normMax } : {}),
        };
        const p = await oceanFetchGlobeVolume(variable, q);
        if (i === 0) { normMin = p.min; normMax = p.max; }
        const tex = makeVolumeTex(p.data, p.width, p.height, p.depth);
        const frame: GlobeFrame = {
          time: t,
          dateStr: timeToDateStr(group, t),
          volumeTex: tex,
          dataRange: [p.min, p.max],
          blockZ: [p.z0, p.z1],
        };
        newFrames.push(frame);
        setFrames([...newFrames]);
      } catch { /* skip failed frame */ }
    }

    setFetchingFrames(false);
    setActiveFrame(0);
  };

  const colormap = useMemo(() => makeColormapTexture(datasetDef(variable).cmap), [variable]);

  const clampLayer = (v: number) => Math.max(0, Math.min(depthLevels - 1, Math.round(v)));

  const setLoadedRange = ([a, b]: [number, number]) => {
    const z0 = clampLayer(Math.min(a, b));
    const z1 = clampLayer(Math.max(a, b));
    setLoadedRangeState([z0, z1]);
    // Deep end tracks loadedRange[1]; shallow cutoff clamped into the new range.
    setRenderRangeState(([r0]) => [Math.max(z0, Math.min(z1, r0)), z1]);
  };

  const setRenderRange = ([a, b]: [number, number]) => {
    const d0 = clampLayer(Math.min(a, b));
    const d1 = clampLayer(Math.max(a, b));
    setRenderRangeState([
      Math.max(loadedRange[0], Math.min(loadedRange[1], d0)),
      Math.max(loadedRange[0], Math.min(loadedRange[1], d1)),
    ]);
  };

  // Fetch the latitude LUT for the active grid family: LLC mosaic, full
  // uniform-latitude equirect (reprojected cubed-sphere), or a lat-bounded
  // equirect (CMIP6 covers only −60…90°; rows outside map to −1 = no data).
  const activeGrid = datasetDef(variable).grid;
  const lutGrid = activeGrid === "llc" ? "llc4320" : "equirect";
  const lutBounds: [number, number] | undefined =
    activeGrid === "latlon" ? [-60, 90] : undefined;
  const lutCacheKey = lutBounds ? `${lutGrid}:${lutBounds[0]}:${lutBounds[1]}` : lutGrid;
  useEffect(() => {
    if (!enabled) return;
    const cached = latTexCacheRef.current[lutCacheKey];
    if (cached) {
      setLatTex((prev) => (prev === cached ? prev : cached));
      return;
    }
    let cancelled = false;
    oceanFetchLatLUT(lutGrid, lutBounds)
      .then((arr) => {
        if (cancelled) return;
        const tex = new THREE.DataTexture(
          arr as unknown as Float32Array<ArrayBuffer>,
          arr.length, 1,
          THREE.RedFormat, THREE.FloatType,
        );
        tex.minFilter = THREE.NearestFilter;
        tex.magFilter = THREE.NearestFilter;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.needsUpdate = true;
        latTexCacheRef.current[lutCacheKey] = tex;
        setLatTex(tex);
        setError(null);
        setLog(prev => [...prev, { t: (Date.now() - fetchT0.current) / 1000, msg: `Latitude LUT (${lutCacheKey}) ready (${(arr.byteLength / 1024).toFixed(1)} KB)` }]);
      })
      .catch((e) => {
        if (!cancelled) setError(`Ocean backend offline: ${e.message}`);
      });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, lutCacheKey]);

  // Fetch depth block with cache-aware logic (partial fetches reuse cached layers)
  // CMIP6 keys vary by scenario too — fold it into the cache identity and the
  // request query so switching experiments refetches.
  const isCmip6 = datasetDef(variable).group === "cmip6";
  const varKey = isCmip6 ? `${variable}@${cmip6Scenario}` : variable;
  const scenarioQ = isCmip6 ? { scenario: cmip6Scenario } : {};
  useEffect(() => {
    if (!enabled) return;

    const cache = cacheRef.current;
    const cacheValid = cache !== null &&
      cache.variable === varKey && cache.time === time && cache.quality === quality;
    // Per-layer slicing/merging only works when the cached block has full z
    // resolution (quality can downsample z on small grids).
    const cacheSliceable = cacheValid && cache!.layers === cache!.z1 - cache!.z0 + 1;
    const [newZ0, newZ1] = loadedRange;

    if (cacheValid && (cacheSliceable
      ? newZ0 >= cache!.z0 && newZ1 <= cache!.z1
      : newZ0 === cache!.z0 && newZ1 === cache!.z1)) {
      const sliced = cacheSliceable ? sliceBlock(cache!, newZ0, newZ1) : cache!.data;
      const nl = cacheSliceable ? newZ1 - newZ0 + 1 : cache!.layers;
      setVolumeTex((prev) => { prev?.dispose(); return makeVolumeTex(sliced, cache!.width, cache!.height, nl); });
      setBlockZ([newZ0, newZ1]);
      setDataRange([cache!.min, cache!.max]);
      setLoadedVariable(variable);
      return;
    }

    setVolumeTex((prev) => { prev?.dispose(); return null; });
    setLoading(true);
    fetchT0.current = Date.now();
    setLog([{ t: 0, msg: `Requesting ${datasetDef(variable).label} layers ${newZ0}–${newZ1} (quality ${quality})…` }]);
    setFetchedBytes(null);

    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        let merged: Uint8Array, mergedZ0: number, mergedZ1: number;
        let width: number, height: number, vmin: number, vmax: number;

        const hasOverlap = cacheSliceable && newZ0 <= cache!.z1 && newZ1 >= cache!.z0;
        const elapsed = () => (Date.now() - fetchT0.current) / 1000;

        if (!hasOverlap) {
          setLog(prev => [...prev, { t: elapsed(), msg: `Fetching ${newZ1 - newZ0 + 1} layer(s) from backend…` }]);
          const p = await oceanFetchGlobeVolume(variable, { time, z0: newZ0, z1: newZ1, quality, ...scenarioQ });
          if (cancelled) return;
          merged = p.data; mergedZ0 = p.z0; mergedZ1 = p.z1;
          width = p.width; height = p.height; vmin = p.min; vmax = p.max;
        } else {
          const fetchLow  = newZ0 < cache!.z0 ? { z0: newZ0, z1: cache!.z0 - 1 } : null;
          const fetchHigh = newZ1 > cache!.z1 ? { z0: cache!.z1 + 1, z1: newZ1 } : null;
          const normParams = { norm_min: cache!.min, norm_max: cache!.max };
          const newLayers = (fetchLow ? fetchLow.z1 - fetchLow.z0 + 1 : 0) + (fetchHigh ? fetchHigh.z1 - fetchHigh.z0 + 1 : 0);
          setLog(prev => [...prev, { t: elapsed(), msg: `Fetching ${newLayers} new layer(s) (${newZ1 - newZ0 + 1 - newLayers} cached)…` }]);

          const [lowP, highP] = await Promise.all([
            fetchLow  ? oceanFetchGlobeVolume(variable, { time, quality, ...fetchLow, ...normParams, ...scenarioQ }) : null,
            fetchHigh ? oceanFetchGlobeVolume(variable, { time, quality, ...fetchHigh, ...normParams, ...scenarioQ }) : null,
          ]);
          if (cancelled) return;

          const ref = (lowP ?? highP)!;
          const partFullRes = (pp: typeof lowP) =>
            !pp || pp.data.length === (pp.z1 - pp.z0 + 1) * pp.width * pp.height;

          if (!partFullRes(lowP) || !partFullRes(highP) || ref.width !== cache!.width || ref.height !== cache!.height) {
            // z axis (or resolution) was quality-downsampled — the parts can't
            // be stitched layer-by-layer; refetch the whole range instead.
            const p = await oceanFetchGlobeVolume(variable, { time, z0: newZ0, z1: newZ1, quality, ...scenarioQ });
            if (cancelled) return;
            merged = p.data; mergedZ0 = p.z0; mergedZ1 = p.z1;
            width = p.width; height = p.height; vmin = p.min; vmax = p.max;
          } else {
            width = ref.width; height = ref.height;
            vmin = cache!.min; vmax = cache!.max;
            mergedZ0 = newZ0; mergedZ1 = newZ1;

            const mid = sliceBlock(cache!, Math.max(newZ0, cache!.z0), Math.min(newZ1, cache!.z1));
            const parts: Uint8Array[] = [];
            if (lowP)  parts.push(lowP.data);
            parts.push(mid);
            if (highP) parts.push(highP.data);
            merged = concatBlocks(...parts);
          }
        }

        // Actual layer count — may be less than the z range if quality
        // downsampled the z axis (GEOS faces, coarse LLC reads).
        const layers = Math.max(1, Math.round(merged!.length / (width! * height!)));
        const bytes = merged!.byteLength;
        const kb = bytes / 1024;
        const sizeStr = kb < 1024 ? `${kb.toFixed(0)} KB` : `${(kb / 1024).toFixed(2)} MB`;
        setLog(prev => [...prev, {
          t: elapsed(),
          msg: `Done  —  ${layers} layer(s) · ${width}×${height} px · ${sizeStr} downloaded`,
        }]);
        setFetchedBytes(bytes);

        cacheRef.current = { variable: varKey, time, quality, z0: mergedZ0!, z1: mergedZ1!, data: merged!, width: width!, height: height!, min: vmin!, max: vmax!, layers };
        setVolumeTex((prev) => { prev?.dispose(); return makeVolumeTex(merged!, width!, height!, layers); });
        setBlockZ([mergedZ0!, mergedZ1!]);
        setDataRange([vmin!, vmax!]);
        setError(null);
        setLoading(false);
        setLoadedVariable(variable);
      } catch (e) {
        if (!cancelled) {
          setError(`Failed to fetch ocean data: ${e instanceof Error ? e.message : String(e)}`);
          setLoading(false);
        }
      }
    }, 400);

    return () => { cancelled = true; clearTimeout(timer); };
  // fetchTrigger has no effect on the logic above — it's a pure re-run knob so
  // the VR "LOAD" button can force a refetch even when nothing else changed.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, variable, time, quality, loadedRange, fetchTrigger, varKey]);

  // Wind overlay fetch — coupled to the ocean base's timestep. Uses a fixed
  // coarse quality (a glyph lattice / particle field needs no fine grid) and a
  // small (time, level) cache so scrubbing back is instant.
  const WIND_QUALITY = -12;
  useEffect(() => {
    if (!enabled || !windOverlay || group !== "dyamond_ocean") return;
    // Match the ocean frame actually on screen: during multi-frame playback the
    // globe shows frames[activeFrame].time, so the wind must use that too.
    const af = Math.max(0, Math.min(frames.length - 1, activeFrame));
    const effTime = frames.length > 0 ? (frames[af]?.time ?? time) : time;
    const key = `${effTime}:${windLevel}`;
    const cached = windCacheRef.current[key];
    if (cached) { setWindField(cached); return; }
    let cancelled = false;
    setWindLoading(true);
    fetchWindField(effTime, windLevel, WIND_QUALITY)
      .then((wf) => {
        if (cancelled) return;
        windCacheRef.current[key] = wf;
        setWindField(wf);
        setWindLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        console.warn("Wind overlay fetch failed:", e);
        setWindLoading(false);
      });
    return () => { cancelled = true; };
  }, [enabled, windOverlay, group, time, windLevel, frames, activeFrame]);

  const activeFrame0 = Math.max(0, Math.min(frames.length - 1, activeFrame));
  const activeTexture = frames.length > 0 ? (frames[activeFrame0]?.volumeTex ?? null) : volumeTex;
  const activeBlockZ: [number, number] = frames.length > 0 ? (frames[activeFrame0]?.blockZ ?? blockZ) : blockZ;
  const activeDataRange: [number, number] = frames.length > 0 ? (frames[activeFrame0]?.dataRange ?? dataRange) : dataRange;

  const maxTimestep = timeInfo.nSteps - 1;
  const dateToTime = (d: Date) => groupDateToTime(group, d);

  return {
    group, setGroup, variables,
    depthLevels, depthName: timeInfo.depthName,
    dateMin, dateMax,
    currentIsoDate: timestepToIsoDate(group, time),
    timestepForDate: (d: string) => dateStrToTimestep(group, d),
    variable, setVariable,
    time, setTime,
    currentDateStr: timeToDateStr(group, time),
    loadedRange, setLoadedRange,
    renderRange, setRenderRange,
    quality, setQuality,
    volumeTex, latTex, blockZ, colormap, dataRange, loadedVariable, loading, error,
    log, fetchedBytes, fetchSingle,
    activeTexture, activeBlockZ, activeDataRange,
    startDate, endDate, setStartDate, setEndDate,
    frames, activeFrame: activeFrame0, setActiveFrame, playing, setPlaying,
    nFramesTarget, setNFramesTarget, fetchFrames, fetchingFrames,
    maxTimestep, currentDate: groupTimeToDate(group, time), dateToTime,
    showLand, setShowLand,
    cmip6Scenario, setCmip6Scenario,
    windOverlay, setWindOverlay,
    windLevel, setWindLevel, windDensity, setWindDensity, windField, windLoading,
  };
}

// ─── Scene helpers ────────────────────────────────────────────────────────────

function GeoGrid() {
  const geo = useMemo(() => {
    const N = 64, r = 1.002;
    const pts: number[] = [];
    const addCircle = (mkPt: (a: number) => [number, number, number]) => {
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * 2 * Math.PI, a1 = ((i + 1) / N) * 2 * Math.PI;
        pts.push(...mkPt(a0), ...mkPt(a1));
      }
    };
    for (let lon = 0; lon < 360; lon += 30) {
      const phi = (lon * Math.PI) / 180;
      addCircle((t) => [r * Math.sin(t) * Math.cos(phi), r * Math.cos(t), r * Math.sin(t) * Math.sin(phi)]);
    }
    for (let lat = -60; lat <= 60; lat += 30) {
      const s = Math.sin((lat * Math.PI) / 180), c = Math.cos((lat * Math.PI) / 180);
      addCircle((phi) => [r * c * Math.cos(phi), r * s, r * c * Math.sin(phi)]);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, []);
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial color="#1e3d5a" transparent opacity={0.35} />
    </lineSegments>
  );
}

// ─── Wind overlay rendering ──────────────────────────────────────────────────
// Radius just outside the surface shell (r=1.0) but comfortably outside the
// depth occluder (0.999) so near-side glyphs draw and far-side ones are hidden
// by the opaque globe, matching the ocean surface.
const WIND_R = GLOBE_R_OUT * 1.008;
const DEG2RAD = Math.PI / 180;

// (lon°, lat°) → world position, matching the land mesh / volume convention
// (phiPrime = lonRad; z sign flipped). See land geo build + GLOBE_FRAG line ~124.
function lonLatToVec3(lonDeg: number, latDeg: number, R: number, target: THREE.Vector3): THREE.Vector3 {
  const lon = lonDeg * DEG2RAD, lat = latDeg * DEG2RAD;
  const cl = Math.cos(lat);
  return target.set(R * cl * Math.cos(lon), R * Math.sin(lat), -R * cl * Math.sin(lon));
}

// Speed → colour ramp (calm blue → strong red), sampled on the CPU for both
// instanced arrows and particle lines.
const WIND_STOPS: [number, THREE.Color][] = [
  [0.0, new THREE.Color("#2b4b8f")],
  [0.35, new THREE.Color("#3fa7d6")],
  [0.6, new THREE.Color("#7ee787")],
  [0.8, new THREE.Color("#f2e94e")],
  [1.0, new THREE.Color("#f25f5c")],
];
function windColor(t: number, target: THREE.Color): THREE.Color {
  const c = Math.max(0, Math.min(1, t));
  for (let i = 1; i < WIND_STOPS.length; i++) {
    if (c <= WIND_STOPS[i][0]) {
      const [t0, a] = WIND_STOPS[i - 1], [t1, b] = WIND_STOPS[i];
      return target.copy(a).lerp(b, (c - t0) / (t1 - t0));
    }
  }
  return target.copy(WIND_STOPS[WIND_STOPS.length - 1][1]);
}

// Flowing-particle wind field (earth.nullschool.net style): a pool of particles
// advected along the wind, each drawn as a fading trail so the flow reads as
// moving streaks. Trails are stored in WORLD space (not lon/lat) so they stay
// continuous across the antimeridian — only respawns are discontinuous, and
// those collapse the whole trail to the new point.
const WIND_T = 20;          // trail points per particle (T-1 segments)
const WIND_GAIN = 1.5;      // advection gain: deg travelled per (m/s · second)

// Selectable particle-pool sizes (desktop + VR "density" control).
export const WIND_DENSITIES = [
  { label: "Sparse", n: 1200 },
  { label: "Normal", n: 2500 },
  { label: "Dense",  n: 5000 },
] as const;

function WindParticles({ field, count }: { field: WindField; count: number }) {
  const lineRef = useRef<THREE.LineSegments>(null);
  const N = count;

  const sim = useMemo(() => ({
    lon: new Float32Array(N),
    lat: new Float32Array(N),
    age: new Float32Array(N),
    maxAge: new Float32Array(N),
    history: new Float32Array(N * WIND_T * 3), // ring buffer of world positions
    head: 0,
    positions: new Float32Array(N * (WIND_T - 1) * 2 * 3),
    // RGBA per vertex: an itemSize-4 color attribute makes three.js enable
    // vertex alpha on LineBasicMaterial, so trails fade by transparency with
    // NORMAL blending. (The first version used additive blending, but adding
    // color onto the already-bright SST globe saturates straight to white —
    // which is why the speed ramp never showed up on screen.)
    colors: new Float32Array(N * (WIND_T - 1) * 2 * 4),
  }), [N]);

  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(sim.positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(sim.colors, 4));
    return g;
  }, [sim]);
  useEffect(() => () => geom.dispose(), [geom]);

  const spawn = useMemo(() => {
    const P = new THREE.Vector3();
    return (i: number) => {
      const lon = Math.random() * 360;
      const lat = Math.asin(Math.random() * 2 - 1) / DEG2RAD; // area-uniform
      sim.lon[i] = lon; sim.lat[i] = lat;
      sim.age[i] = 0;
      sim.maxAge[i] = 1.5 + Math.random() * 2.5;
      lonLatToVec3(lon, lat, WIND_R, P);
      // collapse the whole trail onto the spawn point (no slash from old pos)
      for (let s = 0; s < WIND_T; s++) {
        const h = (i * WIND_T + s) * 3;
        sim.history[h] = P.x; sim.history[h + 1] = P.y; sim.history[h + 2] = P.z;
      }
    };
  }, [sim]);

  // Seed / reseed whenever the field (or pool size) changes.
  useEffect(() => {
    for (let i = 0; i < N; i++) { spawn(i); sim.age[i] = Math.random() * sim.maxAge[i]; }
  }, [field, sim, spawn, N]);

  useFrame((_, rawDt) => {
    const dt = Math.min(0.033, rawDt);
    const P = new THREE.Vector3(), base = new THREE.Color();
    // Advance the ring head, then write each particle's new position there.
    sim.head = (sim.head + 1) % WIND_T;
    const head = sim.head;
    for (let i = 0; i < N; i++) {
      let lon = sim.lon[i], lat = sim.lat[i];
      const { u, v, speed } = sampleWindField(field, lon, lat);
      sim.age[i] += dt;
      if (sim.age[i] > sim.maxAge[i] || speed < 1e-3 || Math.abs(lat) > 88) {
        spawn(i);
      } else {
        lat = Math.max(-88, Math.min(88, lat + v * WIND_GAIN * dt));
        lon += (u * WIND_GAIN * dt) / Math.max(0.2, Math.cos(lat * DEG2RAD));
        sim.lon[i] = lon; sim.lat[i] = lat;
        lonLatToVec3(lon, lat, WIND_R, P);
        const h = (i * WIND_T + head) * 3;
        sim.history[h] = P.x; sim.history[h + 1] = P.y; sim.history[h + 2] = P.z;
      }
    }
    // Rebuild trail segments newest→oldest. The true speed-ramp RGB is kept
    // un-whitened; head→tail fade goes into ALPHA, and overall opacity is
    // weighted by speed so fast wind pops while calm regions stay quiet —
    // this is what makes the legend's colors actually readable on the globe.
    for (let i = 0; i < N; i++) {
      const { speed } = sampleWindField(field, sim.lon[i], sim.lat[i]);
      const t = Math.min(1, speed / field.speedMax);
      windColor(t, base);
      const aMax = 0.35 + 0.65 * t; // speed-weighted visibility
      const iBase = i * (WIND_T - 1) * 8;
      const hBase = i * WIND_T;
      for (let k = 0; k < WIND_T - 1; k++) {
        const slotA = (head - k + WIND_T) % WIND_T;
        const slotB = (head - k - 1 + WIND_T) % WIND_T;
        const fA = 1 - k / (WIND_T - 1);
        const fB = 1 - (k + 1) / (WIND_T - 1);
        const ha = (hBase + slotA) * 3, hb = (hBase + slotB) * 3;
        const po = i * (WIND_T - 1) * 6 + k * 6;
        sim.positions[po]     = sim.history[ha];
        sim.positions[po + 1] = sim.history[ha + 1];
        sim.positions[po + 2] = sim.history[ha + 2];
        sim.positions[po + 3] = sim.history[hb];
        sim.positions[po + 4] = sim.history[hb + 1];
        sim.positions[po + 5] = sim.history[hb + 2];
        const co = iBase + k * 8;
        sim.colors[co]     = base.r; sim.colors[co + 1] = base.g; sim.colors[co + 2] = base.b; sim.colors[co + 3] = aMax * fA;
        sim.colors[co + 4] = base.r; sim.colors[co + 5] = base.g; sim.colors[co + 6] = base.b; sim.colors[co + 7] = aMax * fB;
      }
    }
    geom.attributes.position.needsUpdate = true;
    geom.attributes.color.needsUpdate = true;
  });

  return (
    <lineSegments ref={lineRef} geometry={geom} renderOrder={2} frustumCulled={false}>
      <lineBasicMaterial vertexColors transparent depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

function WindOverlay({ state }: { state: GlobeDataState }) {
  if (!state.windOverlay || state.group !== "dyamond_ocean" || !state.windField) return null;
  return <WindParticles field={state.windField} count={state.windDensity} />;
}

function GlobeCameraInit() {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(0, 0, 2.8);
    camera.updateProjectionMatrix();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

// ─── GlobeScene ───────────────────────────────────────────────────────────────

type GlobeDragState =
  | { mode: "rotate";    prevPoint: THREE.Vector3; pointerId: number }
  | { mode: "translate"; pointers: Map<number, THREE.Vector3>; prevMid: THREE.Vector3 };

export function GlobeScene({ globeState }: { globeState: GlobeDataState }) {
  const matRef = useRef<THREE.ShaderMaterial | null>(null);
  const orbitRef = useRef<any>(null);
  const globeGroupRef = useRef<THREE.Group>(null);
  const dragRef = useRef<GlobeDragState | null>(null);
  const vrSavedRef = useRef<{ pos: THREE.Vector3; scale: THREE.Vector3 } | null>(null);
  const { camera } = useThree();

  const uniforms = useMemo(() => ({
    uVolume:   { value: null as THREE.Data3DTexture | null },
    uColormap: { value: null as THREE.DataTexture | null },
    uLatLUT:   { value: null as THREE.DataTexture | null },
    uBlockZ:   { value: new THREE.Vector2(0, 15) },
    uRenderZ:  { value: new THREE.Vector2(0, 15) },
    uSteps:    { value: 128 },
    uDensity:  { value: 450.0 },
    uHasData:  { value: false },
    uROut:     { value: GLOBE_R_OUT },
    uRIn:      { value: GLOBE_R_IN },
    uShellZ:   { value: new THREE.Vector2(0, 15) },
    uOpaque:   { value: true },
    uLandShown: { value: false },
  }), []);

  // Ocean groups deepen INWARD from the surface (layer 0 = shallow, at
  // GLOBE_R_OUT; deeper layers move toward GLOBE_R_IN, i.e. the core). The
  // atmosphere group is the opposite: layer 0 is the top of the atmosphere,
  // which belongs OUTSIDE the planet, and the highest layer index is the
  // surface — so its shell renders outward from GLOBE_R_OUT instead, using
  // the same shell thickness as the ocean case for a visually consistent
  // exaggeration.
  useEffect(() => {
    if (!matRef.current) return;
    const shellThickness = GLOBE_R_OUT - GLOBE_R_IN;
    const isAtmosphere = globeState.group === "dyamond_atm";
    const isSingleLayer = globeState.depthLevels === 1; // CMIP6 2D surface fields
    if (isSingleLayer) {
      // Thin opaque skin at the surface: crop window (lo −0.5 … hi +0.5, from
      // uRenderZ/uBlockZ = [0,0]) maps to radii [1.0, 0.98], and the inverse
      // mapping lands the single texture slice at w = layer + 0.5 ∈ [0,1].
      matRef.current.uniforms.uROut.value = GLOBE_R_OUT;
      matRef.current.uniforms.uRIn.value = GLOBE_R_OUT - 0.02;
      matRef.current.uniforms.uShellZ.value.set(-0.5, 0.5);
      return;
    }
    matRef.current.uniforms.uROut.value = isAtmosphere ? GLOBE_R_OUT + shellThickness : GLOBE_R_OUT;
    matRef.current.uniforms.uRIn.value = isAtmosphere ? GLOBE_R_OUT : GLOBE_R_IN;
    // Atmosphere: stretch the loaded level block over the full outward shell
    // so the rendered air column visibly rises above the globe (loading only
    // near-surface levels would otherwise all map within a few % of r=1).
    // Ocean: keep the fixed full-range mapping (surface..seafloor).
    if (isAtmosphere) {
      matRef.current.uniforms.uShellZ.value.set(globeState.activeBlockZ[0], globeState.activeBlockZ[1]);
    } else {
      matRef.current.uniforms.uShellZ.value.set(0, globeState.depthLevels - 1);
    }
  }, [globeState.group, globeState.activeBlockZ, globeState.depthLevels]);

  useEffect(() => {
    if (!matRef.current) return;
    const u = matRef.current.uniforms;
    u.uVolume.value = globeState.activeTexture;
    u.uBlockZ.value.set(globeState.activeBlockZ[0], globeState.activeBlockZ[1]);
    u.uHasData.value = !!(globeState.activeTexture && globeState.latTex);
  }, [globeState.activeTexture, globeState.activeBlockZ, globeState.latTex]);

  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uLatLUT.value = globeState.latTex;
    matRef.current.uniforms.uHasData.value = !!(globeState.volumeTex && globeState.latTex);
  }, [globeState.latTex, globeState.volumeTex]);

  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uRenderZ.value.set(globeState.renderRange[0], globeState.renderRange[1]);
  }, [globeState.renderRange]);

  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uColormap.value = globeState.colormap;
  }, [globeState.colormap]);

  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uSteps.value = globeState.quality <= -12 ? 64 : globeState.quality <= -8 ? 128 : 256;
  }, [globeState.quality]);

  // Land/geography relief layer — a static, evenly-lit equirectangular Earth
  // relief texture (NASA/USGS public-domain elevation + land-cover data, "no
  // clouds" composite) shown wherever the ray-marched data shell has no data
  // (land, for ocean views — the shell fully `discard`s those pixels, see
  // GLOBE_FRAG line ~138, so whatever's drawn underneath just shows through).
  // An earlier photographic Blue Marble texture was swapped out here because
  // it had baked-in shadow/cloud shading meant for lit rendering — as a flat
  // unlit texture it made parts of the globe (e.g. Scandinavia) render as
  // patches of near-black instead of visible terrain.
  const [landTex, setLandTex] = useState<THREE.Texture | null>(null);

  // Keep the shader's far-side culling in sync with land-mesh visibility.
  const landVisible = globeState.showLand && !!landTex;
  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uLandShown.value = landVisible;
  }, [landVisible]);

  useEffect(() => {
    let cancelled = false;
    new THREE.TextureLoader().load(
      "/textures/earth-relief.jpg",
      (tex) => {
        if (cancelled) return;
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.wrapS = THREE.RepeatWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.needsUpdate = true;
        setLandTex(tex);
      },
      undefined,
      (err) => console.warn("Land relief texture failed to load:", err),
    );
    return () => { cancelled = true; };
  }, []);

  // Custom geometry (not THREE.SphereGeometry's default UVs) so texture
  // coordinates use the exact same lon/lat convention as GLOBE_FRAG's volume
  // sampling (u = atan2(-z,x)/2π + 0.5, line ~124) — otherwise the land layer
  // would be rotated/mirrored relative to the ocean/atmosphere data. Built by
  // hand (rather than post-editing SphereGeometry's uv attribute) so the
  // antimeridian seam column gets distinct u=0/u=1 vertices like three.js's
  // own builder does, avoiding a seam-stretch artifact.
  const landGeo = useMemo(() => {
    const W = 64, H = 32;
    const radius = GLOBE_R_OUT * 0.9995;
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    for (let iy = 0; iy <= H; iy++) {
      const phi = Math.PI * (iy / H); // 0 at north pole, π at south pole
      const sinPhi = Math.sin(phi);
      const y = radius * Math.cos(phi);
      for (let ix = 0; ix <= W; ix++) {
        const u = ix / W;
        const phiPrime = 2 * Math.PI * u - Math.PI; // == atan2(-z,x) by construction
        const x = radius * sinPhi * Math.cos(phiPrime);
        const z = -radius * sinPhi * Math.sin(phiPrime);
        positions.push(x, y, z);
        uvs.push(u, 1 - iy / H); // flipY-style: north pole → top of source image
      }
    }
    for (let iy = 0; iy < H; iy++) {
      for (let ix = 0; ix < W; ix++) {
        const a = iy * (W + 1) + ix;
        const b = a + W + 1;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, []);

  // On entering VR, pull the globe to arm's reach in front of the user and
  // shrink it to a holdable size; restore its desktop transform on exit.
  useEffect(() => {
    let wasActive = false;
    return oceanXRStore.subscribe((s) => {
      const isActive = !!s.session;
      const group = globeGroupRef.current;
      if (isActive && !wasActive) {
        vrSavedRef.current = {
          pos: group?.position.clone() ?? new THREE.Vector3(),
          scale: group?.scale.clone() ?? new THREE.Vector3(1, 1, 1),
        };
        group?.position.set(0, 1.55, -1.2);
        group?.scale.setScalar(0.35);
      } else if (!isActive && wasActive) {
        const saved = vrSavedRef.current;
        if (saved && group) {
          group.position.copy(saved.pos);
          group.scale.copy(saved.scale);
          vrSavedRef.current = null;
        }
      }
      wasActive = isActive;
    });
  }, []);

  return (
    <>
      <GlobeCameraInit />
      <group ref={globeGroupRef}>
        <mesh
          onPointerDown={(e) => {
            e.stopPropagation();
            try { (e.target as any).setPointerCapture(e.pointerId); } catch {}
            if (!dragRef.current) {
              dragRef.current = { mode: "rotate", prevPoint: e.point.clone(), pointerId: e.pointerId };
              if (orbitRef.current) orbitRef.current.enabled = false;
            } else if (dragRef.current.mode === "rotate") {
              // Second controller pressed while the first still holds the globe
              // → switch to two-hand translate (grab-and-move) mode.
              const firstPt = dragRef.current.prevPoint.clone();
              const secondPt = e.point.clone();
              const pointers = new Map<number, THREE.Vector3>();
              pointers.set(dragRef.current.pointerId, firstPt);
              pointers.set(e.pointerId, secondPt);
              dragRef.current = {
                mode: "translate",
                pointers,
                prevMid: firstPt.clone().add(secondPt).multiplyScalar(0.5),
              };
            }
          }}
          onPointerMove={(e) => {
            if (!dragRef.current || !globeGroupRef.current) return;
            e.stopPropagation();
            if (dragRef.current.mode === "rotate") {
              if (dragRef.current.pointerId !== e.pointerId) return;
              const delta = e.point.clone().sub(dragRef.current.prevPoint);
              const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
              const up    = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
              const k = 3.0;
              globeGroupRef.current.rotation.y += delta.dot(right) * k;
              globeGroupRef.current.rotation.x = Math.max(
                -Math.PI / 2,
                Math.min(Math.PI / 2, globeGroupRef.current.rotation.x - delta.dot(up) * k),
              );
              dragRef.current.prevPoint = e.point.clone();
            } else {
              if (!dragRef.current.pointers.has(e.pointerId)) return;
              dragRef.current.pointers.set(e.pointerId, e.point.clone());
              const pts = [...dragRef.current.pointers.values()];
              const newMid = pts[0]!.clone().add(pts[1]!).multiplyScalar(0.5);
              globeGroupRef.current.position.add(newMid.clone().sub(dragRef.current.prevMid));
              dragRef.current.prevMid = newMid;
            }
          }}
          onPointerUp={(e) => {
            if (!dragRef.current) return;
            e.stopPropagation();
            try { (e.target as any).releasePointerCapture(e.pointerId); } catch {}
            if (dragRef.current.mode === "rotate") {
              if (dragRef.current.pointerId !== e.pointerId) return;
              dragRef.current = null;
              if (orbitRef.current) orbitRef.current.enabled = true;
            } else {
              dragRef.current.pointers.delete(e.pointerId);
              if (dragRef.current.pointers.size === 1) {
                // Fall back to single-hand rotate with whichever controller remains.
                const [[remainId, remainPt]] = [...dragRef.current.pointers.entries()];
                dragRef.current = { mode: "rotate", prevPoint: remainPt!.clone(), pointerId: remainId! };
              } else {
                dragRef.current = null;
                if (orbitRef.current) orbitRef.current.enabled = true;
              }
            }
          }}
        >
          <sphereGeometry args={[GLOBE_R_BOUND, 64, 32]} />
          <shaderMaterial
            ref={matRef}
            glslVersion={THREE.GLSL3}
            vertexShader={GLOBE_VERT}
            fragmentShader={GLOBE_FRAG}
            uniforms={uniforms}
            transparent
            depthWrite={false}
            side={THREE.FrontSide}
          />
        </mesh>
        {/* Land/geography relief — sits just inside the data shell (nearer to
            camera than the invisible occluder below, so it isn't hidden by it)
            and shows through wherever the shell has no data. */}
        {globeState.showLand && landTex && (
          <mesh geometry={landGeo}>
            {/* toneMapped=false: R3F's Canvas applies ACES filmic tone mapping
                by default, which washes out ordinary textured materials. The
                ocean/atmosphere shader writes outColor directly and bypasses
                that curve, so without this the land layer looks faint next
                to it even though it's fully opaque. */}
            <meshBasicMaterial map={landTex} toneMapped={false} />
          </mesh>
        )}
        {/* Invisible occluder: the ray-marched shell never writes depth (it isn't
            a real surface), so without this the graticule's far hemisphere would
            render unoccluded on top of the globe instead of being hidden behind it. */}
        <mesh renderOrder={-1}>
          <sphereGeometry args={[GLOBE_R_OUT * 0.999, 32, 16]} />
          <meshBasicMaterial colorWrite={false} depthWrite />
        </mesh>
        <GeoGrid />
        <WindOverlay state={globeState} />
      </group>
      <ambientLight intensity={0.7} />
      <directionalLight position={[3, 5, 5]} intensity={0.9} />
      <OrbitControls ref={orbitRef} enableDamping dampingFactor={0.08} minDistance={1.2} maxDistance={8} />
      <VRLoadPanel state={globeState} />
      <VRRenderPanel state={globeState} />
      <VRGlobeColorLegend state={globeState} globeGroupRef={globeGroupRef} />
      <VRWindLegend state={globeState} globeGroupRef={globeGroupRef} />
      <VRRenderRangeControl state={globeState} />
      <VRTimelineControl state={globeState} />
    </>
  );
}

// ─── Quality options (shared between panel and overlay) ──────────────────────

const QUALITY_OPTS = [
  { v: -4,  label: "High" },
  { v: -8,  label: "Medium" },
  { v: -12, label: "Low" },
  { v: -15, label: "Very low" },
];

// ─── GlobeControlsPanel — embeds into the dashboard right panel ───────────────

export function GlobeControlsPanel({ state }: { state: GlobeDataState }) {
  const lbl = "font-mono text-[11px] uppercase tracking-widest text-white/35";

  const varBtn = (active: boolean) =>
    cn(
      "w-full py-2 px-3 rounded text-left text-[12px] font-mono font-semibold border transition-colors",
      active
        ? "border-[#00c4a8]/50 bg-[#00c4a8]/10 text-[#00c4a8]"
        : "border-white/[0.07] text-white/35 hover:text-white/55 hover:border-white/15",
    );

  const qualBtn = (active: boolean) =>
    cn(
      "flex-1 py-1.5 font-mono text-[11px] font-bold border-y border-r first:border-l first:rounded-l last:rounded-r transition-colors",
      active
        ? "border-[#00c4a8]/50 bg-[#00c4a8]/10 text-[#00c4a8]"
        : "border-white/[0.07] text-white/28 hover:text-white/50",
    );

  return (
    <div className="space-y-4 px-3 pb-3 pt-2">
      {/* Dataset */}
      <div className="space-y-1.5">
        <p className={lbl}>Dataset</p>
        <select
          value={state.group}
          onChange={(e) => state.setGroup(e.target.value as DatasetGroupId)}
          className="w-full bg-transparent border border-white/[0.07] rounded px-2 py-2 font-mono text-[12px] font-semibold text-white/70 focus:outline-none focus:border-[#00c4a8]/40 [color-scheme:dark]"
        >
          {DATASET_GROUPS.map((g) => (
            <option key={g.id} value={g.id} className="bg-[#0a0f1a] text-white/80">
              {g.label}
            </option>
          ))}
        </select>
      </div>

      {/* Variable */}
      <div className="space-y-1.5">
        <p className={lbl}>Variable</p>
        {state.variables.map((v) => (
          <button key={v} type="button" onClick={() => state.setVariable(v)} className={varBtn(state.variable === v)}>
            {datasetDef(v).label}
          </button>
        ))}
      </div>

      {/* CMIP6 experiment — historical (1950–2014) vs SSP futures (2015–2100) */}
      {state.group === "cmip6" && (
        <div>
          <p className={cn(lbl, "mb-1.5")}>Scenario</p>
          <div className="flex overflow-hidden rounded">
            {CMIP6_SCENARIOS.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => state.setCmip6Scenario(s.id)}
                className={cn(qualBtn(state.cmip6Scenario === s.id), "px-0 text-[9.5px]")}
              >
                {s.label}
              </button>
            ))}
          </div>
          <p className="mt-1.5 text-[9.5px] text-white/30 leading-relaxed">
            {state.cmip6Scenario === "historical"
              ? "Observed-era simulation · valid 1950 – 2014"
              : "Future projection · valid 2015 – 2100"}
          </p>
        </div>
      )}

      {/* Single timestep */}
      <div>
        <p className={lbl}>Current date</p>
        <input
          type="date"
          min={state.dateMin} max={state.dateMax}
          value={state.currentIsoDate}
          onChange={(e) => { if (e.target.value) state.setTime(state.timestepForDate(e.target.value)); }}
          className="mt-1.5 w-full bg-transparent border border-white/[0.07] rounded px-2 py-1.5 font-mono text-[12px] text-white/60 focus:outline-none focus:border-[#00c4a8]/40 [color-scheme:dark]"
        />
        <p className="mt-1 font-mono text-[10px] text-white/30 tabular-nums">{state.currentDateStr}</p>
      </div>

      {/* Multi-frame time range */}
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <p className={lbl}>Start date</p>
            <input
              type="date"
              min={state.dateMin} max={state.dateMax}
              value={state.startDate}
              onChange={(e) => { if (e.target.value) state.setStartDate(e.target.value); }}
              className="mt-1 w-full bg-transparent border border-white/[0.07] rounded px-2 py-1.5 font-mono text-[11px] text-white/60 focus:outline-none focus:border-[#00c4a8]/40 [color-scheme:dark]"
            />
          </div>
          <div>
            <p className={lbl}>End date</p>
            <input
              type="date"
              min={state.dateMin} max={state.dateMax}
              value={state.endDate}
              onChange={(e) => { if (e.target.value) state.setEndDate(e.target.value); }}
              className="mt-1 w-full bg-transparent border border-white/[0.07] rounded px-2 py-1.5 font-mono text-[11px] text-white/60 focus:outline-none focus:border-[#00c4a8]/40 [color-scheme:dark]"
            />
          </div>
        </div>

        {/* Quick-fill end date from start date */}
        {(() => {
          const addDays = (days: number) => {
            const d = new Date(state.startDate + "T00:00:00Z");
            d.setUTCDate(d.getUTCDate() + days);
            const s = d.toISOString().slice(0, 10);
            state.setEndDate(s > state.dateMax ? state.dateMax : s);
          };
          const qBtn = "flex-1 py-1 font-mono text-[10px] font-bold border border-white/[0.07] text-white/40 rounded hover:text-white/65 hover:border-white/20 transition-colors";
          return (
            <div className="flex gap-1.5">
              <button type="button" className={qBtn} onClick={() => addDays(1)}>+24h</button>
              <button type="button" className={qBtn} onClick={() => addDays(180)}>+6 mo</button>
              <button type="button" className={qBtn} onClick={() => addDays(365)}>+1 yr</button>
            </div>
          );
        })()}

        <div>
          <p className={lbl}>Frames</p>
          <input
            type="number" min={2} max={20} value={state.nFramesTarget}
            onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) state.setNFramesTarget(Math.max(2, Math.min(20, v))); }}
            className="mt-1 w-full bg-transparent border border-white/[0.07] rounded px-2 py-1.5 font-mono text-[12px] text-white/60 focus:outline-none focus:border-[#00c4a8]/40 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none"
          />
        </div>

        {/* Est. fetch */}
        {(() => {
          const n = state.nFramesTarget;
          const layers = state.loadedRange[1] - state.loadedRange[0] + 1;
          const estBytes = 6936 * Math.pow(2, state.quality + 15) * layers;
          const fmt = (b: number) => b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / (1024 * 1024)).toFixed(1)} MB`;
          return (
            <div className="rounded border border-white/[0.07] bg-white/[0.02] px-2.5 py-2 font-mono text-[10.5px]">
              <p className="text-[9px] font-bold uppercase tracking-widest text-white/28 mb-1">Est. fetch</p>
              <p className="text-white/40">{n} frames × {fmt(estBytes)}/frame</p>
              <p className="mt-0.5 font-semibold text-[12px] text-white/65">≈ {fmt(estBytes * n)}</p>
            </div>
          );
        })()}

        <button
          type="button"
          onClick={state.fetchFrames}
          disabled={state.fetchingFrames}
          className={cn(
            "w-full py-2.5 rounded font-mono text-[12px] font-bold uppercase tracking-widest transition-colors",
            state.fetchingFrames
              ? "bg-white/[0.04] text-white/25 cursor-not-allowed"
              : "bg-[#00c4a8] text-black hover:bg-[#00d4b8]",
          )}
        >
          {state.fetchingFrames
            ? `Fetching ${state.frames.length} / ${state.nFramesTarget}…`
            : "▶ Render"}
        </button>

        {state.frames.length > 0 && !state.fetchingFrames && (
          <p className="font-mono text-[10px] text-[#00c4a8]/55 text-center">
            {state.frames.length} frames loaded
          </p>
        )}
      </div>

      {/* Depth / render range (hidden for single-layer 2D groups like CMIP6).
          Atmosphere levels count DOWN from the top of the atmosphere (0 = TOA,
          51 = surface) — the opposite of ocean depth levels — so for that
          group both sliders are direction-flipped: dragging left always moves
          toward the surface, dragging right always moves away from it,
          matching the ocean sliders' feel. */}
      {state.depthLevels > 1 && (() => {
        const atm = state.group === "dyamond_atm";
        const N = state.depthLevels;
        const [lr0, lr1] = state.loadedRange;
        return (
          <div className="space-y-1">
            <p className={lbl}>
              {state.depthName} layers <span className="text-white/50 ml-1">{lr0}–{lr1}</span>
            </p>
            <p className="font-mono text-[10px] text-white/35">
              {formatLevelPhysicalOnly(state.group, lr0, N)}
              {" – "}
              {formatLevelPhysicalOnly(state.group, lr1, N)}
            </p>
            <input
              type="range" min={0} max={N - 1} step={1}
              value={atm ? N - 1 - lr0 : lr1}
              onChange={(e) => {
                const v = Number(e.target.value);
                state.setLoadedRange(atm ? [N - 1 - v, N - 1] : [lr0, v]);
              }}
              className="w-full accent-[#00c4a8]"
            />
            <p className={cn(lbl, "mt-3")}>
              Render range{" "}
              <span className="text-white/50 ml-1">
                {atm ? "level" : "depth"} {state.renderRange[1]} → {state.renderRange[0]}
              </span>
            </p>
            <p className="font-mono text-[10px] text-white/35">
              {formatLevelPhysicalOnly(state.group, state.renderRange[1], N)}
              {" → "}
              {formatLevelPhysicalOnly(state.group, state.renderRange[0], N)}
            </p>
            <input
              type="range" min={lr0} max={lr1} step={1}
              value={atm ? lr0 + lr1 - state.renderRange[0] : state.renderRange[0]}
              onChange={(e) => {
                const v = Number(e.target.value);
                state.setRenderRange(atm ? [lr0 + lr1 - v, lr1] : [v, lr1]);
              }}
              className="w-full accent-[#00c4a8]"
            />
          </div>
        );
      })()}

      {/* Resolution */}
      <div>
        <p className={cn(lbl, "mb-1.5")}>Resolution</p>
        <div className="flex overflow-hidden rounded">
          {QUALITY_OPTS.map(({ v, label }) => (
            <button key={v} type="button" onClick={() => state.setQuality(v)} className={qualBtn(state.quality === v)}>
              {label}
            </button>
          ))}
        </div>
        {state.quality === -4 && (
          <p className="mt-1.5 text-[9.5px] text-white/30 leading-relaxed">
            High quality exceeds the GPU 3D-texture limit (2048 px) — backend auto-downsamples to fit.
          </p>
        )}
      </div>

      {/* Land overlay */}
      <div>
        <p className={cn(lbl, "mb-1.5")}>Land overlay</p>
        <button
          type="button"
          onClick={() => state.setShowLand(!state.showLand)}
          className={varBtn(state.showLand)}
        >
          {state.showLand ? "Showing land" : "Hidden"}
        </button>
      </div>

      {/* Wind overlay — coupled DYAMOND atmosphere winds over the ocean SST */}
      {state.group === "dyamond_ocean" && (
        <div>
          <p className={cn(lbl, "mb-1.5")}>Wind overlay</p>
          <button
            type="button"
            onClick={() => state.setWindOverlay(!state.windOverlay)}
            className={varBtn(state.windOverlay)}
          >
            {state.windOverlay ? (state.windLoading ? "Loading wind…" : "Showing wind") : "Off"}
          </button>
          {state.windOverlay && (
            <div className="mt-2 space-y-2">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className={lbl}>Wind level</span>
                  <span className="font-mono text-[10px] text-[#00c4a8]">
                    {formatLevelPhysicalOnly("dyamond_atm", state.windLevel, 52)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={51}
                  step={1}
                  // GEOS levels run 0 = top-of-atmosphere … 51 = surface. Flip so
                  // dragging left = toward the surface, matching the depth sliders.
                  value={51 - state.windLevel}
                  onChange={(e) => state.setWindLevel(51 - Number(e.target.value))}
                  className="w-full accent-[#00c4a8]"
                />
              </div>
              <div>
                <p className={cn(lbl, "mb-1")}>Particle density</p>
                <div className="flex overflow-hidden rounded">
                  {WIND_DENSITIES.map((d) => (
                    <button key={d.n} type="button" onClick={() => state.setWindDensity(d.n)} className={qualBtn(state.windDensity === d.n)}>
                      {d.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Data size */}
      {(() => {
        const n = state.loadedRange[1] - state.loadedRange[0] + 1;
        // Empirical reference: q=-15 → ~6936 bytes/layer. Each quality step doubles linear res → ×4 data.
        const estBytes = 6936 * Math.pow(2, state.quality + 15) * n;
        const fmtBytes = (b: number) => b < 1024 * 1024
          ? `${(b / 1024).toFixed(0)} KB`
          : `${(b / (1024 * 1024)).toFixed(1)} MB`;
        const isLarge = estBytes > 50 * 1024 * 1024;
        return (
          <div className={cn(
            "rounded border px-2.5 py-2 font-mono text-[10.5px]",
            isLarge ? "border-amber-500/30 bg-amber-500/5" : "border-white/[0.07] bg-white/[0.02]",
          )}>
            <p className="text-[9px] font-bold uppercase tracking-widest text-white/28 mb-1">Est. download</p>
            <p className="text-white/45">
              {n} layer{n !== 1 ? "s" : ""} at quality {state.quality}
            </p>
            <p className={cn("mt-0.5 font-semibold text-[12px]", isLarge ? "text-amber-400" : "text-white/65")}>
              ≈ {fmtBytes(estBytes)}
              {isLarge && <span className="ml-1.5 text-[10px] font-normal text-amber-400/70">— large fetch</span>}
            </p>
            {state.fetchedBytes !== null && (
              <p className="mt-1 text-[10px] text-[#00c4a8]/60">
                Last fetch: {fmtBytes(state.fetchedBytes)} actual
              </p>
            )}
            <p className="mt-1.5 text-[9.5px] text-white/22 leading-relaxed">
              Data fetched from remote OpenVisus server in one request. Use <span className="text-white/40">Very low</span> for fastest exploration.
            </p>
          </div>
        );
      })()}
    </div>
  );
}

// ─── VR panel primitives ──────────────────────────────────────────────────────

function VRBtn({
  label, active = false, disabled = false,
  x, y, w = 0.12, h = 0.030, onSelect,
}: {
  label: string; active?: boolean; disabled?: boolean;
  x: number; y: number; w?: number; h?: number; onSelect: () => void;
}) {
  const [hov, setHov] = useState(false);
  return (
    <group position={[x, y, 0.003]}>
      <mesh
        onPointerEnter={() => { if (!disabled) setHov(true); }}
        onPointerLeave={() => setHov(false)}
        onPointerDown={(e) => { e.stopPropagation(); if (!disabled) onSelect(); }}
      >
        <planeGeometry args={[w, h]} />
        <meshBasicMaterial color={disabled ? "#0d1c30" : active ? "#00c4a8" : hov ? "#1e3a5a" : "#0d2040"} />
      </mesh>
      <Text
        key={`${active}-${disabled}`}
        position={[0, 0, 0.001]}
        fontSize={Math.min(h * 0.44, 0.013)}
        color={disabled ? "#ffffff20" : active ? "#000000" : "#ffffffaa"}
        anchorX="center"
        anchorY="middle"
        maxWidth={w - 0.006}
        overflowWrap="break-word"
      >
        {label}
      </Text>
    </group>
  );
}

function VRLabel({
  text, x, y, size = 0.011, color = "#ffffff55",
  align = "left" as "left" | "center" | "right", maxWidth = 0.41,
}: {
  text: string; x: number; y: number; size?: number; color?: string;
  align?: "left" | "center" | "right"; maxWidth?: number;
}) {
  return (
    <Text
      position={[x, y, 0.003]}
      fontSize={size}
      color={color}
      anchorX={align}
      anchorY="middle"
      maxWidth={maxWidth}
    >
      {text}
    </Text>
  );
}

function VRDivider({ y }: { y: number }) {
  return (
    <mesh position={[0, y, 0.002]}>
      <planeGeometry args={[0.41, 0.001]} />
      <meshBasicMaterial color="#1e3d5a" transparent opacity={0.5} />
    </mesh>
  );
}

// ─── VRSlider ─────────────────────────────────────────────────────────────────

function VRSlider({
  value, min, max, x, y, trackW = 0.25, panelGroupRef, onChange,
}: {
  value: number; min: number; max: number;
  x: number; y: number; trackW?: number;
  panelGroupRef: React.RefObject<THREE.Group | null>;
  onChange: (v: number) => void;
}) {
  // useState (not ref) so the capture overlay mounts/unmounts reactively
  const [dragging, setDragging] = useState(false);
  const [hov, setHov] = useState(false);
  const dragPointerIdRef = useRef<number | null>(null);

  const frac = max > min ? Math.max(0, Math.min(1, (value - min) / (max - min))) : 0;
  const startX = x - trackW / 2;
  const handleX = startX + frac * trackW;
  const fillW = Math.max(0.001, frac * trackW);
  const fillX = startX + fillW / 2;

  const applyDrag = (e: any) => {
    if (!panelGroupRef.current) return;
    const local = panelGroupRef.current.worldToLocal(e.point.clone());
    const f = Math.max(0, Math.min(1, (local.x - startX) / trackW));
    onChange(Math.round(min + f * (max - min)));
  };

  const startDrag = (e: any) => {
    e.stopPropagation();
    dragPointerIdRef.current = e.pointerId;
    setDragging(true);
    applyDrag(e);
  };

  const endDrag = (e: any) => {
    e.stopPropagation();
    dragPointerIdRef.current = null;
    setDragging(false);
  };

  return (
    <group>
      {/* Track */}
      <mesh position={[x, y, 0.002]}>
        <planeGeometry args={[trackW, 0.006]} />
        <meshBasicMaterial color="#1a3050" />
      </mesh>
      {/* Fill */}
      <mesh position={[fillX, y, 0.003]}>
        <planeGeometry args={[fillW, 0.006]} />
        <meshBasicMaterial color="#00c4a8" />
      </mesh>
      {/* Hit zone along the track */}
      <mesh position={[x, y, 0.004]} onPointerDown={startDrag} onPointerUp={endDrag}>
        <planeGeometry args={[trackW + 0.02, 0.036]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      {/* Handle dot — also initiates drag */}
      <mesh
        position={[handleX, y, 0.005]}
        onPointerEnter={() => setHov(true)}
        onPointerLeave={() => setHov(false)}
        onPointerDown={startDrag}
        onPointerUp={endDrag}
      >
        <circleGeometry args={[hov || dragging ? 0.014 : 0.011, 24]} />
        <meshBasicMaterial color={dragging ? "#ffffff" : "#00c4a8"} />
      </mesh>
      {/* Full-panel capture overlay — mounts on drag start; sits above all elements so
          pointermove fires even when the controller drifts off the narrow track.
          pointerId guard ensures only the initiating controller drives the drag. */}
      {dragging && (
        <mesh
          position={[0, 0, 0.015]}
          onPointerMove={(e) => {
            if (e.pointerId !== dragPointerIdRef.current) return;
            e.stopPropagation();
            applyDrag(e);
          }}
          onPointerUp={(e) => {
            if (e.pointerId !== dragPointerIdRef.current) return;
            e.stopPropagation();
            dragPointerIdRef.current = null;
            setDragging(false);
          }}
        >
          <planeGeometry args={[4, 4]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

// ─── VRDatePicker — Y/M/D column spinner ──────────────────────────────────────

function VRDatePicker({
  label, value, y, min, max, onChange,
}: {
  label: string; value: string; y: number; min: string; max: string;
  onChange: (d: string) => void;
}) {
  const lxd = -0.244; // -0.52/2 + 0.016
  const [yr, mo, dy] = value.split("-").map(Number) as [number, number, number];
  const fmt2 = (n: number) => String(n).padStart(2, "0");
  // mirrors the desktop <input min max> date bounds for the active dataset group
  const clampDate = (d: string) => (d < min ? min : d > max ? max : d);
  const minYr = +min.slice(0, 4);
  const maxYr = +max.slice(0, 4);
  // days in month mm (1-indexed) of year yy
  const daysIn = (yy: number, mm: number) => new Date(yy, mm, 0).getDate();

  const set = (ny: number, nm: number, nd: number) => {
    // carry month overflow into year
    while (nm < 1)  { nm += 12; ny -= 1; }
    while (nm > 12) { nm -= 12; ny += 1; }
    ny = Math.max(minYr, Math.min(maxYr, ny));
    // wrap day within the resolved month
    const maxD = daysIn(ny, nm);
    if (nd < 1) nd = maxD;
    if (nd > maxD) nd = 1;
    // final clamp matches desktop min/max as full ISO date strings
    onChange(clampDate(`${ny}-${fmt2(nm)}-${fmt2(nd)}`));
  };

  const by = y - 0.038;
  const bw = 0.034, bh = 0.026;
  const yc = -0.163, mc = 0, dc = 0.163; // column centres

  return (
    <group>
      <VRLabel text={label} x={lxd} y={y} size={0.010} color="#ffffff55" />
      <VRLabel text={value} x={0.244} y={y} size={0.011} color="#ffffffbb" align="right" />

      <VRBtn label="◀" x={yc - 0.052} y={by} w={bw} h={bh} onSelect={() => set(yr - 1, mo, dy)} />
      <VRLabel text={String(yr)} x={yc} y={by} size={0.012} color="#ffffffdd" align="center" />
      <VRBtn label="▶" x={yc + 0.052} y={by} w={bw} h={bh} onSelect={() => set(yr + 1, mo, dy)} />

      <VRBtn label="◀" x={mc - 0.052} y={by} w={bw} h={bh} onSelect={() => set(yr, mo - 1, dy)} />
      <VRLabel text={fmt2(mo)} x={mc} y={by} size={0.012} color="#ffffffdd" align="center" />
      <VRBtn label="▶" x={mc + 0.052} y={by} w={bw} h={bh} onSelect={() => set(yr, mo + 1, dy)} />

      <VRBtn label="◀" x={dc - 0.052} y={by} w={bw} h={bh} onSelect={() => set(yr, mo, dy - 1)} />
      <VRLabel text={fmt2(dy)} x={dc} y={by} size={0.012} color="#ffffffdd" align="center" />
      <VRBtn label="▶" x={dc + 0.052} y={by} w={bw} h={bh} onSelect={() => set(yr, mo, dy + 1)} />
    </group>
  );
}

// ─── VRDragHandle — pill bar at panel bottom; drag to reposition panel ────────

function VRDragHandle({
  bottomY, onDragStart, onDragMove, onDragEnd,
}: {
  // Y-coordinate (in the panel group's local space) of the backdrop's bottom
  // edge. Passing this directly — rather than a bare panelH assumed to be
  // centred on y=0 — keeps the handle glued to the true bottom edge even when
  // the backdrop itself is re-centred lower to grow downward (see panelH/
  // background-offset math in VRLoadPanel).
  bottomY: number;
  onDragStart: (e: any) => void;
  onDragMove: (e: any) => void;
  onDragEnd: () => void;
}) {
  const [hov, setHov] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragPointerIdRef = useRef<number | null>(null);
  const hy = bottomY - 0.014;
  return (
    <group>
      <mesh
        position={[0, hy, 0.002]}
        onPointerEnter={() => setHov(true)}
        onPointerLeave={() => { if (!dragging) setHov(false); }}
        onPointerDown={(e) => {
          e.stopPropagation();
          dragPointerIdRef.current = e.pointerId;
          setDragging(true);
          onDragStart(e);
        }}
        onPointerUp={(e) => {
          e.stopPropagation();
          dragPointerIdRef.current = null;
          setDragging(false);
          setHov(false);
          onDragEnd();
        }}
      >
        <planeGeometry args={[0.14, 0.018]} />
        <meshBasicMaterial color={hov || dragging ? "#00c4a8" : "#2a3a4a"} />
      </mesh>
      {dragging && (
        <mesh
          position={[0, 0, 0.015]}
          onPointerMove={(e) => {
            if (e.pointerId !== dragPointerIdRef.current) return;
            e.stopPropagation();
            onDragMove(e);
          }}
          onPointerUp={(e) => {
            if (e.pointerId !== dragPointerIdRef.current) return;
            e.stopPropagation();
            dragPointerIdRef.current = null;
            setDragging(false);
            setHov(false);
            onDragEnd();
          }}
        >
          <planeGeometry args={[20, 20]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

// ─── VRLoadPanel — DATASET / VARIABLE / RESOLUTION / DATE / DEPTH → LOAD ──────

function VRLoadPanel({ state }: { state: GlobeDataState }) {
  const session = useXR((s: any) => s.session as XRSession | null);
  const panelRef = useRef<THREE.Group | null>(null);
  const [panelPos, setPanelPos] = useState<[number, number, number]>([-1.05, 1.55, -0.9]);
  // Yaw that exactly faces the user's assumed origin (0, y, 0) from this
  // panel's spawn position — recompute alongside panelPos if either changes.
  const [panelRotY, setPanelRotY] = useState(Math.atan2(1.05, 0.9));
  const dragStartRef = useRef<{
    t: number;                  // distance along ray to panel centre at drag-start
    perpWorld: THREE.Vector3;   // world-space offset from on-ray point to panel centre
    startAzimuth: number;       // atan2(d.x, d.z) of controller ray at drag-start
    startPanelRotY: number;     // panel Y rotation at drag-start
  } | null>(null);
  if (!session) return null;

  const handleDragStart = (e: any) => {
    const d = e.ray.direction.clone().normalize();
    const o = e.ray.origin;
    const p = new THREE.Vector3(...panelPos);
    const t = Math.max(0.05, p.clone().sub(o).dot(d));
    const perpWorld = p.clone().sub(o.clone().add(d.clone().multiplyScalar(t)));
    dragStartRef.current = {
      t,
      perpWorld,
      startAzimuth: Math.atan2(d.x, d.z),
      startPanelRotY: panelRotY,
    };
  };
  const handleDragMove = (e: any) => {
    if (!dragStartRef.current) return;
    const { t, perpWorld, startAzimuth, startPanelRotY } = dragStartRef.current;
    const d = e.ray.direction.clone().normalize();
    // Y-rotation delta: how much the controller has turned since drag-start
    let delta = Math.atan2(d.x, d.z) - startAzimuth;
    if (delta >  Math.PI) delta -= 2 * Math.PI;
    if (delta < -Math.PI) delta += 2 * Math.PI;
    // rotate the perp offset by the same delta so panel stays attached to the controller
    const rotatedPerp = perpWorld.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), delta);
    const newPos = e.ray.origin.clone().add(d.clone().multiplyScalar(t)).add(rotatedPerp);
    setPanelPos([newPos.x, newPos.y, newPos.z]);
    setPanelRotY(startPanelRotY + delta);
  };
  const handleDragEnd = () => { dragStartRef.current = null; };

  const W = 0.52;
  const H = 0.5208; // 0.46 + 1.6 row-units, to fit the LAND toggle row inserted below
  const lx = -W / 2 + 0.016;
  const bw = W - 0.032;
  const r = (n: number) => H / 2 - 0.028 - n * 0.038;
  // Variable-dependent row offsets. The VARIABLE section wraps to rows of 3
  // buttons (the atmosphere group has 9 variables, CMIP6 has 9), pushing the
  // rows below it down by `vo`; the WIND section (3.2 row-units) only appears
  // for the DYAMOND ocean group and pushes the rows below IT down by `wo`.
  // Rather than shift every row's r(n), the top rows keep their fixed layout
  // and the panel grows downward: rv() shifts rows below the variable block,
  // rb() shifts rows below the wind block, and panelH extends the backdrop
  // (centred lower) so the added space is at the bottom.
  const nVarRows = Math.max(1, Math.ceil(state.variables.length / 3));
  const vo = (nVarRows - 1) * 0.8;
  const wo = state.group === "dyamond_ocean" ? 3.2 : 0;
  const rv = (n: number) => r(n + vo);
  const rb = (n: number) => r(n + vo + wo);
  // Extra breathing room below the LOAD button so the drag handle doesn't
  // hug it — added only to the backdrop/handle, not to any row position.
  const bottomPad = 0.9;
  const panelH = H + (vo + wo + bottomPad) * 0.038;
  // Backdrop is re-centred lower by half its growth so the extra height is
  // added below (top edge stays fixed) — bottomY is its true bottom edge,
  // used by the drag handle so it always tracks the panel's actual extent.
  const bgCenterY = -(vo + wo + bottomPad) * 0.038 / 2;
  const bottomY = bgCenterY - panelH / 2;

  // Convert an hour delta into whole timesteps for the group's cadence
  // (1 step/hour for the runs; for daily CMIP6 even the HR buttons move ±1 day).
  const stepTime = (hours: number) => {
    const dtH = GROUP_TIME[state.group].dtMs / 3600_000;
    const steps = Math.sign(hours) * Math.max(1, Math.round(Math.abs(hours) / dtH));
    state.setTime(Math.max(0, Math.min(state.maxTimestep, state.time + steps)));
  };

  const curD = state.currentDate;
  const addYear  = (n: number) => { const nd = new Date(curD); nd.setUTCFullYear(nd.getUTCFullYear() + n); state.setTime(state.dateToTime(nd)); };
  const addMonth = (n: number) => { const nd = new Date(curD); nd.setUTCMonth(nd.getUTCMonth() + n); state.setTime(state.dateToTime(nd)); };

  return (
    <group ref={panelRef} position={panelPos} rotation={[0, panelRotY, 0]}>
      {/* Panel background — onPointerMove registers it as interactive so XR shows ray highlight */}
      <mesh onPointerMove={() => {}} position={[0, bgCenterY, 0]}>
        <planeGeometry args={[W, panelH]} />
        <meshBasicMaterial color="#060c18" transparent opacity={0.94} />
      </mesh>

      {/* ── DATASET ── */}
      <VRLabel text="DATASET" x={lx} y={r(0)} />
      {DATASET_GROUPS.map((g, i) => (
        <VRBtn key={g.id}
          // Short per-group names: with four groups (incl. CMIP6) the full
          // dataset labels no longer fit the panel width, but these stay
          // distinct — unlike the old first-word truncation that made both
          // DYAMOND groups read as just "DYAMOND".
          label={{ ecco: "ECCO", dyamond_ocean: "OCEAN", dyamond_atm: "ATMOS", cmip6: "CMIP6" }[g.id]}
          active={state.group === g.id}
          x={lx + 0.072 + i * 0.118} y={r(1)} w={0.110} h={0.030}
          onSelect={() => state.setGroup(g.id)}
        />
      ))}

      <VRDivider y={r(1.8)} />

      {/* ── VARIABLE (wraps to rows of 3 — atm/CMIP6 have 9 variables) ── */}
      <VRLabel text="VARIABLE" x={lx} y={r(2.3)} />
      {state.variables.map((v, i) => (
        <VRBtn key={v}
          label={datasetDef(v).label.split(" ")[0] ?? v}
          active={state.variable === v}
          x={lx + 0.081 + (i % 3) * 0.163} y={r(3.1 + Math.floor(i / 3) * 0.8)} w={0.130} h={0.030}
          onSelect={() => state.setVariable(v)}
        />
      ))}

      <VRDivider y={rv(3.9)} />

      {/* ── RESOLUTION ── */}
      <VRLabel text="RESOLUTION" x={lx} y={rv(4.4)} />
      {QUALITY_OPTS.map((q, i) => (
        <VRBtn key={q.v}
          label={q.label}
          active={state.quality === q.v}
          x={lx + 0.068 + i * 0.118} y={rv(5.2)} w={0.100} h={0.028}
          onSelect={() => state.setQuality(q.v)}
        />
      ))}

      {/* ── LAND (+ CMIP6 scenario readout — the picker itself is desktop-only) ── */}
      <VRLabel text="LAND" x={lx} y={rv(6.0)} />
      <VRBtn
        label={state.showLand ? "ON" : "OFF"}
        active={state.showLand}
        x={lx + 0.09} y={rv(6.8)} w={0.110} h={0.028}
        onSelect={() => state.setShowLand(!state.showLand)}
      />
      {state.group === "cmip6" && (
        <VRLabel
          text={`SCENARIO  ${(CMIP6_SCENARIOS.find((s) => s.id === state.cmip6Scenario)?.label ?? state.cmip6Scenario).toUpperCase()}`}
          x={lx + 0.20} y={rv(6.8)} size={0.009} color="#7ffcdf"
        />
      )}

      {/* ── WIND (DYAMOND ocean only) ── */}
      {state.group === "dyamond_ocean" && (
        <>
          {/* Same two-row layout as LAND above (title row, then toggle below at
              the same x) so the two ON/OFF buttons line up in one column. */}
          <VRLabel text="WIND" x={lx} y={rv(7.6)} />
          <VRBtn
            label={state.windOverlay ? "ON" : "OFF"}
            active={state.windOverlay}
            x={lx + 0.09} y={rv(8.4)} w={0.110} h={0.028}
            onSelect={() => state.setWindOverlay(!state.windOverlay)}
          />
          <VRLabel
            text={`LEVEL  ${formatLevelPhysicalOnly("dyamond_atm", state.windLevel, 52)}`}
            x={lx} y={rv(9.2)} size={0.009} color="#7ffcdf"
          />
          {/* Particle density (pool size) — S / M / D */}
          {WIND_DENSITIES.map((d, i) => (
            <VRBtn key={d.n}
              label={d.label[0]}
              active={state.windDensity === d.n}
              x={0.10 + i * 0.062} y={rv(8.4)} w={0.052} h={0.026}
              onSelect={() => state.setWindDensity(d.n)}
            />
          ))}
          {/* GEOS levels count down from TOA → flip so left = toward surface. */}
          <VRSlider
            value={51 - state.windLevel}
            min={0}
            max={51}
            x={0} y={rv(10.0)} trackW={0.35}
            panelGroupRef={panelRef}
            onChange={(v) => state.setWindLevel(51 - v)}
          />
        </>
      )}

      <VRDivider y={rb(7.6)} />

      {/* ── CURRENT DATE ── */}
      <VRLabel text="CURRENT DATE" x={lx} y={rb(8.1)} />
      {/* Row 1: YEAR | MON */}
      <VRLabel text="YEAR" x={-0.228} y={rb(8.9)} size={0.008} color="#ffffff55" align="right" />
      <VRBtn label="◀" x={-0.195} y={rb(8.9)} w={0.046} h={0.026} onSelect={() => addYear(-1)} />
      <VRLabel text={String(curD.getUTCFullYear())} x={-0.13} y={rb(8.9)} size={0.012} color="#ffffffcc" align="center" />
      <VRBtn label="▶" x={-0.065} y={rb(8.9)} w={0.046} h={0.026} onSelect={() => addYear(1)} />
      <VRLabel text="MON"  x={0.034} y={rb(8.9)} size={0.008} color="#ffffff55" align="right" />
      <VRBtn label="◀" x={ 0.065} y={rb(8.9)} w={0.046} h={0.026} onSelect={() => addMonth(-1)} />
      <VRLabel text={MONTHS[curD.getUTCMonth()]} x={0.13} y={rb(8.9)} size={0.012} color="#ffffffcc" align="center" />
      <VRBtn label="▶" x={ 0.195} y={rb(8.9)} w={0.046} h={0.026} onSelect={() => addMonth(1)} />
      {/* Row 2: DAY | HR */}
      <VRLabel text="DAY"  x={-0.228} y={rb(9.6)} size={0.008} color="#ffffff55" align="right" />
      <VRBtn label="◀" x={-0.195} y={rb(9.6)} w={0.046} h={0.026} onSelect={() => stepTime(-24)} />
      <VRLabel text={String(curD.getUTCDate()).padStart(2, "0")} x={-0.13} y={rb(9.6)} size={0.012} color="#ffffffcc" align="center" />
      <VRBtn label="▶" x={-0.065} y={rb(9.6)} w={0.046} h={0.026} onSelect={() => stepTime(24)} />
      <VRLabel text="HR"   x={0.034} y={rb(9.6)} size={0.008} color="#ffffff55" align="right" />
      <VRBtn label="◀" x={ 0.065} y={rb(9.6)} w={0.046} h={0.026} onSelect={() => stepTime(-1)} />
      <VRLabel text={String(curD.getUTCHours()).padStart(2, "0")} x={0.13} y={rb(9.6)} size={0.012} color="#ffffffcc" align="center" />
      <VRBtn label="▶" x={ 0.195} y={rb(9.6)} w={0.046} h={0.026} onSelect={() => stepTime(1)} />

      <VRDivider y={rb(10.3)} />

      {/* ── DEPTH LAYERS (single-layer 2D groups show a static label instead) ── */}
      {/* Single line, same plain style as the other section headers
          (DATASET/VARIABLE/RESOLUTION/LAND/WIND/CURRENT DATE) — no accent
          color and no background box behind it. */}
      <VRLabel
        text={state.depthLevels > 1
          ? `${state.depthName.toUpperCase()} LAYERS  ${state.loadedRange[0]}–${state.loadedRange[1]}  (${formatLevelPhysicalOnly(state.group, state.loadedRange[0], state.depthLevels)} → ${formatLevelPhysicalOnly(state.group, state.loadedRange[1], state.depthLevels)})`
          : "2D SURFACE FIELD · DAILY"}
        x={lx} y={rb(10.8)} maxWidth={bw}
      />
      {/* Atmosphere levels count down from the TOA (0 = top, 51 = surface),
          so the slider is direction-flipped there: dragging left always moves
          toward the surface, matching the ocean groups' feel. */}
      {state.depthLevels > 1 && (
        <VRSlider
          value={state.group === "dyamond_atm"
            ? state.depthLevels - 1 - state.loadedRange[0]
            : state.loadedRange[1]}
          min={state.group === "dyamond_atm" ? 0 : state.loadedRange[0]}
          max={state.depthLevels - 1}
          x={0} y={rb(11.6)} trackW={0.35}
          panelGroupRef={panelRef}
          onChange={(v) => state.setLoadedRange(state.group === "dyamond_atm"
            ? [state.depthLevels - 1 - v, state.depthLevels - 1]
            : [state.loadedRange[0], v])}
        />
      )}
      <VRBtn
        label={state.loading ? "Loading…" : "⬇  LOAD"}
        active={!state.loading} disabled={state.loading}
        x={0} y={rb(12.9)} w={bw} h={0.034}
        onSelect={state.fetchSingle}
      />

      <VRDragHandle bottomY={bottomY} onDragStart={handleDragStart} onDragMove={handleDragMove} onDragEnd={handleDragEnd} />
    </group>
  );
}

// ─── VRRenderPanel — date range + multi-frame RENDER ─────────────────────────

function VRRenderPanel({ state }: { state: GlobeDataState }) {
  const session = useXR((s: any) => s.session as XRSession | null);
  const panelRef = useRef<THREE.Group | null>(null);
  const [panelPos, setPanelPos] = useState<[number, number, number]>([1.05, 1.55, -0.9]);
  // Mirror of VRLoadPanel's yaw — exactly faces the user's assumed origin
  // (0, y, 0) from this panel's spawn position.
  const [panelRotY, setPanelRotY] = useState(-Math.atan2(1.05, 0.9));
  const dragStartRef = useRef<{
    t: number;
    perpWorld: THREE.Vector3;
    startAzimuth: number;
    startPanelRotY: number;
  } | null>(null);
  if (!session) return null;

  const handleDragStart = (e: any) => {
    const d = e.ray.direction.clone().normalize();
    const o = e.ray.origin;
    const p = new THREE.Vector3(...panelPos);
    const t = Math.max(0.05, p.clone().sub(o).dot(d));
    const perpWorld = p.clone().sub(o.clone().add(d.clone().multiplyScalar(t)));
    dragStartRef.current = { t, perpWorld, startAzimuth: Math.atan2(d.x, d.z), startPanelRotY: panelRotY };
  };
  const handleDragMove = (e: any) => {
    if (!dragStartRef.current) return;
    const { t, perpWorld, startAzimuth, startPanelRotY } = dragStartRef.current;
    const d = e.ray.direction.clone().normalize();
    let delta = Math.atan2(d.x, d.z) - startAzimuth;
    if (delta >  Math.PI) delta -= 2 * Math.PI;
    if (delta < -Math.PI) delta += 2 * Math.PI;
    const rotatedPerp = perpWorld.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), delta);
    const newPos = e.ray.origin.clone().add(d.clone().multiplyScalar(t)).add(rotatedPerp);
    setPanelPos([newPos.x, newPos.y, newPos.z]);
    setPanelRotY(startPanelRotY + delta);
  };
  const handleDragEnd = () => { dragStartRef.current = null; };

  const W = 0.52;
  const H = 0.36;
  const lx = -W / 2 + 0.016;
  const bw = W - 0.032;
  const r = (n: number) => H / 2 - 0.028 - n * 0.038;

  const shiftEndDate = (days: number) => {
    const d = new Date(state.startDate + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + days);
    const s = d.toISOString().slice(0, 10);
    state.setEndDate(s > state.dateMax ? state.dateMax : s);
  };
  const layers = state.loadedRange[1] - state.loadedRange[0] + 1;
  const estBpF = 6936 * Math.pow(2, state.quality + 15) * layers;
  const fmtB = (b: number) =>
    b < 1048576 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1048576).toFixed(1)} MB`;

  return (
    <group ref={panelRef} position={panelPos} rotation={[0, panelRotY, 0]}>
      <mesh onPointerMove={() => {}}>
        <planeGeometry args={[W, H]} />
        <meshBasicMaterial color="#060c18" transparent opacity={0.94} />
      </mesh>

      {/* ── DATE RANGE ── */}
      <VRLabel text="DATE RANGE" x={lx} y={r(0)} />
      <VRDatePicker label="START" value={state.startDate} y={r(0.6)} min={state.dateMin} max={state.dateMax} onChange={state.setStartDate} />
      <VRDatePicker label="END"   value={state.endDate}   y={r(2.3)} min={state.dateMin} max={state.dateMax} onChange={state.setEndDate} />
      <VRBtn label="+24h"  x={-0.147} y={r(4.1)} w={0.100} h={0.028} onSelect={() => shiftEndDate(1)} />
      <VRBtn label="+6 mo" x={ 0.000} y={r(4.1)} w={0.100} h={0.028} onSelect={() => shiftEndDate(180)} />
      <VRBtn label="+1 yr" x={ 0.147} y={r(4.1)} w={0.100} h={0.028} onSelect={() => shiftEndDate(365)} />

      <VRDivider y={r(4.9)} />

      {/* ── FRAMES & RENDER ── */}
      <VRLabel text="FRAMES" x={lx} y={r(5.5)} />
      <VRBtn label="−" x={0.080} y={r(5.5)} w={0.036} h={0.024}
        onSelect={() => state.setNFramesTarget(Math.max(2, state.nFramesTarget - 1))} />
      <VRLabel text={String(state.nFramesTarget)} x={0.134} y={r(5.5)} size={0.013} color="#ffffffcc" align="center" />
      <VRBtn label="+" x={0.188} y={r(5.5)} w={0.036} h={0.024}
        onSelect={() => state.setNFramesTarget(Math.min(20, state.nFramesTarget + 1))} />
      <VRLabel
        text={`${state.nFramesTarget} × ${fmtB(estBpF)} ≈ ${fmtB(estBpF * state.nFramesTarget)}`}
        x={0} y={r(6.2)} size={0.010} color="#ffffff45" align="center"
      />
      <VRBtn
        label={state.fetchingFrames
          ? `Fetching ${state.frames.length} / ${state.nFramesTarget}…`
          : "▶  RENDER"}
        active={!state.fetchingFrames} disabled={state.fetchingFrames}
        x={0} y={r(7.1)} w={bw} h={0.034}
        onSelect={state.fetchFrames}
      />
      {state.frames.length > 0 && !state.fetchingFrames && (
        <VRLabel text={`${state.frames.length} frames loaded`} x={0} y={r(7.8)} size={0.010} color="#00c4a855" align="center" />
      )}

      <VRDragHandle bottomY={-H / 2} onDragStart={handleDragStart} onDragMove={handleDragMove} onDragEnd={handleDragEnd} />
    </group>
  );
}

// ─── VRRenderRangeControl — hold X (left controller) to show a depth slider ──
//
// Pressing the X button summons a vertical slider anchored to the controller.
// While held, moving the controller up/down adjusts renderRange[0] (the shallow
// depth cutoff). The slider disappears when the button is released.

function VRRenderRangeControl({ state }: { state: GlobeDataState }) {
  const session    = useXR((s: any) => s.session as XRSession | null);
  const selfRef    = useRef<THREE.Group | null>(null);
  const [active, setActive] = useState(false);
  const startRef   = useRef<{ ctrlY: number; startValue: number } | null>(null);
  const lastEmitRef = useRef(0);
  const spawnRef   = useRef<{ pos: THREE.Vector3; rotY: number } | null>(null);

  useFrame((r3f, _, xrFrame) => {
    // Apply stored spawn transform as soon as the group mounts (one-shot after setActive)
    if (selfRef.current && spawnRef.current) {
      selfRef.current.position.copy(spawnRef.current.pos);
      selfRef.current.rotation.y = spawnRef.current.rotY;
      spawnRef.current = null;
    }

    if (!session || !xrFrame) {
      if (startRef.current) { startRef.current = null; setActive(false); }
      return;
    }
    const refSpace = r3f.gl.xr.getReferenceSpace();
    if (!refSpace) return;

    let xPressed = false;
    for (const src of session.inputSources) {
      if (src.handedness !== "left" || !src.gripSpace) continue;
      if (!(src.gamepad?.buttons[4]?.pressed)) continue;
      xPressed = true;

      const pose = xrFrame.getPose(src.gripSpace, refSpace);
      if (!pose) continue;

      const { x: px, y: py, z: pz } = pose.transform.position;

      if (!startRef.current) {
        // Button just pressed — compute spawn position and store for next frame
        lastEmitRef.current = state.renderRange[0];
        startRef.current = { ctrlY: py, startValue: state.renderRange[0] };
        const right   = new THREE.Vector3().setFromMatrixColumn(r3f.camera.matrixWorld, 0);
        const forward = new THREE.Vector3().setFromMatrixColumn(r3f.camera.matrixWorld, 2).negate();
        const cam     = r3f.camera.position;
        spawnRef.current = {
          pos:  new THREE.Vector3(px + right.x * 0.12 + forward.x * 0.28, py, pz + right.z * 0.12 + forward.z * 0.28),
          rotY: Math.atan2(cam.x - px, cam.z - pz),
        };
        setActive(true);
      }

      // Map vertical hand movement → renderRange[0]. 0.25 m of travel covers
      // the full loaded range. No group-dependent flip here: the on-screen
      // handle position (below) is driven directly by this raw index, so
      // keeping this a straight 1:1 mapping is what makes "hand moves down"
      // and "handle moves down" agree for every group — a flip here without
      // a matching flip in the handle's frac math (it previously had none)
      // just made the atmosphere group's slider visually run backwards.
      const [rMin, rMax] = state.loadedRange;
      const span = rMax - rMin;
      if (span > 0) {
        const yDelta = py - startRef.current.ctrlY;
        const newVal = Math.max(rMin, Math.min(rMax,
          startRef.current.startValue + yDelta * span / 0.25,
        ));
        if (Math.abs(newVal - lastEmitRef.current) >= 0.05) {
          lastEmitRef.current = newVal;
          state.setRenderRange([newVal, state.loadedRange[1]]);
        }
      }
      break; // only process the first left controller found
    }

    if (!xPressed && startRef.current) {
      startRef.current = null;
      setActive(false);
    }
  });

  if (!session || !active) return null;

  const [rMin, rMax] = state.loadedRange;
  const span    = Math.max(1, rMax - rMin);
  const val     = state.renderRange[0];
  const frac    = Math.max(0, Math.min(1, (val - rMin) / span));
  const H       = 0.16;
  const handleY = -H / 2 + frac * H;
  const valInt  = Math.round(val);
  const atEdge  = valInt === Math.round(rMin) || valInt === Math.round(rMax);
  // Rendered region is val → rMax (upper portion of track)
  const fillH   = (1 - frac) * H;
  const fillY   = frac * H / 2; // center of the upper fill segment

  return (
    <group ref={selfRef}>
      {/* Track */}
      <mesh>
        <planeGeometry args={[0.006, H]} />
        <meshBasicMaterial color="#1a3050" />
      </mesh>
      {/* Fill (current value → rMax) */}
      <mesh position={[0, fillY, 0.001]}>
        <planeGeometry args={[0.006, Math.max(0.001, fillH)]} />
        <meshBasicMaterial color="#00c4a8" />
      </mesh>
      {/* Handle */}
      <mesh position={[0, handleY, 0.002]}>
        <planeGeometry args={[0.022, 0.008]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      {/* Current value — integer, hidden when at rMin or rMax */}
      {!atEdge && (
        <Text position={[0.016, handleY, 0.003]}
          fontSize={0.008} color="#ffffffcc" anchorX="left" anchorY="middle">
          {valInt.toString()}
        </Text>
      )}
      {/* Title */}
      <Text position={[0, H / 2 + 0.018, 0.002]}
        fontSize={0.009} color="#ffffffaa" anchorX="center" anchorY="bottom">
        RENDER RANGE
      </Text>
      {/* Min / Max labels */}
      <Text position={[0.016, -H / 2, 0.002]}
        fontSize={0.008} color="#ffffff55" anchorX="left" anchorY="middle">
        {Math.round(rMin).toString()}
      </Text>
      <Text position={[0.016, H / 2, 0.002]}
        fontSize={0.008} color="#ffffff55" anchorX="left" anchorY="middle">
        {Math.round(rMax).toString()}
      </Text>
    </group>
  );
}

// ─── VRTimelineControl — hold A (right controller) to show a horizontal frame scrubber ──
//
// Pressing A on the right controller summons a horizontal slider above the
// controller. While held, moving the controller left/right scrubs through
// loaded frames. A short tap (no hold) toggles play/pause instead. Disappears
// on release. Hidden when fewer than 2 frames exist.

function VRTimelineControl({ state }: { state: GlobeDataState }) {
  const session     = useXR((s: any) => s.session as XRSession | null);
  const selfRef     = useRef<THREE.Group | null>(null);
  const [active, setActive] = useState(false);
  const startRef    = useRef<{ ctrlX: number; startFrame: number } | null>(null);
  const lastEmitRef = useRef(0);
  const spawnRef    = useRef<{ pos: THREE.Vector3; rotY: number } | null>(null);
  // Tap-vs-hold: record when A was first pressed; null means not pressed
  const pressTimeRef    = useRef<number | null>(null);
  // Whether the press has been committed to drag mode
  const isDragRef       = useRef(false);
  const HOLD_MS         = 280; // ms threshold between tap and hold
  // Playback tick: timestamp of last frame advance
  const lastFrameTimeRef = useRef<number>(0);
  const PLAY_INTERVAL_MS = 250; // ~4 fps

  useFrame((r3f, _, xrFrame) => {
    // Apply stored spawn transform as soon as the group mounts (one-shot after setActive)
    if (selfRef.current && spawnRef.current) {
      selfRef.current.position.copy(spawnRef.current.pos);
      selfRef.current.rotation.y = spawnRef.current.rotY;
      spawnRef.current = null;
    }

    // Playback tick — runs every frame when playing, independent of A-button / session state
    if (state.playing && state.frames.length >= 2 && !isDragRef.current) {
      const now = performance.now();
      if (now - lastFrameTimeRef.current >= PLAY_INTERVAL_MS) {
        lastFrameTimeRef.current = now;
        state.setActiveFrame((state.activeFrame + 1) % state.frames.length);
      }
    }

    const hasFrames = state.frames.length >= 2;
    if (!session || !xrFrame || !hasFrames) {
      if (startRef.current) { startRef.current = null; setActive(false); }
      pressTimeRef.current = null; isDragRef.current = false;
      return;
    }
    const refSpace = r3f.gl.xr.getReferenceSpace();
    if (!refSpace) return;

    let aPressed = false;
    for (const src of session.inputSources) {
      if (src.handedness !== "right" || !src.gripSpace) continue;
      if (!(src.gamepad?.buttons[4]?.pressed)) continue;
      aPressed = true;

      const pose = xrFrame.getPose(src.gripSpace, refSpace);
      if (!pose) continue;

      const { x: px, y: py, z: pz } = pose.transform.position;

      // First frame of this press
      if (pressTimeRef.current === null) {
        pressTimeRef.current = performance.now();
        lastEmitRef.current  = state.activeFrame;
        startRef.current     = { ctrlX: px, startFrame: state.activeFrame };
      }

      const heldMs = performance.now() - pressTimeRef.current;

      // Transition to drag mode once held long enough
      if (!isDragRef.current && heldMs >= HOLD_MS) {
        isDragRef.current = true;
        if (state.playing) state.setPlaying(false); // pause if animating
        const forward = new THREE.Vector3().setFromMatrixColumn(r3f.camera.matrixWorld, 2).negate();
        const cam     = r3f.camera.position;
        spawnRef.current = {
          pos:  new THREE.Vector3(px + forward.x * 0.20, py + 0.16, pz + forward.z * 0.20),
          rotY: Math.atan2(cam.x - px, cam.z - pz),
        };
        setActive(true);
      }

      // Drag scrubbing (only after hold threshold)
      if (isDragRef.current && startRef.current) {
        const total  = state.frames.length - 1;
        const xDelta = px - startRef.current.ctrlX;
        const newFrame = Math.round(
          Math.max(0, Math.min(total, startRef.current.startFrame + xDelta * total / 0.25))
        );
        if (newFrame !== lastEmitRef.current) {
          lastEmitRef.current = newFrame;
          state.setActiveFrame(newFrame);
        }
      }
      break;
    }

    if (!aPressed && pressTimeRef.current !== null) {
      const heldMs = performance.now() - pressTimeRef.current;
      pressTimeRef.current = null;

      if (!isDragRef.current && heldMs < HOLD_MS) {
        // Short tap → toggle play/pause
        state.setPlaying((p) => !p);
      }
      // Clean up drag state
      isDragRef.current = false;
      startRef.current  = null;
      setActive(false);
    }
  });

  if (!session || !active || state.frames.length < 2) return null;

  const total   = state.frames.length - 1;
  const idx     = Math.max(0, Math.min(total, state.activeFrame));
  const frac    = total > 0 ? idx / total : 0;
  const W       = 0.28;
  const handleX = -W / 2 + frac * W;
  const atEdge  = idx === 0 || idx === total;
  const dateStr = state.frames[idx]?.dateStr ?? "";

  return (
    <group ref={selfRef}>
      {/* Track */}
      <mesh>
        <planeGeometry args={[W, 0.006]} />
        <meshBasicMaterial color="#1a3050" />
      </mesh>
      {/* Fill (0 → current frame) */}
      <mesh position={[-W / 2 + (frac * W) / 2, 0, 0.001]}>
        <planeGeometry args={[Math.max(0.001, frac * W), 0.006]} />
        <meshBasicMaterial color="#00c4a8" />
      </mesh>
      {/* Handle */}
      <mesh position={[handleX, 0, 0.002]}>
        <planeGeometry args={[0.008, 0.022]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      {/* Current frame index — hidden at edges */}
      {!atEdge && (
        <Text position={[handleX, 0.016, 0.003]}
          fontSize={0.008} color="#ffffffcc" anchorX="center" anchorY="bottom">
          {idx.toString()}
        </Text>
      )}
      {/* Date label for current frame */}
      {dateStr !== "" && (
        <Text position={[handleX, -0.016, 0.003]}
          fontSize={0.008} color="#ffffffaa" anchorX="center" anchorY="top">
          {dateStr}
        </Text>
      )}
      {/* Min / Max frame labels */}
      <Text position={[-W / 2, 0.016, 0.002]}
        fontSize={0.008} color="#ffffff55" anchorX="center" anchorY="bottom">
        0
      </Text>
      <Text position={[W / 2, 0.016, 0.002]}
        fontSize={0.008} color="#ffffff55" anchorX="center" anchorY="bottom">
        {total.toString()}
      </Text>
      {/* Title */}
      <Text position={[0, -0.030, 0.002]}
        fontSize={0.009} color="#ffffffaa" anchorX="center" anchorY="top">
        TIMELINE
      </Text>
    </group>
  );
}

// ─── VRGlobeColorLegend — colormap legend to the left of the globe in VR ─────
//
// Sibling of globeGroupRef; tracks globe world POSITION via useFrame but
// ignores globe rotation so it stays upright and readable.

function VRGlobeColorLegend({
  state, globeGroupRef,
}: {
  state: GlobeDataState;
  globeGroupRef: React.RefObject<THREE.Group | null>;
}) {
  const session = useXR((s: any) => s.session as XRSession | null);
  const selfRef = useRef<THREE.Group | null>(null);
  const _tmp    = useMemo(() => new THREE.Vector3(), []);

  const def = datasetDef(state.loadedVariable);
  const stops = COLORMAP_CSS_STOPS[def.cmap];

  // Vertical canvas gradient: stops[0] = min (bottom), stops[N-1] = max (top).
  // createLinearGradient(0,0,0,SIZE) paired with CanvasTexture's default flipY=true
  // places stops[0] at mesh UV-bottom and stops[N-1] at mesh UV-top.
  const gradTex = useMemo(() => {
    if (typeof document === "undefined") return null;
    const SIZE = 256;
    const canvas = document.createElement("canvas");
    canvas.width = 1; canvas.height = SIZE;
    const ctx = canvas.getContext("2d")!;
    // Gradient from canvas bottom (y=SIZE) to canvas top (y=0).
    // flipY=true maps canvas-top→mesh-top (UV v=1), so this places
    // stops[0] (min) at mesh bottom and stops[N-1] (max) at mesh top.
    const grad = ctx.createLinearGradient(0, SIZE, 0, 0);
    stops.forEach((color, i) => grad.addColorStop(i / (stops.length - 1), color));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1, SIZE);
    return new THREE.CanvasTexture(canvas);
  }, [def.cmap]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => { gradTex?.dispose(); }, [gradTex]);

  useFrame(() => {
    if (selfRef.current && globeGroupRef.current) {
      globeGroupRef.current.getWorldPosition(_tmp);
      // Stay left of globe, same world Y and Z, ignoring globe rotation
      selfRef.current.position.set(
        _tmp.x - GLOBE_R_OUT * 0.35 - 0.16,
        _tmp.y,
        _tmp.z,
      );
    }
  });

  if (!session || !gradTex) return null;

  const [vmin, vmax] = state.activeDataRange;
  const N_TICKS = 5;
  const ticks = Array.from({ length: N_TICKS }, (_, i) => vmax - (vmax - vmin) * (i / (N_TICKS - 1)));
  const fmt = (v: number) => {
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
    return v % 1 === 0 ? `${v}` : v.toFixed(1);
  };

  const BAR_W = 0.016, BAR_H = 0.28;
  const TICK_X  = BAR_W / 2 + 0.005;  // start of horizontal tick mark
  const LABEL_X = BAR_W / 2 + 0.016;  // start of text label

  return (
    <group ref={selfRef}>
      {/* Variable label */}
      <Text position={[0, BAR_H / 2 + 0.016, 0.001]}
        fontSize={0.011} color="#ffffffaa" anchorX="center" anchorY="bottom">
        {def.label}
      </Text>

      {/* Gradient bar */}
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[BAR_W, BAR_H]} />
        <meshBasicMaterial map={gradTex} />
      </mesh>

      {/* Tick marks + value labels (top = vmax, bottom = vmin) */}
      {ticks.map((v, i) => {
        const ty = BAR_H / 2 - (i / (N_TICKS - 1)) * BAR_H;
        return (
          <group key={i} position={[0, ty, 0.001]}>
            <mesh position={[TICK_X, 0, 0]}>
              <planeGeometry args={[0.008, 0.001]} />
              <meshBasicMaterial color="#ffffff35" />
            </mesh>
            <Text position={[LABEL_X, 0, 0]}
              fontSize={0.010} color="#ffffffaa" anchorX="left" anchorY="middle">
              {fmt(v)}
            </Text>
          </group>
        );
      })}

      {/* Unit label */}
      <Text position={[0, -BAR_H / 2 - 0.014, 0.001]}
        fontSize={0.009} color="#ffffff55" anchorX="center" anchorY="top">
        {def.unit}
      </Text>
    </group>
  );
}

// ─── VRWindLegend — wind-speed color ramp, mirrored right of the globe in VR ─
//
// Same structure as VRGlobeColorLegend (vertical gradient bar + tick marks +
// title/unit labels, no background), just mirrored to the globe's right side
// and driven by the wind-speed ramp/range instead of the data colormap.

function VRWindLegend({
  state, globeGroupRef,
}: {
  state: GlobeDataState;
  globeGroupRef: React.RefObject<THREE.Group | null>;
}) {
  const session = useXR((s: any) => s.session as XRSession | null);
  const selfRef = useRef<THREE.Group | null>(null);
  const _tmp    = useMemo(() => new THREE.Vector3(), []);

  // Vertical canvas gradient: WIND_STOPS[0] (speed 0) = bottom, [N-1] (speedMax) = top.
  // Same createLinearGradient(0,SIZE,0,0) + flipY=true trick as VRGlobeColorLegend.
  const gradTex = useMemo(() => {
    if (typeof document === "undefined") return null;
    const SIZE = 256;
    const canvas = document.createElement("canvas");
    canvas.width = 1; canvas.height = SIZE;
    const ctx = canvas.getContext("2d")!;
    const grad = ctx.createLinearGradient(0, SIZE, 0, 0);
    WIND_STOPS.forEach(([t, c]) => grad.addColorStop(t, `#${c.getHexString()}`));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1, SIZE);
    return new THREE.CanvasTexture(canvas);
  }, []);

  useEffect(() => () => { gradTex?.dispose(); }, [gradTex]);

  useFrame(() => {
    if (selfRef.current && globeGroupRef.current) {
      globeGroupRef.current.getWorldPosition(_tmp);
      // Mirror of VRGlobeColorLegend's left offset, same world Y/Z.
      selfRef.current.position.set(
        _tmp.x + GLOBE_R_OUT * 0.35 + 0.16,
        _tmp.y,
        _tmp.z,
      );
    }
  });

  if (!session || !gradTex || !state.windOverlay || state.group !== "dyamond_ocean" || !state.windField) return null;

  const smax = state.windField.speedMax;
  const N_TICKS = 5;
  const ticks = Array.from({ length: N_TICKS }, (_, i) => smax * (1 - i / (N_TICKS - 1)));
  const fmt = (v: number) => (v % 1 === 0 ? `${v}` : v.toFixed(1));

  const BAR_W = 0.016, BAR_H = 0.28;
  const TICK_X  = BAR_W / 2 + 0.005;
  const LABEL_X = BAR_W / 2 + 0.016;

  return (
    <group ref={selfRef}>
      {/* Title */}
      <Text position={[0, BAR_H / 2 + 0.016, 0.001]}
        fontSize={0.011} color="#ffffffaa" anchorX="center" anchorY="bottom">
        WIND SPEED
      </Text>

      {/* Gradient bar */}
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[BAR_W, BAR_H]} />
        <meshBasicMaterial map={gradTex} />
      </mesh>

      {/* Tick marks + value labels (top = smax, bottom = 0) */}
      {ticks.map((v, i) => {
        const ty = BAR_H / 2 - (i / (N_TICKS - 1)) * BAR_H;
        return (
          <group key={i} position={[0, ty, 0.001]}>
            <mesh position={[TICK_X, 0, 0]}>
              <planeGeometry args={[0.008, 0.001]} />
              <meshBasicMaterial color="#ffffff35" />
            </mesh>
            <Text position={[LABEL_X, 0, 0]}
              fontSize={0.010} color="#ffffffaa" anchorX="left" anchorY="middle">
              {fmt(v)}
            </Text>
          </group>
        );
      })}

      {/* Unit label */}
      <Text position={[0, -BAR_H / 2 - 0.014, 0.001]}
        fontSize={0.009} color="#ffffff55" anchorX="center" anchorY="top">
        m/s
      </Text>
    </group>
  );
}

// ─── GlobeColorLegend — vertical gradient bar with labeled ticks ─────────────

function GlobeColorLegend({ state }: { state: GlobeDataState }) {
  const def = datasetDef(state.loadedVariable);
  const stops = COLORMAP_CSS_STOPS[def.cmap];
  const [vmin, vmax] = state.activeDataRange;
  const unit = def.unit;

  const BAR_H = 420;
  const N = 5;
  const ticks = Array.from({ length: N }, (_, i) => vmax - (vmax - vmin) * (i / (N - 1)));

  function fmt(v: number) {
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
    if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
    return v % 1 === 0 ? `${v}` : v.toFixed(1);
  }

  const gradient = `linear-gradient(to top, ${stops.join(",")})`;

  return (
    <div className="absolute left-5 top-1/2 -translate-y-1/2 z-10 flex gap-2 items-start select-none pointer-events-none">
      {/* gradient bar */}
      <div
        className="w-[14px] flex-shrink-0 rounded-sm"
        style={{
          height: BAR_H,
          background: gradient,
          boxShadow: "0 0 0 1px rgba(255,255,255,0.10)",
        }}
      />
      {/* tick labels */}
      <div className="relative flex-shrink-0" style={{ height: BAR_H + 28 }}>
        {ticks.map((v, i) => (
          <div
            key={i}
            className="absolute flex items-center gap-1.5"
            style={{ top: Math.round((i / (N - 1)) * BAR_H) - 7 }}
          >
            <div className="w-2 h-px bg-white/30" />
            <span className="font-mono text-[12px] text-white/70 leading-none tabular-nums">
              {fmt(v)}
            </span>
          </div>
        ))}
        <div className="absolute flex items-center" style={{ top: BAR_H + 10 }}>
          <span className="font-mono text-[12px] font-semibold text-white/45">{unit}</span>
        </div>
      </div>
    </div>
  );
}

// ─── GlobeDepthBadge — bold, boxed current-depth/altitude readout, top-left ──
// Positioned to the right of ProgressLogPanel (master-dashboard.tsx, which sits
// at top-[5.25rem] left-4 w-[360px] in the same positioned ancestor) so it never
// overlaps that panel regardless of whether the log is collapsed or expanded.

function GlobeDepthBadge({ state }: { state: GlobeDataState }) {
  if (!state.activeTexture) return null;
  return (
    <div className="absolute top-[5.25rem] left-[392px] z-10 max-w-[220px] pointer-events-none select-none">
      <div className="rounded-lg border border-[#00c4a8]/50 bg-black/70 backdrop-blur-md px-3 py-2 shadow-[0_0_0_1px_rgba(255,255,255,0.05)]">
        <p className="font-mono text-[9px] uppercase tracking-widest text-[#00c4a8]/70 mb-0.5">
          {state.depthLevels === 1 ? "Level" : state.group === "dyamond_atm" ? "Altitude" : "Depth"}
        </p>
        {/* Single value: the moving render-range cutoff (renderRange[0]) is
            "the current depth/altitude" the user is scrubbing through. */}
        <p className="font-mono text-[14px] font-bold text-white">
          {state.depthLevels === 1
            ? "Surface"
            : formatLevelPhysicalOnly(state.group, state.renderRange[0], state.depthLevels)}
        </p>
      </div>
    </div>
  );
}

// ─── GlobeWindLegend — small wind-speed ramp shown when the overlay is on ────

function GlobeWindLegend({ state }: { state: GlobeDataState }) {
  if (!state.windOverlay || state.group !== "dyamond_ocean" || !state.windField) return null;
  const smax = state.windField.speedMax;
  const gradient = `linear-gradient(to right, ${WIND_STOPS.map(
    ([t, c]) => `#${c.getHexString()} ${(t * 100).toFixed(0)}%`,
  ).join(", ")})`;
  return (
    <div className="absolute top-[8.5rem] left-[392px] z-10 max-w-[220px] pointer-events-none select-none">
      <div className="rounded-lg border border-[#00c4a8]/50 bg-black/70 backdrop-blur-md px-3 py-2 shadow-[0_0_0_1px_rgba(255,255,255,0.05)]">
        <p className="font-mono text-[9px] uppercase tracking-widest text-[#00c4a8]/70 mb-1">
          Wind speed
        </p>
        <div className="h-[10px] w-[180px] rounded-sm" style={{ background: gradient, boxShadow: "0 0 0 1px rgba(255,255,255,0.10)" }} />
        <div className="mt-0.5 flex justify-between font-mono text-[10px] text-white/60 tabular-nums">
          <span>0</span>
          <span>{smax.toFixed(0)} m/s</span>
        </div>
      </div>
    </div>
  );
}

// ─── GlobeTimelineBar — bottom scrubber shown when multi-frame is loaded ─────

export function GlobeTimelineBar({ state }: { state: GlobeDataState }) {
  if (state.frames.length === 0 && !state.fetchingFrames) return null;

  const pct = state.frames.length > 1
    ? (state.activeFrame / (state.frames.length - 1)) * 100
    : 0;

  return (
    <div className="absolute bottom-5 left-5 z-10 flex flex-col gap-2 pointer-events-auto select-none"
         style={{ width: "calc(100% - 320px)", minWidth: 280 }}>
      {/* date label */}
      <div className="flex items-center justify-between">
        <span className="font-mono text-[12px] text-white/70 tabular-nums">
          {state.frames[state.activeFrame]?.dateStr ?? ""}
        </span>
        {state.frames.length > 0 && (
          <span className="font-mono text-[10px] text-white/30 tabular-nums">
            {state.activeFrame + 1} / {state.frames.length}
          </span>
        )}
      </div>

      {/* scrubber */}
      {state.frames.length > 1 ? (
        <input
          type="range"
          min={0}
          max={state.frames.length - 1}
          value={state.activeFrame}
          onChange={(e) => state.setActiveFrame(Number(e.target.value))}
          className="w-full accent-[#00c4a8]"
        />
      ) : (
        <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
          <div className="h-full bg-[#00c4a8]/50 rounded-full animate-pulse" style={{ width: `${pct || 40}%` }} />
        </div>
      )}

      {/* frame tick labels */}
      {state.frames.length > 1 && (
        <div className="flex justify-between">
          {state.frames.map((f, i) => (
            <button key={i} type="button" onClick={() => state.setActiveFrame(i)}
              className={cn(
                "font-mono text-[9px] px-1 rounded transition-colors leading-none",
                i === state.activeFrame ? "text-[#00c4a8]" : "text-white/25 hover:text-white/50",
              )}>
              {f.dateStr.slice(3, 10)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── OceanGlobeViewer — canvas only; controls live in the dashboard panel ─────

export function OceanGlobeViewer({
  state,
  vrButtonRight = 12,
  vrButtonBottom = 12,
}: {
  state: GlobeDataState;
  vrButtonRight?: number;
  vrButtonBottom?: number;
}) {
  const xrSupported = useXRSupport();
  const [xrActive, setXrActive] = useState(false);
  useEffect(() => oceanXRStore.subscribe((s) => setXrActive(!!s.session)), []);

  return (
    <div className="relative w-full h-full" style={{ background: "#06080c" }}>
      <Canvas
        gl={{ antialias: true }}
        camera={{ position: [0, 0, 2.8], fov: 40 }}
        style={{ width: "100%", height: "100%" }}
      >
        <XR store={oceanXRStore}>
          <color attach="background" args={["#06080c"]} />
          <GlobeScene globeState={state} />
        </XR>
      </Canvas>

      <VRButton
        supported={xrSupported}
        active={xrActive}
        onEnter={() => { void oceanXRStore.enterVR(); }}
        onExit={() => oceanXRStore.getState().session?.end()}
        className="absolute z-10"
        style={{ right: vrButtonRight, bottom: vrButtonBottom }}
      />

      {/* offline overlay */}
      {state.error && !state.loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="rounded-xl border border-red-900/40 bg-black/80 px-8 py-5 text-center backdrop-blur-md">
            <p className="font-mono text-[13px] font-bold text-red-300 mb-1">Ocean backend offline</p>
            <p className="font-mono text-[11px] text-white/35 mt-1">
              Start the backend:{" "}
              <span className="text-white/55">uvicorn main:app --port 8000</span>
            </p>
          </div>
        </div>
      )}

      {/* color legend — only show once data is loaded */}
      {state.activeTexture && <GlobeColorLegend state={state} />}
      <GlobeDepthBadge state={state} />
      <GlobeWindLegend state={state} />

      {/* timeline scrubber — shown when multi-frame is loaded or fetching */}
      <GlobeTimelineBar state={state} />
    </div>
  );
}
