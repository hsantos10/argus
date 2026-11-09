// Shared 90-level vertical grid for the LLC4320 / LLC2160 ocean datasets.
// Index i (0..89) → Level (i+1) → depth in meters at the cell center.
// Anchors from contest data: L1=0.5, L10=17.7, L20=78, L50=900, L90=6760 (max).
// Intermediate values are PCHIP-interpolated for monotonic, smoothly thickening cells.
export const DEPTH_LEVELS_METERS: number[] = [
  0.50, 0.81, 1.68, 3.04, 4.83, 6.98, 9.41, 12.05, 14.84, 17.70,
  20.91, 24.79, 29.33, 34.49, 40.28, 46.67, 53.66, 61.22, 69.33, 78.00,
  87.59, 98.47, 110.65, 124.12, 138.86, 154.87, 172.15, 190.67, 210.45, 231.46,
  253.71, 277.18, 301.86, 327.75, 354.85, 383.13, 412.60, 443.25, 475.06, 508.04,
  542.17, 577.45, 613.86, 651.41, 690.08, 729.86, 770.76, 812.75, 855.83, 900.00,
  948.07, 1002.80, 1064.05, 1131.69, 1205.61, 1285.67, 1371.74, 1463.71, 1561.44, 1664.81,
  1773.69, 1887.96, 2007.48, 2132.14, 2261.80, 2396.34, 2535.63, 2679.55, 2827.96, 2980.75,
  3137.79, 3298.94, 3464.09, 3633.11, 3805.86, 3982.23, 4162.09, 4345.31, 4531.76, 4721.32,
  4913.86, 5109.25, 5307.38, 5508.10, 5711.30, 5916.85, 6124.62, 6334.49, 6546.32, 6760.00,
];

export function formatDepth(levelIdx: number): string {
  const i = Math.max(0, Math.min(DEPTH_LEVELS_METERS.length - 1, Math.round(levelIdx)));
  const m = DEPTH_LEVELS_METERS[i];
  if (m < 1000) return `${m.toFixed(1)} m (Level ${i})`;
  return `${(m / 1000).toFixed(2)} km (Level ${i})`;
}

export function formatDepthShort(levelIdx: number): string {
  const i = Math.max(0, Math.min(DEPTH_LEVELS_METERS.length - 1, Math.round(levelIdx)));
  const m = DEPTH_LEVELS_METERS[i];
  if (m < 1000) return `Level ${i} · ${m.toFixed(1)} m`;
  return `Level ${i} · ${(m / 1000).toFixed(2)} km`;
}

// ─── GEOS atmosphere levels — APPROXIMATE altitude, for display only ─────────
// GEOS-5's true hybrid sigma-pressure level definition isn't present in the
// IDX metadata we stream from, so this is not the real model grid: it's a
// smooth top-of-atmosphere (index 0, ~0.01 hPa) → surface (index N-1,
// ~1013 hPa) profile, converted to altitude via the US Standard Atmosphere
// barometric formula. Good enough to orient a viewer ("this is upper
// stratosphere" vs "this is near the surface"); not a substitute for the
// model's actual level pressures.
const P_TOP_HPA = 0.01;
const P_SURF_HPA = 1013.25;

function approxPressureHpa(levelIdx: number, nLevels: number): number {
  const sigma = Math.max(0, Math.min(1, levelIdx / (nLevels - 1)));
  return P_TOP_HPA * Math.pow(P_SURF_HPA / P_TOP_HPA, Math.pow(sigma, 0.6));
}

function pressureToAltitudeM(pHpa: number): number {
  // Troposphere (0–11 km): ICAO barometric formula.
  if (pHpa >= 226.32) return 44330 * (1 - Math.pow(pHpa / P_SURF_HPA, 1 / 5.255));
  // Lower stratosphere (11–20 km): isothermal layer, T=216.65K.
  if (pHpa >= 54.75) return 11000 - 6341.62 * Math.log(pHpa / 226.32);
  // Upper stratosphere and above: extend the isothermal formula as a rough
  // approximation — errors grow here, but these levels are the dim,
  // low-detail top-of-atmosphere layers anyway.
  return 20000 - 6341.62 * Math.log(pHpa / 54.75);
}

export function approxAtmAltitudeKm(levelIdx: number, nLevels: number): number {
  return pressureToAltitudeM(approxPressureHpa(levelIdx, nLevels)) / 1000;
}

export function formatAtmLevel(levelIdx: number, nLevels: number): string {
  const i = Math.max(0, Math.min(nLevels - 1, Math.round(levelIdx)));
  const km = approxAtmAltitudeKm(i, nLevels);
  return `Level ${i} · ~${km.toFixed(1)} km alt.`;
}

// ─── Group-aware helpers ─────────────────────────────────────────────────────

// `group` is intentionally a plain string: atmosphere levels format as
// altitude, single-layer surface groups (CMIP6) as "surface", and anything
// else as ocean depth.
export function formatLevelLabel(
  group: string,
  levelIdx: number,
  nLevels: number,
): string {
  return group === "dyamond_atm" ? formatAtmLevel(levelIdx, nLevels) : formatDepthShort(levelIdx);
}

// Physical value only (no "Level N ·" prefix) — for compact inline use.
export function formatLevelPhysicalOnly(
  group: string,
  levelIdx: number,
  nLevels: number,
): string {
  if (group === "dyamond_atm") {
    const i = Math.max(0, Math.min(nLevels - 1, Math.round(levelIdx)));
    return `~${approxAtmAltitudeKm(i, nLevels).toFixed(1)} km alt.`;
  }
  if (nLevels <= 1) return "surface";
  const i = Math.max(0, Math.min(DEPTH_LEVELS_METERS.length - 1, Math.round(levelIdx)));
  const m = DEPTH_LEVELS_METERS[i];
  return m < 1000 ? `${m.toFixed(1)} m` : `${(m / 1000).toFixed(2)} km`;
}
