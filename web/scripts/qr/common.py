"""qr.common — shared constants, helpers, and matplotlib render functions.

Imported via `from qr.common import *` by every dataset query module.
"""
import io
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np

# ensure the repo root is importable so analysis.core.* resolves, and this
# scripts/ dir so `from qr import ...` resolves when spawned by the Node route.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    '_DYAMOND_EPOCH',
    '_ECCO_EPOCH',
    '_CMIP6_EPOCH',
    '_dyamond_date',
    '_ecco_date',
    '_cmip6_date',
    '_DYAMOND_CMAP',
    '_DYAMOND_SYMMETRIC',
    '_ECCO_CMAP',
    '_CMIP6_CMAP',
    '_OCN_CMAP',
    '_progress',
    '_load_dataset_with_retry',
    'DEPTH_LEVELS_METERS',
    '_emit_event',
    '_png_data_url',
    '_parse_params',
    '_subsample',
    '_cap_uniform',
    '_DMD_SNAPSHOT_CAP',
    '_render_equirectangular',
    '_midpoint_timestep',
    '_render_grid',
    '_render_face_square',
    '_dyamond_read',
    '_FACE_REGIONS',
    '_equirect_from_cubed_faces',
    '_render_layer_png',
    '_ocn_read',
    '_emit_frame',
    '_period_human',
    '_render_eigenvalue_plot',
    '_dmd_behavior',
    '_BEHAVIOR_COLOR',
    '_render_dmd_spectrum',
    '_render_triptych',
    '_render_all_modes_frame',
    '_render_anim_frame',
    '_render_uv_panel',
    '_pad_to_square',
    '_render_wind_panel',
    '_render_dmd_overview',
    '_MAX_FETCH_MB',
    '_QUALITY_TIP',
    '_probe_and_plan',
]

_DYAMOND_EPOCH = datetime(2020, 1, 20, tzinfo=timezone.utc)
_ECCO_EPOCH    = datetime(2011, 9, 13, tzinfo=timezone.utc)
_CMIP6_EPOCH   = datetime(1,    1,  1, tzinfo=timezone.utc)

def _dyamond_date(idx): return (_DYAMOND_EPOCH + timedelta(minutes=30 * int(idx))).strftime("%Y-%b-%d %H:%MZ")
def _ecco_date(idx):    return (_ECCO_EPOCH    + timedelta(hours=int(idx))).strftime("%Y-%b-%d %H:%MZ")
def _cmip6_date(idx):   return (_CMIP6_EPOCH   + timedelta(days=int(idx))).strftime("%Y-%b-%d")

# ── Colourmap tables (used across all render paths) ──────────────────────────

_DYAMOND_CMAP = {
    "u": "RdBu_r", "v": "RdBu_r", "w": "RdBu_r",
    "dthdt": "RdBu_r", "dthdtcn": "RdBu_r",
    "t": "coolwarm",
    "fcld": "Blues", "qi": "Blues", "ql": "Blues", "ri": "Blues", "rl": "Blues",
    "p": "viridis", "h": "viridis", "delp": "viridis",
    "co": "YlOrRd", "co2": "YlOrRd",
}
_DYAMOND_SYMMETRIC = {"u", "v", "w", "dthdt", "dthdtcn"}

_ECCO_CMAP = {
    "theta": "magma",
    "salt":  "cividis",
    "w":     "RdBu_r",
}

_CMIP6_CMAP = {
    "tas":     "inferno",
    "tasmin":  "inferno",
    "tasmax":  "inferno",
    "pr":      "Blues",
    "rsds":    "YlOrRd",
    "rlds":    "YlOrRd",
    "sfcWind": "viridis",
    "hurs":    "YlGnBu",
}

_OCN_CMAP = {
    "theta": "magma",
    "salt":  "viridis",
    "u":     "RdBu_r",
    "v":     "RdBu_r",
    "w":     "RdBu_r",
}


def _progress(msg):
    try:
        sys.stderr.write(f"PROGRESS {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _load_dataset_with_retry(load_fn, *args, label="dataset", retries=3, delay=5.0):
    """Retry wrapper for flaky remote dataset loads (Pelican / OpenVisus network errors)."""
    import time
    last_exc = None
    for attempt in range(retries):
        try:
            return load_fn(*args)
        except (SystemError, OSError, RuntimeError, Exception) as exc:
            last_exc = exc
            if attempt < retries - 1:
                _progress(f"  {label} load failed (attempt {attempt + 1}/{retries}): {exc!r} — retrying in {delay:.0f}s")
                time.sleep(delay)
            else:
                _progress(f"  {label} load failed after {retries} attempts: {exc!r}")
    raise last_exc


# LLC 90-level depth grid (meters at cell centers). Anchors from contest data:
# L1=0.5, L10=17.7, L20=78, L50=900, L90=6760 (max). PCHIP-interpolated.
# Keep in sync with sciviz-web/src/lib/depth-levels.ts.
DEPTH_LEVELS_METERS = [
    0.50, 0.81, 1.68, 3.04, 4.83, 6.98, 9.41, 12.05, 14.84, 17.70,
    20.91, 24.79, 29.33, 34.49, 40.28, 46.67, 53.66, 61.22, 69.33, 78.00,
    87.59, 98.47, 110.65, 124.12, 138.86, 154.87, 172.15, 190.67, 210.45, 231.46,
    253.71, 277.18, 301.86, 327.75, 354.85, 383.13, 412.60, 443.25, 475.06, 508.04,
    542.17, 577.45, 613.86, 651.41, 690.08, 729.86, 770.76, 812.75, 855.83, 900.00,
    948.07, 1002.80, 1064.05, 1131.69, 1205.61, 1285.67, 1371.74, 1463.71, 1561.44, 1664.81,
    1773.69, 1887.96, 2007.48, 2132.14, 2261.80, 2396.34, 2535.63, 2679.55, 2827.96, 2980.75,
    3137.79, 3298.94, 3464.09, 3633.11, 3805.86, 3982.23, 4162.09, 4345.31, 4531.76, 4721.32,
    4913.86, 5109.25, 5307.38, 5508.10, 5711.30, 5916.85, 6124.62, 6334.49, 6546.32, 6760.00,
]


def _emit_event(obj):
    """Emit a single EVENT line on stderr for the Node streaming layer."""
    try:
        sys.stderr.write("EVENT " + json.dumps(obj) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _png_data_url(png_bytes):
    import base64
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _parse_params():
    raw = sys.stdin.read()
    return json.loads(raw)


def _subsample(all_ts, t_start, t_end, cap=8):
    sub = all_ts[int(t_start):int(t_end) + 1]
    if len(sub) == 0:
        sub = all_ts[: max(1, min(len(all_ts), 8))]
    if len(sub) > cap:
        step = len(sub) // cap
        sub = [sub[i * step] for i in range(cap)]
    return list(sub)


def _cap_uniform(seq, cap):
    """Uniformly thin an already-selected list of timesteps down to `cap`.

    Unlike _subsample (which slices all_ts positionally), this operates on a
    pre-filtered list. DMD/Tucker are unaffected by uniform temporal
    subsampling, so capping a 1000-step window to ~120 snapshots keeps the
    dynamics while avoiding multi-GB fetches.
    """
    seq = list(seq)
    if cap <= 0 or len(seq) <= cap:
        return seq
    step = len(seq) / cap
    return [seq[int(i * step)] for i in range(cap)]


# Max snapshots fetched for a single DMD/Tucker run. A long calendar window
# (e.g. 42 ECCO days ≈ 1000 hourly steps) is uniformly thinned to this many
# snapshots — enough to resolve the dynamics without multi-GB fetches.
_DMD_SNAPSHOT_CAP = 120
def _render_equirectangular(arr, cmap="viridis", vmin=None, vmax=None, llc=False):
    """Render a single 2D array as an equirectangular PNG (2:1, no chrome).

    Pass vmin/vmax to lock the colorscale across an animation sequence; if
    omitted they are computed from the 2nd/98th percentile of finite values.

    llc=True: mask the top 17% of rows before rendering (removes the
    Lat-Lon-Cap Arctic cap face, which is in a rotated coordinate system and
    creates spoke seam artifacts when displayed as equirectangular).
    """
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(arr.ndim - 2)))
    # Replace LLC Arctic cap: top ~17% of y rows (1/6 of the LLC4320 grid)
    # are the rotated cap face.  They don't correspond to regular lat/lon and
    # produce spoke seam artifacts above ~60°N.  Fill with the last valid row
    # (boundary extrapolation) so the polar region shows a smooth uniform
    # temperature rather than a black void.
    if llc and arr.ndim == 2:
        cap = max(1, int(arr.shape[0] * 0.17))
        arr = arr.copy()
        boundary = arr[-(cap + 1), :]
        arr[-cap:, :] = boundary
    # Compute percentile clip on FINITE values first — before NaN replacement.
    # If we fill NaN→0 first, ocean/missing pixels (often 69%+ of CMIP6) drive
    # vmin to 0 and the whole colorscale collapses.
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        if vmin is None: vmin = float(np.percentile(finite, 2))
        if vmax is None: vmax = float(np.percentile(finite, 98))
        if vmax <= vmin: vmax = vmin + 1e-6
        arr = np.where(np.isnan(arr), vmin, arr)   # fill NaN with min → blends in
    else:
        if vmin is None: vmin = 0.0
        if vmax is None: vmax = 1.0
        arr = np.zeros_like(arr)

    # Target 2:1 aspect, 2048x1024.
    out_w, out_h = 2048, 1024
    dpi = 128
    fig = plt.figure(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # fill entire canvas
    ax.imshow(
        arr,
        cmap=cmap,
        origin="lower",
        extent=[-180.0, 180.0, -90.0, 90.0],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_axis_off()
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    _progress("Encoding PNG")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _midpoint_timestep(all_ts, t_start, t_end):
    sub = all_ts[int(t_start):int(t_end) + 1]
    if not sub:
        sub = all_ts[: max(1, min(len(all_ts), 1))]
    return sub[len(sub) // 2]


def _render_grid(frames, titles, suptitle, cmap="viridis"):
    n = len(frames)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    _progress(f"Rendering {nrows}x{ncols} grid ({n} panels)")
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), squeeze=False)
    for i, (frame, title) in enumerate(zip(frames, titles)):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        arr = np.squeeze(np.asarray(frame, dtype=np.float32))
        if arr.ndim > 2:
            arr = arr.mean(axis=tuple(range(arr.ndim - 2)))
        finite_f = arr[np.isfinite(arr)]
        if finite_f.size > 0:
            vmin_f = float(np.percentile(finite_f, 2))
            vmax_f = float(np.percentile(finite_f, 98))
            if vmax_f <= vmin_f: vmax_f = vmin_f + 1e-6
            arr = np.where(np.isnan(arr), vmin_f, arr)
        else:
            vmin_f, vmax_f = 0.0, 1.0
            arr = np.zeros_like(arr)
        im = ax.imshow(arr, cmap=cmap, origin="lower", aspect="auto",
                       vmin=vmin_f, vmax=vmax_f)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    _progress("Encoding PNG")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_face_square(arr, cmap="RdBu_r", face_idx=None):
    """Render a DYAMOND cubed-sphere face as a square PNG texture.

    The raw array has pixels at (py, px) where py=0 is the southern/lower edge
    (gnomonic t=−1) and py=H-1 is the northern/upper edge (t=+1).

    origin="upper" puts arr[0,:] at the top of the PNG.  Three.js TextureLoader's
    default flipY=true maps PNG-top to UV v=0 (face bottom), so south lands at the
    face bottom — geographically correct for equatorial faces 0–3.

    Polar faces need additional rotation so their gnomonic s/t axes align with the
    u/v axes expected by Three.js BoxGeometry slot 2 (+Y, face 4) and slot 3 (−Y, face 5):
      face 4 north: np.flipud(np.rot90(arr, k=1))
      face 5 south: arr.T
    """
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(arr.ndim - 2)))
    arr = np.nan_to_num(arr, nan=0.0)

    if face_idx == 4:
        arr = np.flipud(np.rot90(arr, k=1))
    elif face_idx == 5:
        arr = arr.T

    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    out = 1024
    dpi = 128
    fig = plt.figure(figsize=(out / dpi, out / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(arr, cmap=cmap, origin="upper", aspect="equal", vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _dyamond_read(db, time, z_level, quality):
    """
    Read one z-slice from a DYAMOND GEOS dataset with automatic quality fallback.

    OpenVisus raises "begin query failed cannot find a good end_resolution to start with"
    when the requested quality maps to a resolution level that doesn't exist in the IDX
    pyramid (e.g. quality too negative for a small dataset).  We step quality toward 0
    until a read succeeds.

    DYAMOND GEOS faces have depth_52 (52 z-levels, indices 0-51).  We cap z_level here
    to keep the z-range inside the dataset bounds.
    """
    DYAMOND_MAX_Z = 51  # depth_52 in URL → levels 0..51
    z_safe = min(int(z_level), DYAMOND_MAX_Z)
    z_range = [z_safe, min(z_safe + 1, DYAMOND_MAX_Z + 1)]

    q = int(quality)
    last_exc = None
    for attempt in range(6):          # try q, q+1, q+2, … up to quality=0
        try:
            _progress(f"  db.read time={time} z={z_range} quality={q} (attempt {attempt+1})")
            arr = np.array(db.read(time=int(time), z=z_range, quality=q), dtype=np.float32)
            if attempt > 0:
                _progress(f"  succeeded at quality={q}")
            return arr
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "end_resolution" in msg or "begin query failed" in msg:
                q = min(q + 1, 0)
                _progress(f"  quality retry → {q}")
            else:
                raise  # non-resolution error: propagate immediately
    raise last_exc
_FACE_REGIONS = [
    "~Atlantic/Africa",
    "~Indian/Asia",
    "~Pacific",
    "~Americas",
    "Arctic (N Pole)",
    "Antarctic (S Pole)",
]
def _equirect_from_cubed_faces(face_arrays, out_w=720, out_h=360):
    """
    Reproject 6 cubed-sphere face arrays into one equirectangular field.

    `face_arrays` is a list of 6 2D float arrays (any matching square shape).
    Returns an (out_h, out_w) float32 array.

    Face layout verified against GEOS c1440 data on 2026-07-04 (seam-energy
    minimization over all dihedral orientations + visual coastline check):
        data face 0: equatorial, gnomonic center lon 0°
        data face 1: equatorial, center lon 90°E
        data face 3: equatorial, center lon 180°
        data face 4: equatorial, center lon 90°W
        data face 2: NORTH pole (rotated 180°)
        data face 5: SOUTH pole (fliplr + rot90)

    For each (lat, lon) target, compute the unit vector v on the sphere,
    pick the dominant axis (= the cube face), then map the remaining two
    components into face-local (s, t) coordinates in [-1, 1] and bilinearly
    sample the corresponding (re-oriented) face.
    """
    # cube position order: E0 (lon 0°), E1 (90°E), E2 (180°), E3 (90°W), N, S
    CUBE_POS_TO_FACE = (0, 1, 3, 4, 2, 5)
    # dihedral op per cube position: fliplr if op>=4, then rot90(k=op%4)
    CUBE_POS_OP = (0, 0, 1, 1, 2, 5)
    # Faces are logically square (1440²); quality can downsample the two axes
    # unevenly, so resample everything to S×S before re-orienting.
    S = 0
    for arr in face_arrays:
        if arr is None:
            continue
        S = max(S, arr.shape[-2], arr.shape[-1])
    if S == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)

    # Stack in CUBE POSITION order, re-oriented so row = t and col = s.
    stack = np.zeros((6, S, S), dtype=np.float32)
    for p in range(6):
        arr = face_arrays[CUBE_POS_TO_FACE[p]]
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim > 2:
            a = np.squeeze(a)
        if a.ndim > 2:
            a = a.mean(axis=tuple(range(a.ndim - 2)))
        if a.shape != (S, S):
            # Resize via simple block-resampling (nearest) — preserves a usable
            # texture even if loaded face shapes are mismatched.
            yi = (np.linspace(0, a.shape[0] - 1, S)).astype(np.int32)
            xi = (np.linspace(0, a.shape[1] - 1, S)).astype(np.int32)
            a = a[yi[:, None], xi[None, :]]
        op = CUBE_POS_OP[p]
        if op >= 4:
            a = np.fliplr(a)
        a = np.rot90(a, k=op % 4)
        stack[p] = np.nan_to_num(a, nan=0.0)

    # Build target (lat, lon) grid.
    lat = np.linspace(-90.0, 90.0, out_h, dtype=np.float32)
    lon = np.linspace(-180.0, 180.0, out_w, dtype=np.float32)
    lat_r = np.deg2rad(lat)[:, None]              # (out_h, 1)
    lon_r = np.deg2rad(lon)[None, :]              # (1, out_w)
    cos_lat = np.cos(lat_r)
    x = cos_lat * np.cos(lon_r)
    y = cos_lat * np.sin(lon_r)
    z = np.sin(lat_r) * np.ones_like(lon_r)

    # Cube position per pixel: 0..3 equatorial (centers lon 0/90E/180/90W),
    # 4 = north pole, 5 = south pole.
    ax_, ay_, az_ = np.abs(x), np.abs(y), np.abs(z)
    face_idx = np.where(
        (az_ >= ax_) & (az_ >= ay_), np.where(z > 0, 4, 5),
        np.where(ax_ >= ay_, np.where(x > 0, 0, 2), np.where(y > 0, 1, 3)),
    ).astype(np.int32)

    # (s, t) gnomonic face-local coordinates in [-1, 1]: s along local east
    # (equatorial) or +X (polar), t along north (equatorial) or +Y (polar).
    normals = {0: (1, 0, 0), 1: (0, 1, 0), 2: (-1, 0, 0), 3: (0, -1, 0),
               4: (0, 0, 1), 5: (0, 0, -1)}
    east = {0: (0, 1, 0), 1: (-1, 0, 0), 2: (0, -1, 0), 3: (1, 0, 0)}
    s = np.zeros_like(x)
    t = np.zeros_like(x)
    for p in range(6):
        m = face_idx == p
        n = normals[p]
        d = np.maximum(x * n[0] + y * n[1] + z * n[2], 1e-9)
        if p < 4:
            e = east[p]
            sp = (x * e[0] + y * e[1]) / d
            tp = z / d
        else:
            sp = x / d
            tp = y / d
        s = np.where(m, sp, s)
        t = np.where(m, tp, t)

    # Map (s, t) ∈ [-1, 1] → pixel (px, py) ∈ [0, S-1]².
    px = (s + 1.0) * 0.5 * (S - 1)
    py = (t + 1.0) * 0.5 * (S - 1)

    # Bilinear sample on the per-pixel face.
    px = np.clip(px, 0.0, S - 1.0)
    py = np.clip(py, 0.0, S - 1.0)
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, S - 1)
    y1 = np.clip(y0 + 1, 0, S - 1)
    fx = (px - x0).astype(np.float32)
    fy = (py - y0).astype(np.float32)

    # Fancy index: stack[face_idx, y, x]
    a00 = stack[face_idx, y0, x0]
    a10 = stack[face_idx, y0, x1]
    a01 = stack[face_idx, y1, x0]
    a11 = stack[face_idx, y1, x1]
    out = (a00 * (1 - fx) * (1 - fy)
           + a10 * fx * (1 - fy)
           + a01 * (1 - fx) * fy
           + a11 * fx * fy)
    return out.astype(np.float32)


def _render_layer_png(arr, cmap, out_w=720, out_h=360, symmetric=False):
    """Render a 2D field as an opaque equirectangular PNG at out_w x out_h."""
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(arr.ndim - 2)))
    arr = np.nan_to_num(arr, nan=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if symmetric:
            m = max(abs(vmin), abs(vmax))
            vmin, vmax = -m, m
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    dpi = 120
    fig = plt.figure(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(
        arr,
        cmap=cmap,
        origin="lower",
        extent=[-180.0, 180.0, -90.0, 90.0],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()
def _ocn_read(db, source, t, depth_level, quality):
    """Read one depth slice from DYAMOND LLC2160 or ECCO fallback, return clean 2-D float32 array.

    Some datasets don't have resolution levels as coarse as the requested quality.
    We retry with progressively finer quality (up to quality=0) until one succeeds.
    """
    last_exc = None
    # Try the requested quality, then step toward 0 (coarser) until something works.
    # More-negative quality = more IDX refinement bits = finer = larger.
    # Stepping toward 0 means reducing the number of bits loaded (coarser/smaller).
    for q in range(quality, 1):
        try:
            if source == "ecco_fallback":
                arr = np.array(
                    db.db.read(time=int(t), z=[depth_level, depth_level + 1], quality=q),
                    dtype=np.float32,
                )
            else:
                arr = np.array(
                    db.read(time=int(t), z=[depth_level, depth_level + 1], quality=q),
                    dtype=np.float32,
                )
            if q != quality:
                _progress(f"  [_ocn_read] quality={quality} unavailable — used q={q} instead")
            break
        except Exception as exc:
            last_exc = exc
            continue
    else:
        raise RuntimeError(
            f"_ocn_read: cannot read t={t} depth={depth_level} at any quality {quality}..0"
        ) from last_exc
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = np.where(np.abs(arr) > 1e10, np.nan, arr)
    arr = np.where(arr < -100, np.nan, arr)
    return arr
def _emit_frame(idx, total, t_label, png_bytes):
    """Emit a single animation frame event."""
    _emit_event({
        "type": "image_frame",
        "frame_idx": idx,
        "total_frames": total,
        "t_label": t_label,
        "dataUrl": _png_data_url(png_bytes),
    })


def _period_human(angle_rad, dt_min=30.0):
    """
    Convert a DMD eigenvalue angle (radians) to a human-readable period string.
    dt_min is the time step in minutes (DYAMOND = 30 min, ECCO = 60 min).
    """
    if abs(angle_rad) < 1e-8:
        return "∞  (quasi-stationary)"
    p_steps = 2.0 * np.pi / abs(angle_rad)
    p_min   = p_steps * dt_min
    if not np.isfinite(p_min) or p_min > 999.0 * 24 * 60:
        return "∞"
    p_hours = p_min / 60.0
    p_days  = p_min / (24.0 * 60.0)
    if p_days >= 1.0:
        return f"{p_days:.1f} days"
    if p_hours >= 1.0:
        return f"{p_hours:.1f} h"
    return f"{p_min:.0f} min"


def _render_eigenvalue_plot(eigenvalues, amplitudes, data_desc):
    """
    Two-panel figure:
      Left  — eigenvalues on the complex plane + unit circle, coloured by amplitude
      Right — mode amplitude bar chart with |lambda| annotations
    Returns PNG bytes.  Emitted as the first frame before the mode animations.
    """
    n = len(eigenvalues)
    mags_b = np.abs(amplitudes)
    norm_b = mags_b / max(float(mags_b.max()), 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="black")
    fig.patch.set_facecolor("black")
    for ax in (ax1, ax2):
        ax.set_facecolor("#06101e")
        for sp in ax.spines.values():
            sp.set_color("#334")

    # ── Left: complex plane ───────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 360)
    ax1.plot(np.cos(theta), np.sin(theta), "-", color="white", alpha=0.25, lw=1.2, label="unit circle")
    ax1.axhline(0, color="white", alpha=0.15, lw=0.7)
    ax1.axvline(0, color="white", alpha=0.15, lw=0.7)

    sc = ax1.scatter(
        eigenvalues.real, eigenvalues.imag,
        c=norm_b, cmap="plasma", s=80, zorder=5,
        edgecolors="white", linewidths=0.6,
    )
    for i, e in enumerate(eigenvalues):
        ax1.annotate(
            str(i), (e.real, e.imag),
            fontsize=6, color="white", ha="center", va="center",
            zorder=10, fontweight="bold",
        )

    lim = max(float(np.abs(eigenvalues).max()) * 1.25, 1.25)
    ax1.set_xlim(-lim, lim);  ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Real(λ)", color="white", fontsize=8)
    ax1.set_ylabel("Imag(λ)", color="white", fontsize=8)
    ax1.set_title("Eigenvalue spectrum", color="white", fontsize=9, pad=6)
    ax1.tick_params(colors="white", labelsize=6)
    cb1 = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("amplitude (normalised)", color="white", fontsize=7)
    cb1.ax.tick_params(labelsize=5, colors="white")
    cb1.outline.set_edgecolor("none")

    # Annotate stability: inside unit circle = decaying, outside = growing
    ax1.text(0.02, 0.98, "inside unit circle  =  decaying\noutside  =  growing",
             transform=ax1.transAxes, fontsize=6, color="white", alpha=0.6,
             va="top", ha="left")

    # ── Right: amplitude bar chart ────────────────────────────────────────
    colors = plt.cm.plasma(norm_b)
    bars = ax2.bar(range(n), mags_b, color=colors, edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("Mode index", color="white", fontsize=8)
    ax2.set_ylabel("Amplitude |b|", color="white", fontsize=8)
    ax2.set_title("Mode amplitudes + |λ|", color="white", fontsize=9, pad=6)
    ax2.set_xticks(range(n))
    ax2.tick_params(colors="white", labelsize=7)

    for i, (bar, e) in enumerate(zip(bars, eigenvalues)):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h * 1.02,
            f"|λ|={abs(e):.3f}",
            ha="center", va="bottom", fontsize=5.5,
            color="white", rotation=75,
        )

    fig.suptitle(f"Tucker+DMD eigenvalue overview  |  {data_desc}", color="white", fontsize=9, y=1.01)
    fig.tight_layout(pad=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _dmd_behavior(mag, tol=0.01):
    """Classify a DMD eigenvalue by |λ|: growing / neutral (persists) / decaying."""
    if mag > 1.0 + tol:
        return "growing"
    if mag < 1.0 - tol:
        return "decaying"
    return "neutral"


_BEHAVIOR_COLOR = {"growing": "#f87171", "neutral": "#e5e7eb", "decaying": "#60a5fa"}


def _render_dmd_spectrum(eigenvalues, amplitudes, dt_eff_min, data_desc):
    """
    Two readable panels:
      Left  — eigenvalues on the complex plane + unit circle, coloured by behaviour
      Right — physical view: period (log, real units) vs growth rate (%/day),
              marker area ∝ amplitude, coloured by behaviour.
    """
    eigenvalues = np.asarray(eigenvalues)
    amps = np.abs(np.asarray(amplitudes))
    amp_n = amps / max(float(amps.max()), 1e-12)
    mags = np.abs(eigenvalues)
    angles = np.angle(eigenvalues)
    behav = [_dmd_behavior(float(m)) for m in mags]
    colors = [_BEHAVIOR_COLOR[b] for b in behav]

    day = 1440.0
    # growth rate %/day:  |λ| = exp(g * dt) ⇒ g = ln|λ| / dt
    growth = np.log(np.maximum(mags, 1e-12)) / (dt_eff_min / day) * 100.0
    # period in days from the eigenvalue angle
    with np.errstate(divide="ignore"):
        period_days = np.where(np.abs(angles) > 1e-8,
                               (2 * np.pi / np.abs(angles)) * dt_eff_min / day,
                               np.inf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="black")
    for ax in (ax1, ax2):
        ax.set_facecolor("#06101e")
        for sp in ax.spines.values():
            sp.set_color("#334")
        ax.tick_params(colors="white", labelsize=7)

    # ── Left: complex plane ──────────────────────────────────────────────
    th = np.linspace(0, 2 * np.pi, 360)
    ax1.plot(np.cos(th), np.sin(th), color="white", alpha=0.25, lw=1.1)
    ax1.axhline(0, color="white", alpha=0.12, lw=0.7)
    ax1.axvline(0, color="white", alpha=0.12, lw=0.7)
    ax1.scatter(eigenvalues.real, eigenvalues.imag, c=colors,
                s=40 + 240 * amp_n, edgecolors="black", linewidths=0.5, zorder=5)
    lim = max(float(mags.max()) * 1.25, 1.2)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_aspect("equal")
    ax1.set_xlabel("Re(λ)", color="white", fontsize=8)
    ax1.set_ylabel("Im(λ)", color="white", fontsize=8)
    ax1.set_title("Eigenvalues (unit circle = neutral)", color="white", fontsize=9)

    # ── Right: period vs growth (physical) ───────────────────────────────
    finite = np.isfinite(period_days)
    if finite.any():
        ax2.scatter(period_days[finite], growth[finite],
                    c=[colors[i] for i in np.where(finite)[0]],
                    s=40 + 240 * amp_n[finite], edgecolors="black", linewidths=0.5, zorder=5)
        ax2.set_xscale("log")
    ax2.axhline(0, color="white", alpha=0.3, lw=0.8, ls="--")
    ax2.set_xlabel("Period  (days, log)", color="white", fontsize=8)
    ax2.set_ylabel("Growth rate  (%/day)", color="white", fontsize=8)
    n_stat = int((~finite).sum())
    ttl = "Period vs growth  (marker ∝ amplitude)"
    if n_stat:
        ttl += f"  ·  {n_stat} stationary"
    ax2.set_title(ttl, color="white", fontsize=9)

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                      markeredgecolor="black", markersize=7, label=b)
               for b, c in _BEHAVIOR_COLOR.items()]
    ax2.legend(handles=handles, loc="best", fontsize=6, framealpha=0.2,
               labelcolor="white")

    fig.suptitle(f"DMD spectrum  |  {data_desc}", color="white", fontsize=9, y=1.02)
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_triptych(arrays, titles, suptitle, cmap="RdBu_r", vmax=None):
    """Render three side-by-side heatmaps (e.g. original / reconstruction / residual)."""
    if vmax is None:
        allv = np.concatenate([np.abs(a).ravel() for a in arrays])
        allv = allv[np.isfinite(allv)]
        vmax = float(np.percentile(allv, 99)) if allv.size else 1.0
    vmax = max(vmax, 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), facecolor="black")
    for ax, arr, ttl in zip(axes, arrays, titles):
        ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", origin="upper")
        ax.set_title(ttl, color="white", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#334")
    fig.suptitle(suptitle, color="white", fontsize=9, y=1.04)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_all_modes_frame(arrays, subtitles, suptitle, cmap, vmaxes, frame_idx, total_frames):
    """
    Render one animation frame that shows ALL DMD modes as a grid.

    arrays    : list of (H, W) float32 spatial arrays, one per mode
    subtitles : per-panel title strings
    suptitle  : figure-level title
    cmap      : matplotlib colormap name
    vmaxes    : per-mode locked vmax (symmetric: vmin = -vmax)
    """
    n     = len(arrays)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.4 * ncols, 2.9 * nrows),
        facecolor="black",
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    # Downsample large arrays to panel-scale before imshow — matplotlib resampling
    # of full-res arrays (e.g. 405×540) inside small panels is the main bottleneck.
    _MAX_PX = 256
    for i, (arr2d, subtitle, vmax_i) in enumerate(zip(arrays, subtitles, vmaxes)):
        ri, ci = divmod(i, ncols)
        ax = axes[ri][ci]
        ax.set_facecolor("black")
        arr2d = np.squeeze(np.asarray(arr2d, dtype=np.float32))
        arr2d = np.nan_to_num(arr2d, nan=0.0)
        sy = max(1, arr2d.shape[0] // _MAX_PX)
        sx = max(1, arr2d.shape[1] // _MAX_PX)
        if sy > 1 or sx > 1:
            arr2d = arr2d[::sy, ::sx]
        im = ax.imshow(arr2d, cmap=cmap, origin="lower", aspect="auto",
                       interpolation="nearest", vmin=-vmax_i, vmax=vmax_i)
        ax.set_title(subtitle, fontsize=7, color="white", pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5, colors="white")
        cb.outline.set_edgecolor("none")

    for j in range(n, nrows * ncols):
        ri, ci = divmod(j, ncols)
        axes[ri][ci].axis("off")

    fig.suptitle(
        f"{suptitle}  ·  frame {frame_idx + 1}/{total_frames}",
        fontsize=8, color="white", y=1.005,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93, wspace=0.35, hspace=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="black", edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _render_anim_frame(arr, cmap, title="", w=512, h=256, vmin=None, vmax=None):
    """
    Render one animation frame as a small PNG (fast, no chrome).

    Pass vmin/vmax to lock the colorscale across an animation sequence —
    if omitted they are computed per-frame from the 2nd/98th percentile.
    Layout is hardcoded so all frames in a sequence are pixel-identical in size.
    """
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(arr.ndim - 2)))
    arr = np.nan_to_num(arr, nan=0.0)
    if vmin is None or vmax is None:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vmin = float(np.percentile(finite, 2))  if vmin is None else vmin
            vmax = float(np.percentile(finite, 98)) if vmax is None else vmax
            if vmax <= vmin:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
    dpi = 72
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(arr, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
              interpolation="bilinear")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, color="white", pad=2, loc="left",
                     bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.55, ec="none"))
    buf = io.BytesIO()
    # Fixed size — no content-based trimming so all frames are the same dimensions.
    fig.savefig(buf, format="png", dpi=dpi, facecolor="black", bbox_inches=None)
    plt.close(fig)
    return buf.getvalue()


def _render_uv_panel(vec, n_space, cmap="RdBu_r", title=""):
    """
    Render a DMD mode or reconstruction vector as a U/V side-by-side panel.

    vec      : 1-D numpy array of length n_space (stacked [U|V]).
    n_space  : total spatial points (must equal len(vec)).
    Returns  : PNG bytes.
    """
    half = n_space // 2
    u_vec = vec[:half]
    v_vec = vec[half:]

    # Find the closest square that fits.
    side = int(np.round(np.sqrt(half)))
    # Pad to side*side if needed.
    u_2d = _pad_to_square(u_vec, side)
    v_2d = _pad_to_square(v_vec, side)

    # Combined colourscale across both components.
    both = np.concatenate([u_2d.ravel(), v_2d.ravel()])
    finite = both[np.isfinite(both)]
    if finite.size > 0:
        m = max(abs(float(np.percentile(finite, 2))), abs(float(np.percentile(finite, 98))))
        vmin, vmax = -m if m > 0 else -1e-6, m if m > 0 else 1e-6
    else:
        vmin, vmax = -1e-6, 1e-6

    dpi = 72
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    fig.patch.set_facecolor("black")
    for ax, data, comp in zip(axes, [u_2d, v_2d], ["U (zonal)", "V (meridional)"]):
        im = ax.imshow(data, cmap=cmap, origin="lower", aspect="equal",
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(comp, color="white", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, color="white", fontsize=9)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="black", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


def _pad_to_square(vec, side):
    """Reshape vec to (side, side), zero-padding if needed."""
    needed = side * side
    if len(vec) < needed:
        vec = np.concatenate([vec, np.zeros(needed - len(vec), dtype=np.float32)])
    else:
        vec = vec[:needed]
    return vec.reshape(side, side)


def _render_wind_panel(vec, n_space, title="", vmax=None):
    """
    Render a DMD wind mode/reconstruction as a single panel:
    background = speed magnitude sqrt(U²+V²) with quiver arrows for direction.

    Layout is fully hardcoded (no tight_layout, no bbox_inches='tight', no
    aspect='equal') so every frame in an animation lands at exactly the same
    pixel position — no jitter.

    Pass vmax to lock the colorscale across an animation sequence.
    """
    half = n_space // 2
    u_vec = vec[:half]
    v_vec = vec[half:]
    side = int(np.round(np.sqrt(half)))
    u_2d = _pad_to_square(u_vec, side)
    v_2d = _pad_to_square(v_vec, side)

    speed = np.sqrt(u_2d ** 2 + v_2d ** 2)

    # ~20×20 subsampled quiver grid
    step = max(1, side // 20)
    ys = np.arange(0, side, step)
    xs = np.arange(0, side, step)
    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    u_q = u_2d[yg, xg]
    v_q = v_2d[yg, xg]
    mag_q = np.sqrt(u_q ** 2 + v_q ** 2)
    safe_mag = np.where(mag_q > 0, mag_q, 1.0)
    p90 = float(np.percentile(mag_q[mag_q > 0], 90)) if (mag_q > 0).any() else 1.0
    scale_factor = np.clip(mag_q / max(p90, 1e-8), 0, 1)
    u_n = (u_q / safe_mag) * scale_factor
    v_n = (v_q / safe_mag) * scale_factor

    if vmax is None:
        finite = speed.ravel()[np.isfinite(speed.ravel())]
        vmax = float(np.percentile(finite, 99)) if finite.size > 0 else 1.0

    dpi = 110
    # Fixed canvas: 1100×660 px
    fig = plt.figure(figsize=(10, 6), dpi=dpi, facecolor="black")
    # Fixed axes rect: [left, bottom, width, height] in figure fraction.
    # Leave 0.10 on the right for the colorbar; 0.08 on top for the title.
    ax = fig.add_axes([0.03, 0.03, 0.82, 0.88])
    ax.set_facecolor("black")
    # aspect="auto" so the axes fill the rect exactly — no per-frame repositioning.
    im = ax.imshow(speed, cmap="plasma", origin="lower", aspect="auto",
                   vmin=0, vmax=max(vmax, 1e-8), interpolation="bilinear")
    ax.quiver(xg, yg, u_n, v_n, color="white", alpha=0.55,
              scale=22, width=0.003, headwidth=3, headlength=4)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, color="white", fontsize=8, pad=4)
    # Fixed colorbar axes — same position every frame.
    cax = fig.add_axes([0.87, 0.03, 0.025, 0.88])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Wind speed (m/s)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    buf = io.BytesIO()
    # bbox_inches=None → honour the figure size exactly; no content-based trimming.
    fig.savefig(buf, format="png", dpi=dpi, facecolor="black", bbox_inches=None)
    plt.close(fig)
    return buf.getvalue()


def _render_dmd_overview(modes_r, modes_i, eigs_r, eigs_i, dyn_r, dyn_i, n_space, dt_min=30.0):
    """
    DMD overview: eigenvalue scatter + top-3 amplitude-mode speed heatmaps,
    arranged side-by-side in a single wide row.
    Returns PNG bytes.
    """
    n_modes = modes_r.shape[1]
    amps = np.sqrt(dyn_r[:, 0] ** 2 + dyn_i[:, 0] ** 2)
    top3 = np.argsort(amps)[::-1][:3].tolist()

    eigs = eigs_r + 1j * eigs_i
    eig_mags = np.abs(eigs)
    half = n_space // 2
    side = int(np.round(np.sqrt(half)))

    def _period_str(eig):
        return _period_human(float(np.angle(eig)), dt_min=dt_min)

    # Shared vmax across the 3 mode panels so scales are comparable
    all_speeds = []
    mode_data = []
    for m in top3:
        mode_re = modes_r[:, m]
        u_2d = _pad_to_square(mode_re[:half], side)
        v_2d = _pad_to_square(mode_re[half:], side)
        speed = np.sqrt(u_2d ** 2 + v_2d ** 2)
        mode_data.append((m, u_2d, v_2d, speed))
        all_speeds.append(speed.ravel())
    combined = np.concatenate(all_speeds)
    finite_all = combined[np.isfinite(combined)]
    shared_vmax = float(np.percentile(finite_all, 99)) if finite_all.size > 0 else 1e-8

    dpi = 100
    # 4 columns: eigenvalues + 3 mode panels
    fig = plt.figure(figsize=(20, 6), dpi=dpi, facecolor="black")
    gs = fig.add_gridspec(1, 4, wspace=0.28,
                          left=0.04, right=0.98, top=0.88, bottom=0.08,
                          width_ratios=[1, 1, 1, 1])

    # ── Eigenvalue scatter ──────────────────────────────────────────────────
    ax_eig = fig.add_subplot(gs[0, 0])
    ax_eig.set_facecolor("#0d0d0d")
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_eig.plot(np.cos(theta), np.sin(theta), "w-", lw=1.0, alpha=0.4, zorder=1)
    ax_eig.axhline(0, color="w", lw=0.4, alpha=0.25)
    ax_eig.axvline(0, color="w", lw=0.4, alpha=0.25)

    sc = ax_eig.scatter(
        eigs_r, eigs_i,
        c=eig_mags, cmap="coolwarm", s=max(6, 200 // n_modes),
        vmin=0.85, vmax=1.15, zorder=3, alpha=0.9,
    )
    for rank, m in enumerate(top3):
        ax_eig.scatter(eigs_r[m], eigs_i[m], s=110, color="yellow",
                       zorder=5, edgecolors="orange", linewidths=0.9)
        ax_eig.annotate(
            f"#{m}  T={_period_str(eigs[m])}",
            (eigs_r[m], eigs_i[m]),
            textcoords="offset points", xytext=(6, 5),
            color="yellow", fontsize=7.5, zorder=6,
        )
    cb = fig.colorbar(sc, ax=ax_eig, shrink=0.7, pad=0.04)
    cb.set_label("|λ|", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax_eig.set_xlim(-1.35, 1.35)
    ax_eig.set_ylim(-1.35, 1.35)
    ax_eig.set_aspect("equal")
    ax_eig.set_title("Eigenvalues", color="white", fontsize=11, pad=7)
    ax_eig.set_xlabel("Re(λ)", color="white", fontsize=9)
    ax_eig.set_ylabel("Im(λ)", color="white", fontsize=9)
    ax_eig.tick_params(colors="white", labelsize=8)
    for sp in ax_eig.spines.values():
        sp.set_color("#444")

    # ── Top-3 mode speed heatmaps (no quiver clutter in overview) ──────────
    for col_i, (m, u_2d, v_2d, speed) in enumerate(mode_data):
        ax = fig.add_subplot(gs[0, col_i + 1])
        ax.set_facecolor("black")
        im = ax.imshow(speed, cmap="plasma", origin="lower", aspect="equal",
                       vmin=0, vmax=max(shared_vmax, 1e-8), interpolation="bilinear")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     ).ax.yaxis.set_tick_params(color="white", labelsize=6)
        plt.setp(fig.axes[-1].yaxis.get_ticklabels(), color="white")

        eig_mag = float(abs(eigs[m]))
        period_label = _period_str(eigs[m])
        rank = top3.index(m) + 1
        ax.set_title(
            f"Mode {m}  (rank #{rank} by amplitude)\n|λ| = {eig_mag:.3f}   T = {period_label}",
            color="white", fontsize=8.5, pad=5, linespacing=1.4,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "DMD Wind Analysis — Eigenvalues & Top-3 Mode Shapes\n"
        "DYAMOND GEOS · Face 0 · Z-level 50 (near-surface) · U+V wind speed magnitude (m/s)",
        color="white", fontsize=11, y=0.99, linespacing=1.5,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="black", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
_MAX_FETCH_MB = 10_000   # 10 GB hard cap — raise an error before downloading absurd amounts

# OpenVisus IDX quality convention — same rule for ALL datasets:
#   more negative  → more IDX refinement bits loaded → finer → larger frames → slower
#   less negative  → fewer bits → coarser → smaller frames → faster fetch
#
# Dataset pyramid depths (approximate max useful depth):
#   DYAMOND : ~9 levels  — quality=-7 is default; -9 is near-finest available
#   ECCO    : ~15 levels — quality=-9 is mid-range; -12 or -15 is very fine (huge frames)
#   CMIP6   : no IDX pyramid — quality ignored
_QUALITY_TIP = {
    "dyamond": "for smaller frames use less negative quality (e.g. -4 or -5); -9 is near the finest available",
    "ecco":    "for smaller frames use less negative quality (e.g. -4 or -6); -9 is mid-range, -12/-15 are very fine",
    "cmip6":   "quality has no effect for CMIP6 — shorten the time window to reduce step count",
}


def _probe_and_plan(fetch_fn, ts_candidates, n_ts_full, n_workers=32, source=""):
    """
    Fetch ONE probe frame to measure the actual array size, report fetch plan,
    and enforce a hard cap (_MAX_FETCH_MB) to prevent accidental huge downloads.
    Returns ALL candidates if within budget; raises RuntimeError otherwise.
    """
    _progress(f"Probing frame size (1 of {len(ts_candidates)} steps)...")
    probe = fetch_fn(ts_candidates[0])   # (t, arr2d) or (t, arr2d, q)
    frame_bytes = probe[1].nbytes        # second element is always the array

    n_fetch      = len(ts_candidates)
    est_mb       = n_fetch * frame_bytes / 1e6
    ms_per_req   = 150 + frame_bytes / (100e6 / 8) * 1000
    est_s        = max(1.0, n_fetch / n_workers * ms_per_req / 1000)
    coverage_pct = n_fetch / max(n_ts_full, 1) * 100

    _progress(f"  Frame size : {frame_bytes / 1024:.1f} KB")
    _progress(f"  Steps      : {n_fetch} / {n_ts_full} total ({coverage_pct:.1f}% of dataset)")
    _progress(f"  Total data : ~{est_mb:.1f} MB")
    _progress(f"  Est. time  : ~{est_s:.0f}s  ({n_workers} workers)")

    if est_mb > _MAX_FETCH_MB:
        tip = _QUALITY_TIP.get(source, "reduce the time window or adjust quality")
        raise RuntimeError(
            f"Fetch aborted: estimated {est_mb / 1024:.1f} GB exceeds the {_MAX_FETCH_MB // 1024} GB limit.\n"
            f"  Frame size : {frame_bytes / 1024:.1f} KB  ×  {n_fetch} steps\n"
            f"  Tip: {tip}\n"
            f"  Or: shorten the time range to reduce the number of steps."
        )

    return ts_candidates
