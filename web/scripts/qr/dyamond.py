"""qr.dyamond — DYAMOND GEOS queries: single face, z-sweep, cross-section, all-faces, coupled, ocean."""
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

from qr.common import *  # noqa: F401,F403


def query_dyamond(p):
    from analysis.core.utils import load_dyamond_geos
    variable = p.get("variable", "u")
    face = int(p.get("face", 0))
    t_start = int(p.get("t_start", 0))
    t_end = int(p.get("t_end", 4))
    z_level = int(p.get("z_level", 50))
    quality = int(p.get("quality", -4))
    render_mode = p.get("render_mode", "grid")
    all_faces = bool(p.get("all_faces", False))
    cmap = _DYAMOND_CMAP.get(variable, "RdBu_r")

    _progress(f"Using quality={quality}")

    # Multi-face cube render: emit one image_face event per face 0..5.
    if all_faces and render_mode == "globe":
        _progress("Rendering all 6 cubed-sphere faces (parallel)")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _read_face(f):
            db = load_dyamond_geos(variable, f)
            all_ts = list(db.getTimesteps())
            t = _midpoint_timestep(all_ts, t_start, t_end)
            arr = _dyamond_read(db, t, z_level, quality)
            return f, np.squeeze(arr)

        # Fetch all 6 faces concurrently (each is a separate dataset load +
        # network read). Rendering + emission stay single-threaded and in face
        # order; face orientation math (face_idx) is unchanged.
        face_arrs = {}
        done = 0
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                for fut in as_completed([pool.submit(_read_face, f) for f in range(6)]):
                    f, arr = fut.result()
                    face_arrs[f] = arr
                    done += 1
                    _progress(f"  fetched face {f} ({done}/6)")
        except RuntimeError as exc:
            if "timed out" in str(exc).lower() or "unreachable" in str(exc).lower():
                raise RuntimeError(
                    "DYAMOND GEOS server is unreachable "
                    "(nsdf-climate3-origin.nationalresearchplatform.org:50098). "
                    "Try ECCO or CMIP6 instead."
                ) from exc
            raise

        for f in range(6):
            png = _render_face_square(face_arrs[f], cmap=cmap, face_idx=f)
            _emit_event({
                "type": "image_face",
                "face": f,
                "dataUrl": _png_data_url(png),
            })
            _progress(f"Face {f} emitted")
        return b""  # all output was streamed

    _progress(f"Loading DYAMOND GEOS variable={variable} face={face}")
    db = load_dyamond_geos(variable, face)
    _progress("Listing timesteps")
    all_ts = list(db.getTimesteps())

    if render_mode == "globe":
        t = _midpoint_timestep(all_ts, t_start, t_end)
        _progress(f"Reading midpoint timestep {t}")
        arr = _dyamond_read(db, t, z_level, quality)
        return _render_equirectangular(np.squeeze(arr), cmap=cmap)

    ts = _subsample(all_ts, t_start, t_end, cap=6)
    _progress(f"Subsampling {len(ts)} of {len(all_ts)} timesteps")
    frames, titles = [], []
    for i, t in enumerate(ts):
        _progress(f"Reading timestep {t} ({i + 1}/{len(ts)})")
        arr = _dyamond_read(db, t, z_level, quality)
        frames.append(np.squeeze(arr))
        titles.append(f"t={t}")
    return _render_grid(
        frames,
        titles,
        f"DYAMOND GEOS | {variable} | face {face} | z={z_level} | q={quality}",
        cmap=cmap,
    )


# ── Face Explorer queries ──────────────────────────────────────────────────────
def query_face_z_sweep(p):
    """Animate through z-levels on a single DYAMOND face."""
    from analysis.core.utils import load_dyamond_geos
    variable = p.get("variable", "u")
    face = int(p.get("face", 0))
    z_start = max(0, int(p.get("z_start", 0)))
    z_end = min(51, int(p.get("z_end", 20)))
    if z_end < z_start:
        z_end = z_start
    quality = int(p.get("quality", -4))
    t_snap = int(p.get("t_snap", 5000))
    cmap = _DYAMOND_CMAP.get(variable, "RdBu_r")

    _progress(f"Z-sweep face={face} variable={variable} z={z_start}..{z_end} q={quality}")
    db = load_dyamond_geos(variable, face)
    all_ts = list(db.getTimesteps())
    t = all_ts[min(t_snap, len(all_ts) - 1)]
    _progress(f"Using timestep index {t_snap} → actual t={t}")

    levels = list(range(z_start, z_end + 1))
    if len(levels) > 20:
        indices = [int(round(i * (len(levels) - 1) / 19)) for i in range(20)]
        levels = [levels[i] for i in sorted(set(indices))]

    total = len(levels)
    for i, z in enumerate(levels):
        _progress(f"Reading z={z} ({i + 1}/{total})")
        arr = _dyamond_read(db, t, z, quality)
        png = _render_face_square(np.squeeze(arr), cmap=cmap, face_idx=face)
        _emit_event({
            "type": "image_frame",
            "frame_idx": i,
            "total_frames": total,
            "t_label": f"z={z}",
            "dataUrl": _png_data_url(png),
        })

    return b""


def query_face_cross_section(p):
    """Render a longitude-height cross-section for a fixed y-row on a DYAMOND face."""
    from analysis.core.utils import load_dyamond_geos
    variable = p.get("variable", "u")
    face = int(p.get("face", 0))
    quality = int(p.get("quality", -4))
    t_snap = int(p.get("t_snap", 5000))
    y_row = max(0, min(1439, int(p.get("y_row", 720))))
    cmap = _DYAMOND_CMAP.get(variable, "RdBu_r")

    _progress(f"Cross-section face={face} variable={variable} y_row={y_row} q={quality}")
    db = load_dyamond_geos(variable, face)
    all_ts = list(db.getTimesteps())
    t = all_ts[min(t_snap, len(all_ts) - 1)]
    _progress(f"Using timestep index {t_snap} → actual t={t}")

    q = int(quality)
    last_exc = None
    arr = None
    for attempt in range(6):
        try:
            _progress(f"Reading y-row={y_row} all z (attempt {attempt + 1}, q={q})")
            arr = np.array(
                db.read(time=int(t), y=[y_row, min(y_row + 1, 1439)], quality=q),
                dtype=np.float32,
            )
            if attempt > 0:
                _progress(f"  succeeded at quality={q}")
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "end_resolution" in msg or "begin query failed" in msg:
                q = min(q + 1, 0)
                _progress(f"  quality retry → {q}")
            else:
                raise
    if arr is None:
        raise last_exc

    arr = np.squeeze(arr)
    arr = np.nan_to_num(arr, nan=0.0)
    # Normalise to 2-D (nz, nx)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    elif arr.ndim == 1:
        arr = arr[np.newaxis, :]

    finite = arr[np.isfinite(arr)]
    vmin = float(np.percentile(finite, 2)) if finite.size > 0 else 0.0
    vmax = float(np.percentile(finite, 98)) if finite.size > 0 else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    im = ax.imshow(arr, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    ax.set_xlabel("Column within face (px)", fontsize=10)
    ax.set_ylabel("z level  (0 = surface → 51 = top)", fontsize=10)
    region = _FACE_REGIONS[face] if 0 <= face < 6 else ""
    ax.set_title(
        f"DYAMOND GEOS · {variable} · face {face} ({region}) · y-row {y_row} · q={q}",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def query_face_all_panel(p):
    """Render all 6 DYAMOND faces as a 2×3 matplotlib grid at a fixed z-level."""
    from analysis.core.utils import load_dyamond_geos
    from concurrent.futures import ThreadPoolExecutor, as_completed
    variable = p.get("variable", "u")
    z_level = max(0, min(51, int(p.get("z_level", 50))))
    quality = int(p.get("quality", -4))
    t_snap = int(p.get("t_snap", 5000))
    cmap = _DYAMOND_CMAP.get(variable, "RdBu_r")

    _progress(f"All-faces panel variable={variable} z={z_level} q={quality}")

    def _read_face(f):
        db = load_dyamond_geos(variable, f)
        all_ts = list(db.getTimesteps())
        t = all_ts[min(t_snap, len(all_ts) - 1)]
        arr = _dyamond_read(db, t, z_level, quality)
        return f, np.squeeze(arr)

    face_arrs = {}
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for fut in as_completed([pool.submit(_read_face, f) for f in range(6)]):
                f, arr = fut.result()
                face_arrs[f] = arr
                done += 1
                _progress(f"  fetched face {f} ({done}/6)")
    except RuntimeError as exc:
        if "timed out" in str(exc).lower() or "unreachable" in str(exc).lower():
            raise RuntimeError(
                "DYAMOND GEOS server is unreachable. Try ECCO or CMIP6 instead."
            ) from exc
        raise

    frames, titles = [], []
    for f in range(6):
        arr = face_arrs[f].copy()
        if f == 4:
            arr = np.flipud(np.rot90(arr, k=1))
        elif f == 5:
            arr = arr.T
        frames.append(arr)
        titles.append(f"Face {f} ({_FACE_REGIONS[f]})")

    return _render_grid(
        frames,
        titles,
        f"DYAMOND GEOS | {variable} | z={z_level} | q={quality} | all 6 faces",
        cmap=cmap,
    )
def query_dyamond_coupled(p):
    """
    Render a coupled atmosphere-over-ocean composite globe layer pair.

    Loads 6 DYAMOND GEOS atmosphere faces, reprojects them to a 720x360
    equirectangular grid, and renders an ocean snapshot from
    `load_dyamond_llc2160` (which falls back to ECCO LLC4320). Emits
    two `image_layer` events: one for the ocean and one for the atmosphere.
    """
    from analysis.core.utils import load_dyamond_geos, load_dyamond_llc2160
    atm_variable = p.get("atm_variable", "u")
    ocn_variable = p.get("ocn_variable", "theta")
    t_atm = int(p.get("t_atm", 10))
    t_ocn = int(p.get("t_ocn", 200))
    z_level_atm = int(p.get("z_level_atm", 50))
    depth_level_ocn = int(p.get("depth_level_ocn", 10))
    quality = int(p.get("quality", -4))

    atm_cmap = _DYAMOND_CMAP.get(atm_variable, "RdBu_r")
    ocn_cmap = _OCN_CMAP.get(ocn_variable, "viridis")

    _progress(
        f"Coupled mode: atm={atm_variable} z={z_level_atm} t_atm={t_atm} | "
        f"ocn={ocn_variable} depth={depth_level_ocn} t_ocn={t_ocn} q={quality}"
    )

    # 1. Load + read 6 atmosphere faces at the same timestep + z + quality (parallel).
    def _read_atm_face(f):
        _progress(f"Loading DYAMOND atm face {f}/{atm_variable}")
        db = load_dyamond_geos(atm_variable, f)
        all_ts = list(db.getTimesteps())
        if not all_ts:
            return f, np.zeros((1, 1), dtype=np.float32)
        ti = max(0, min(len(all_ts) - 1, t_atm))
        t = all_ts[ti]
        _progress(f"Reading atm face {f} t={t} z={z_level_atm} q={quality}")
        arr = _dyamond_read(db, int(t), z_level_atm, quality)
        return f, np.squeeze(arr)

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    _face_dict = {}
    _done = 0
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for fut in _as_completed([pool.submit(_read_atm_face, f) for f in range(6)]):
                f, arr = fut.result()
                _face_dict[f] = arr
                _done += 1
                _progress(f"  atm face {f} done ({_done}/6)")
    except RuntimeError as exc:
        if "timed out" in str(exc).lower() or "unreachable" in str(exc).lower():
            raise RuntimeError(
                "DYAMOND GEOS atmosphere server is unreachable "
                "(nsdf-climate3-origin.nationalresearchplatform.org:50098). "
                "Try ATM-only or ECCO instead."
            ) from exc
        raise
    face_arrays = [_face_dict[f] for f in range(6)]

    _progress("Reprojecting cubed-sphere → equirectangular (720x360)")
    atm_equirect = _equirect_from_cubed_faces(face_arrays, out_w=720, out_h=360)

    # 2. Render and emit the atmosphere layer.
    _progress("Encoding atmosphere PNG")
    atm_symmetric = atm_variable in _DYAMOND_SYMMETRIC
    atm_png = _render_layer_png(
        atm_equirect, cmap=atm_cmap, out_w=720, out_h=360, symmetric=atm_symmetric,
    )

    # 3. Read the ocean snapshot (DYAMOND LLC2160 → falls back to ECCO).
    _progress(f"Loading ocean variable={ocn_variable}")
    db_ocn, source = load_dyamond_llc2160(ocn_variable)
    all_ts_ocn = list(db_ocn.getTimesteps()) if db_ocn else []
    if not all_ts_ocn:
        ocn_arr = np.zeros((360, 720), dtype=np.float32)
    else:
        ti_ocn = max(0, min(len(all_ts_ocn) - 1, t_ocn))
        t_ocn_real = all_ts_ocn[ti_ocn]
        _progress(f"Reading ocean t={t_ocn_real} depth={depth_level_ocn} q={quality} src={source}")
        # ECCO fallback uses db.db.read; native LLC2160 (ov.LoadDataset) uses db.read.
        if source == "ecco_fallback":
            arr = np.array(
                db_ocn.db.read(
                    time=int(t_ocn_real),
                    z=[depth_level_ocn, depth_level_ocn + 1],
                    quality=quality,
                ),
                dtype=np.float32,
            )
        else:
            arr = np.array(
                db_ocn.read(
                    time=int(t_ocn_real),
                    z=[depth_level_ocn, depth_level_ocn + 1],
                    quality=quality,
                ),
                dtype=np.float32,
            )
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        ocn_arr = arr

    _progress("Encoding ocean PNG")
    ocn_png = _render_layer_png(ocn_arr, cmap=ocn_cmap, out_w=720, out_h=360)

    # 4. Emit ocean first (base layer), then atmosphere (overlay).
    _emit_event({
        "type": "image_layer",
        "layer": "ocean",
        "dataUrl": _png_data_url(ocn_png),
    })
    _emit_event({
        "type": "image_layer",
        "layer": "atm",
        "dataUrl": _png_data_url(atm_png),
    })
    _progress("Coupled globe layers emitted")
    return b""  # all output streamed via events
def query_dyamond_ocean(p):
    """
    Render a DYAMOND LLC2160 ocean snapshot (or animation).

    Uses the per-variable URLs from the contest reference notebook
    (ieee_scivis_dyamond_ocean.ipynb). Falls back to ECCO LLC4320
    for theta/salt if LLC2160 is unreachable — same spatial grid,
    different time period.

    render_mode "globe" → equirectangular PNG for the 3-D globe viewer
    render_mode "grid"  → flat matplotlib imshow (exactly like the reference nb)
    animate=True        → emit image_frame events for each timestep
    """
    from analysis.core.utils import load_dyamond_llc2160
    variable    = p.get("variable", "theta")
    t_start     = int(p.get("t_start", 0))
    t_end       = int(p.get("t_end", 0))
    depth_level = int(p.get("depth_level", 0))
    quality     = int(p.get("quality", -6))
    render_mode = p.get("render_mode", "globe")
    animate     = bool(p.get("animate", False))
    n_frames    = min(int(p.get("n_frames", 20)), 30)
    cmap        = _OCN_CMAP.get(variable, "viridis")
    meters      = DEPTH_LEVELS_METERS[depth_level] if 0 <= depth_level < len(DEPTH_LEVELS_METERS) else 0.0

    _progress(f"Loading DYAMOND ocean LLC2160 variable={variable}")
    db, source = load_dyamond_llc2160(variable)
    _progress(f"Source: {source}")

    all_ts = list(db.getTimesteps())
    _progress(f"Timesteps available: {len(all_ts)}")

    # ── Animation path ──────────────────────────────────────────────────────
    if animate:
        ts = _subsample(all_ts, t_start, t_end, cap=n_frames)
        _progress(f"Generating {len(ts)} animation frames at depth_level={depth_level} mode={render_mode}")
        vmin_g = vmax_g = None
        for i, t in enumerate(ts):
            _progress(f"Frame {i + 1}/{len(ts)} t={t}")
            arr = _ocn_read(db, source, t, depth_level, quality)
            date_str = _ecco_date(t)
            if render_mode == "globe" and i == 0:
                fin = arr[np.isfinite(arr)]
                if fin.size > 0:
                    vmin_g = float(np.percentile(fin, 2))
                    vmax_g = float(np.percentile(fin, 98))
            label = f"DYAMOND ocean {variable} | L{depth_level} ({meters:.0f} m) | {date_str}"
            if render_mode == "globe":
                png = _render_equirectangular(arr, cmap=cmap, vmin=vmin_g, vmax=vmax_g, llc=True)
            else:
                png = _render_anim_frame(arr, cmap, title=label)
            _emit_frame(i, len(ts), date_str, png)
        return b""

    # ── Single snapshot path ────────────────────────────────────────────────
    t = _midpoint_timestep(all_ts, t_start, t_end)
    _progress(f"Reading t={t} depth_level={depth_level} quality={quality}")
    arr = _ocn_read(db, source, t, depth_level, quality)
    _progress(f"Array shape {arr.shape}  range [{float(np.nanmin(arr)):.2f}, {float(np.nanmax(arr)):.2f}]")

    if render_mode == "globe":
        return _render_equirectangular(arr, cmap=cmap, llc=True)
    else:
        return _render_grid(
            [arr],
            [f"{variable} | L{depth_level} ({meters:.0f} m) | t={t}"],
            f"DYAMOND LLC2160 ({source}) | {variable} | q={quality}",
            cmap=cmap,
        )
