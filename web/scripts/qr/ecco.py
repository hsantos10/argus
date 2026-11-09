"""qr.ecco — ECCO LLC4320 ocean query."""
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


def query_ecco(p):
    from analysis.core.utils import load_dyamond_llc2160
    variable = p.get("variable", "theta")
    t_start = int(p.get("t_start", 0))
    t_end = int(p.get("t_end", 3))
    d0 = int(p.get("depth_start", 0))
    d1 = int(p.get("depth_end", max(d0 + 1, 5)))
    if d1 <= d0:
        d1 = d0 + 1
    quality = int(p.get("quality", -6))
    render_mode = p.get("render_mode", "grid")
    cmap = _ECCO_CMAP.get(variable, "cividis")

    _progress(f"Using quality={quality}")
    _progress(f"Loading ocean dataset (LLC2160 → ECCO fallback) for {variable}")
    db, _db_src = load_dyamond_llc2160(variable)
    _progress(f"  source={_db_src}")
    _progress("Ocean dataset loaded — listing timesteps")
    all_ts = list(db.getTimesteps())

    # Coarsen in steps of 3 (not 1) so a deep level that lacks the finest
    # pyramid resolves in ~2-3 reads instead of thrashing through every integer
    # quality (a 13-read stall per deep slice was the depth-stack bottleneck).
    # Always include 0 (coarsest) as the final fallback.
    _q_candidates = list(range(quality, 1, 3))
    if 0 not in _q_candidates:
        _q_candidates.append(0)

    def _qecco_read(t, z0, z1):
        """Read one z-slice with automatic quality fallback (coarsens until success)."""
        last_exc = None
        for q in _q_candidates:
            try:
                if _db_src == "ecco_fallback":
                    arr = db.db.read(time=int(t), z=[z0, z1], quality=q)
                else:
                    arr = db.read(time=int(t), z=[z0, z1], quality=q)
                if q != quality:
                    _progress(f"  quality={quality} unavailable — used q={q}")
                return arr
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"_qecco_read: cannot read t={t} z=[{z0},{z1}] at any quality {_q_candidates}"
        ) from last_exc

    src_name = "ECCO LLC4320" if _db_src == "ecco_fallback" else "LLC2160"

    if render_mode == "globe":
        t = _midpoint_timestep(all_ts, t_start, t_end)
        # Multi-depth scrubber: emit one image_depth event per level in [d0, d1].
        if d1 > d0:
            full = list(range(d0, d1 + 1))
            if len(full) <= 6:
                levels = full
            else:
                levels = sorted({
                    int(round(d0 + (d1 - d0) * i / 5))
                    for i in range(6)
                })
            _progress(f"Multi-depth sweep: {len(levels)} levels from {d0} to {d1} (parallel)")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os as _os
            # Disk cache (shared with the dmd_live frame cache, persists across
            # runs) so re-viewing the same depth stack is instant — the deep
            # ECCO slices are slow to fetch from NSDF but identical each run.
            _cache_dir = _os.path.join(_os.getcwd(), "outputs", "frame_cache")
            _os.makedirs(_cache_dir, exist_ok=True)

            def _read_level(L):
                key = f"ecco_depth_{variable}_d{L}_q{quality}_t{int(t)}"
                cpath = _os.path.join(_cache_dir, key + ".npy")
                if _os.path.exists(cpath):
                    try:
                        return L, np.load(cpath)
                    except Exception:
                        pass
                arr = np.squeeze(np.array(_qecco_read(t, L, L + 1), dtype=np.float32))
                if arr.ndim == 3:
                    arr = arr.mean(axis=0)
                try:
                    np.save(cpath, arr)
                except Exception:
                    pass  # never fail a run due to cache write
                return L, arr

            # Fetch all levels concurrently (network-bound); keep matplotlib
            # rendering + emission single-threaded and in level order.
            arrs = {}
            done = 0
            with ThreadPoolExecutor(max_workers=min(len(levels), 8)) as pool:
                for fut in as_completed([pool.submit(_read_level, L) for L in levels]):
                    L, arr = fut.result()
                    arrs[L] = arr
                    done += 1
                    _progress(f"  fetched depth level {L} ({done}/{len(levels)})")

            for i, L in enumerate(levels):
                png = _render_equirectangular(arrs[L], cmap=cmap, llc=True)
                meters = DEPTH_LEVELS_METERS[L] if 0 <= L < len(DEPTH_LEVELS_METERS) else -1.0
                _emit_event({
                    "type": "image_depth",
                    "idx": i,
                    "level": int(L),
                    "meters": float(meters),
                    "dataUrl": _png_data_url(png),
                })
            return b""
        _progress(f"Reading midpoint timestep {t} depth[{d0},{d0 + 1})")
        arr = np.squeeze(np.array(_qecco_read(t, d0, d0 + 1), dtype=np.float32))
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        return _render_equirectangular(arr, cmap=cmap, llc=True)

    ts = _subsample(all_ts, t_start, t_end, cap=6)
    _progress(f"Subsampling {len(ts)} of {len(all_ts)} timesteps")
    frames, titles = [], []
    for i, t in enumerate(ts):
        _progress(f"Reading timestep {t} ({i + 1}/{len(ts)}) depth[{d0},{d1})")
        arr = np.squeeze(np.array(_qecco_read(t, d0, d1), dtype=np.float32))
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        frames.append(arr)
        titles.append(f"t={t}")
    return _render_grid(
        frames,
        titles,
        f"{src_name} | {variable} | depth [{d0},{d1}) | q={quality}",
        cmap=cmap,
    )
