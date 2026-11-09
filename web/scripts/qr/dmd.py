"""qr.dmd — Animation + DMD queries: query_animate, query_dmd (precomputed), query_dmd_live."""
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


def query_animate(p):
    """
    Multi-frame animation of a live dataset.
    Dispatches to the right dataset and emits `image_frame` events.

    render_mode "globe" → equirectangular PNGs (2048×1024, no chrome) suitable
                          for use as a Three.js globe texture.  Colorscale is
                          locked to the first frame so the globe doesn't flicker.
    render_mode "grid"  → small flat matplotlib frames with title bar (legacy).
    """
    dataset     = p.get("dataset", "ecco")
    n_frames    = min(int(p.get("n_frames", 20)), 30)
    render_mode = p.get("render_mode", "grid")

    def _frame_png(arr, cmap, label, vmin_g, vmax_g, is_globe, llc=False):
        """Render one frame, respecting render_mode and locked colorscale."""
        if is_globe:
            return _render_equirectangular(arr, cmap=cmap, vmin=vmin_g, vmax=vmax_g, llc=llc)
        return _render_anim_frame(arr, cmap, title=label)

    if dataset == "dyamond":
        from analysis.core.utils import load_dyamond_geos
        variable = p.get("variable", "u")
        face     = int(p.get("face", 0))
        t_start  = int(p.get("t_start", 0))
        t_end    = int(p.get("t_end", 160))
        z_level  = int(p.get("z_level", 50))
        quality  = int(p.get("quality", -6))
        cmap     = _DYAMOND_CMAP.get(variable, "RdBu_r")
        all_faces = bool(p.get("all_faces", False))

        # Multi-face globe assembly deferred (face ordering unverified).
        # If all_faces was requested, fall through to single-face path using face=face.
        if all_faces:
            _progress(f"Note: multi-face reproject deferred (ordering unverified); using face={face}.")

        _progress(f"Animation: DYAMOND {variable} face={face} z={z_level} q={quality} mode={render_mode}")
        db = load_dyamond_geos(variable, face)
        all_ts = list(db.getTimesteps())
        ts = _subsample(all_ts, t_start, t_end, cap=n_frames)
        _progress(f"Generating {len(ts)} animation frames")

        vmin_g = vmax_g = None
        for i, t in enumerate(ts):
            _progress(f"Frame {i + 1}/{len(ts)} t={t}")
            arr = np.squeeze(_dyamond_read(db, t, z_level, quality))
            if render_mode == "globe" and i == 0:
                fin = arr[np.isfinite(arr)]
                if fin.size > 0:
                    vmin_g = float(np.percentile(fin, 2))
                    vmax_g = float(np.percentile(fin, 98))
            label = f"DYAMOND {variable} | face {face} | z={z_level} | {_dyamond_date(t)}"
            png = _frame_png(arr, cmap, label, vmin_g, vmax_g, render_mode == "globe")
            _emit_frame(i, len(ts), _dyamond_date(t), png)

    elif dataset == "ecco":
        from analysis.core.utils import load_dyamond_llc2160
        variable    = p.get("variable", "theta")
        t_start     = int(p.get("t_start", 0))
        t_end       = int(p.get("t_end", 100))
        depth_level = int(p.get("depth_level", 0))
        quality     = int(p.get("quality", -8))
        cmap = _ECCO_CMAP.get(variable, "cividis")

        # Try LLC2160 (HTTPS) first, fall back to ECCO LLC4320 (Pelican)
        _progress(f"Animation: ocean {variable} depth={depth_level} q={quality} mode={render_mode}")
        db, _db_src = load_dyamond_llc2160(variable)
        _progress(f"  source={_db_src}")
        all_ts = list(db.getTimesteps())
        ts = _subsample(all_ts, t_start, t_end, cap=n_frames)
        _progress(f"Generating {len(ts)} animation frames")

        vmin_g = vmax_g = None
        src_name = "ECCO LLC4320" if _db_src == "ecco_fallback" else "LLC2160"
        for i, t in enumerate(ts):
            _progress(f"Frame {i + 1}/{len(ts)} t={t}")
            if _db_src == "ecco_fallback":
                raw = db.db.read(time=int(t), z=[depth_level, depth_level + 1], quality=quality)
            else:
                raw = db.read(time=int(t), z=[depth_level, depth_level + 1], quality=quality)
            arr = np.squeeze(np.array(raw, dtype=np.float32))
            if arr.ndim == 3:
                arr = arr.mean(axis=0)
            if render_mode == "globe" and i == 0:
                fin = arr[np.isfinite(arr)]
                if fin.size > 0:
                    vmin_g = float(np.percentile(fin, 2))
                    vmax_g = float(np.percentile(fin, 98))
            meters = DEPTH_LEVELS_METERS[depth_level] if 0 <= depth_level < len(DEPTH_LEVELS_METERS) else 0
            # Both LLC2160 (primary) and LLC4320 (fallback) share the ECCO
            # hourly calendar used by the frontend (dataset-time.ts ECCO_TIME),
            # so always emit calendar dates rather than raw timestep indices.
            t_label = _ecco_date(t)
            label = f"{src_name} {variable} | L{depth_level} ({meters:.1f} m) | {t_label}"
            png = _frame_png(arr, cmap, label, vmin_g, vmax_g, render_mode == "globe", llc=True)
            _emit_frame(i, len(ts), t_label, png)

    elif dataset == "cmip6":
        from analysis.core.utils import load_cmip6, cmip6_field_name
        variable = p.get("variable", "tas")
        model    = p.get("model", "ACCESS-CM2")
        scenario = p.get("scenario", "historical")
        t_start  = int(p.get("t_start", 0))
        t_end    = int(p.get("t_end", 200))
        cmap = _CMIP6_CMAP.get(variable, "viridis")

        _progress(f"Animation: CMIP6 {variable} {model} {scenario} mode={render_mode}")
        db = load_cmip6()
        field = cmip6_field_name(variable, model, scenario)
        all_ts = list(db.getTimesteps())
        ts = _subsample(all_ts, t_start, t_end, cap=n_frames)
        _progress(f"Generating {len(ts)} animation frames")

        vmin_g = vmax_g = None
        for i, t in enumerate(ts):
            _progress(f"Frame {i + 1}/{len(ts)} t={t}")
            arr = np.squeeze(np.array(db.read(field=field, time=int(t)), dtype=np.float32))
            if render_mode == "globe" and i == 0:
                # Compute locked colorscale from finite values only (CMIP6 has ~69% ocean NaN)
                fin = arr[np.isfinite(arr)]
                if fin.size > 0:
                    vmin_g = float(np.percentile(fin, 2))
                    vmax_g = float(np.percentile(fin, 98))
            label = f"CMIP6 {variable} | {model} | {scenario} | {_cmip6_date(t)}"
            png = _frame_png(arr, cmap, label, vmin_g, vmax_g, render_mode == "globe")
            _emit_frame(i, len(ts), _cmip6_date(t), png)

    return b""


# ─────────────────────────────────────────────────────────────────────────────
# DMD / TUCKER ANALYSIS VIEWER
# ─────────────────────────────────────────────────────────────────────────────

def query_dmd(p):
    """
    Render DMD mode reconstructions or Tucker spatial modes from .npz files.

    tech         : "dmd" | "mrdmd" | "tucker"
    task         : "wind" | "ocean" | "coupled"
    mode_idx     : int (for DMD: 0..19, for tucker: 0..4, for mrdmd: node 0..14)
    n_frames     : number of animation frames
    show_static  : if true, emit one frame per mode (static gallery)
    """
    NPZ_ROOT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")), "outputs")
    tech     = p.get("tech", "dmd")
    task     = p.get("task", "wind")
    mode_idx = int(p.get("mode_idx", 0))
    n_frames = min(int(p.get("n_frames", 30)), 60)

    # ── DMD wind ──────────────────────────────────────────────────────────────
    if tech == "dmd" and task == "wind":
        _progress("Loading task1_wind_patterns.npz")
        d = np.load(f"{NPZ_ROOT}/task1_wind_patterns.npz")
        modes_r = d["dmd_modes_real"]                   # (259200, 20)
        modes_i = d["dmd_modes_imag"]
        eigs_r  = d["eigenvalues_real"]                 # (20,)
        eigs_i  = d["eigenvalues_imag"]
        dyn_r   = d["dynamics_real"]                    # (20, 80)
        dyn_i   = d["dynamics_imag"]
        n_space = modes_r.shape[0]

        viz = p.get("viz", "animation")

        # ── Eigenvalue + mode-shapes overview (single static image) ──────────
        if viz == "eigenvalues":
            _progress("Rendering eigenvalue overview image")
            png = _render_dmd_overview(modes_r, modes_i, eigs_r, eigs_i, dyn_r, dyn_i, n_space)
            _emit_event({"type": "image", "dataUrl": _png_data_url(png)})
            return b""

        # ── Full reconstruction animation ────────────────────────────────────
        if viz == "reconstruction":
            n_modes_use = min(int(p.get("n_modes_use", modes_r.shape[1])), modes_r.shape[1])
            T = dyn_r.shape[1]
            ts = np.round(np.linspace(0, T - 1, n_frames)).astype(int)
            _progress(f"Computing reconstruction: {n_modes_use} modes, {len(ts)} frames (T={T})")
            modes_c = modes_r[:, :n_modes_use] + 1j * modes_i[:, :n_modes_use]
            dyn_c   = dyn_r[:n_modes_use, :] + 1j * dyn_i[:n_modes_use, :]
            X_recon = np.real(modes_c @ dyn_c).astype(np.float32)  # (n_space, T)
            half_r = n_space // 2
            sampled = X_recon[:, ts]
            speeds_r = np.sqrt(sampled[:half_r, :] ** 2 + sampled[half_r:, :] ** 2)
            finite_r = speeds_r.ravel()[np.isfinite(speeds_r.ravel())]
            global_vmax = float(np.percentile(finite_r, 99)) if finite_r.size > 0 else 1.0
            for i, t in enumerate(ts):
                x_t = X_recon[:, int(t)]
                date_str = _dyamond_date(int(t))
                title = f"DYAMOND wind | DMD reconstruction ({n_modes_use} modes) | face 0 · z=50 | {date_str}"
                png = _render_wind_panel(x_t, n_space, title=title, vmax=global_vmax)
                _emit_frame(i, len(ts), date_str, png)
            return b""

        m = min(mode_idx, modes_r.shape[1] - 1)
        eig = complex(eigs_r[m], eigs_i[m])
        mag = float(abs(eig))
        angle = float(np.angle(eig))
        dt_min = 30.0  # DYAMOND timestep in minutes
        period_str = _period_human(angle, dt_min=dt_min)

        # Number of steps to show: 1 full oscillation if period is short enough
        if abs(angle) > 1e-8:
            period_steps = 2 * np.pi / abs(angle)
        else:
            period_steps = float("inf")
        if np.isfinite(period_steps) and period_steps < n_frames * 5:
            t_values = np.linspace(0, period_steps, n_frames, endpoint=False)
        else:
            t_values = np.arange(n_frames, dtype=float)

        mode_c = modes_r[:, m] + 1j * modes_i[:, m]   # (259200,)
        amp0   = complex(dyn_r[m, 0], dyn_i[m, 0])

        title_base = (
            f"DYAMOND wind · face 0 · z=50 | DMD Mode {m} | "
            f"|λ|={mag:.3f} | T={period_str}"
        )
        _progress(f"{title_base} | {n_frames} frames")

        # Batch all frames first to lock colorscale across the whole animation
        phi = mode_c * amp0
        eig_powers = np.array([eig ** t for t in t_values])
        X_all = np.real(phi[np.newaxis, :] * eig_powers[:, np.newaxis]).astype(np.float32)
        half_m = n_space // 2
        speeds_m = np.sqrt(X_all[:, :half_m] ** 2 + X_all[:, half_m:] ** 2)
        finite_m = speeds_m.ravel()[np.isfinite(speeds_m.ravel())]
        global_vmax = float(np.percentile(finite_m, 99)) if finite_m.size > 0 else 1.0

        for i, (t, x_t) in enumerate(zip(t_values, X_all)):
            # t is in DYAMOND step units (30 min each, epoch 2020-Jan-20)
            date_str = _dyamond_date(int(round(t)))
            title = f"{title_base} | {date_str}"
            png   = _render_wind_panel(x_t, n_space, title=title, vmax=global_vmax)
            _emit_frame(i, n_frames, date_str, png)

    # ── mrDMD wind nodes ───────────────────────────────────────────────────────
    elif tech == "mrdmd" and task == "wind":
        _progress("Loading task1_mrdmd.npz")
        d = np.load(f"{NPZ_ROOT}/task1_mrdmd.npz")
        node = min(mode_idx, 14)  # 15 nodes (0..14)

        # Gather all nodes or just the selected node
        nodes_to_show = list(range(15)) if p.get("show_all") else [node]
        n_frames_actual = len(nodes_to_show)

        for fi, ni in enumerate(nodes_to_show):
            modes_r = d[f"mrdmd_{ni}_slow_modes_real"]  # (259200, n_slow)
            modes_i = d[f"mrdmd_{ni}_slow_modes_imag"]
            eigs_r  = d[f"mrdmd_{ni}_slow_eigenvalues_real"]
            eigs_i  = d[f"mrdmd_{ni}_slow_eigenvalues_imag"]
            amps_r  = d[f"mrdmd_{ni}_slow_amplitudes_real"]
            amps_i  = d[f"mrdmd_{ni}_slow_amplitudes_imag"]
            level   = int(d[f"mrdmd_{ni}_level"])
            t_start_node = int(d[f"mrdmd_{ni}_time_start"])
            t_end_node   = int(d[f"mrdmd_{ni}_time_end"])
            n_space = int(d[f"mrdmd_{ni}_n_space"])

            # Sum all slow modes at t=0 (snapshot of the slow background)
            x_slow = np.zeros(n_space, dtype=np.float32)
            for j in range(modes_r.shape[1]):
                eig = complex(eigs_r[j], eigs_i[j])
                amp = complex(amps_r[j], amps_i[j])
                mode_c = modes_r[:, j] + 1j * modes_i[:, j]
                x_slow += (mode_c * amp).real.astype(np.float32)

            t_start_date = _dyamond_date(t_start_node)
            t_end_date   = _dyamond_date(t_end_node)
            title = (
                f"mrDMD node {ni} | level {level} | "
                f"{t_start_date} → {t_end_date} | slow background"
            )
            _progress(f"Rendering {title}")
            png = _render_wind_panel(x_slow, n_space, title=title)
            _emit_frame(fi, n_frames_actual, f"node {ni}", png)

    # ── Tucker HOSVD — ocean spatial modes ───────────────────────────────────
    elif tech == "tucker" and task == "ocean":
        _progress("Loading task3_ocean_circulation.npz")
        d = np.load(f"{NPZ_ROOT}/task3_ocean_circulation.npz")
        spatial = d["spatial_modes_3d"]        # (5, 1620, 2160)
        n_modes = spatial.shape[0]
        _progress(f"Tucker ocean: {n_modes} spatial modes ({spatial.shape[1]}x{spatial.shape[2]})")

        if mode_idx < 0:
            # Gallery: all modes as frames
            for i in range(n_modes):
                arr = spatial[i]               # (1620, 2160)
                png = _render_anim_frame(arr, "RdBu_r",
                                         title=f"Tucker ocean spatial mode {i}")
                _emit_frame(i, n_modes, f"mode {i}", png)
        else:
            # Animate: Tucker temporal reconstruction
            factor_time = d.get("factor_time_theta")  # (T, n_modes)
            if factor_time is None:
                factor_time = np.eye(min(n_frames, 10), spatial.shape[0], dtype=np.float32)
            n_t = factor_time.shape[0]
            frames_to_use = min(n_frames, n_t)
            _progress(f"Animating {frames_to_use} Tucker timesteps")

            m = min(mode_idx, n_modes - 1)
            # Pre-compute all frames to lock colorscale — Tucker weights change sign
            # so per-frame vmin/vmax would make the colorbar jump.
            frame_arrays = []
            for i in range(frames_to_use):
                w = float(factor_time[i, m] if factor_time.shape[1] > m else 1.0)
                frame_arrays.append(spatial[m] * w)
            all_vals = np.concatenate([a.ravel() for a in frame_arrays])
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size > 0:
                global_vmin = float(np.percentile(finite_vals, 2))
                global_vmax = float(np.percentile(finite_vals, 98))
                if global_vmax <= global_vmin:
                    global_vmax = global_vmin + 1e-6
            else:
                global_vmin, global_vmax = -1.0, 1.0

            for i, arr in enumerate(frame_arrays):
                # Tucker time indices map to ECCO timesteps (dt=1 h, epoch 2011-Sep-13)
                date_str = _ecco_date(i)
                title = f"Tucker ocean mode {m} | {date_str}"
                png = _render_anim_frame(arr, "RdBu_r", title=title,
                                         vmin=global_vmin, vmax=global_vmax)
                _emit_frame(i, frames_to_use, date_str, png)

    # ── Coupled DMD ────────────────────────────────────────────────────────────
    elif tech == "dmd" and task == "coupled":
        _progress("Loading task2_atm_ocean.npz")
        d = np.load(f"{NPZ_ROOT}/task2_atm_ocean.npz")
        atm_r  = d["atm_modes_real"]            # (64800, 10)
        atm_i  = d["atm_modes_imag"]
        eigs_r = d["eigenvalues_real"]           # (10,)
        eigs_i = d["eigenvalues_imag"]
        # Coupled mode amplitudes assumed unit at t=0 (normalised)
        n_space = atm_r.shape[0]                 # 64800 = 2 × 180 × 180
        n_modes = atm_r.shape[1]
        m = min(mode_idx, n_modes - 1)

        eig = complex(eigs_r[m], eigs_i[m])
        mag = float(abs(eig))
        angle = float(np.angle(eig))
        dt_min = 30.0  # DYAMOND timestep in minutes
        period_str = _period_human(angle, dt_min=dt_min)

        mode_c = atm_r[:, m] + 1j * atm_i[:, m]
        t_values = np.arange(n_frames, dtype=float)

        title_base = (
            f"Coupled DMD mode {m} (atm) | |λ|={mag:.3f} | T={period_str}"
        )
        _progress(f"{title_base}")

        # Pre-compute all frames to lock the colorscale
        X_all = np.array([
            (mode_c * (eig ** t)).real.astype(np.float32) for t in t_values
        ])
        half_c = n_space // 2
        speeds_c = np.sqrt(X_all[:, :half_c] ** 2 + X_all[:, half_c:] ** 2)
        finite_c = speeds_c.ravel()[np.isfinite(speeds_c.ravel())]
        global_vmax = float(np.percentile(finite_c, 99)) if finite_c.size > 0 else 1.0

        for i, (t, x_t) in enumerate(zip(t_values, X_all)):
            # t is in DYAMOND step units (30 min, epoch 2020-Jan-20)
            date_str = _dyamond_date(int(round(t)))
            png = _render_wind_panel(x_t, n_space,
                                     title=f"{title_base} | {date_str}",
                                     vmax=global_vmax)
            _emit_frame(i, n_frames, date_str, png)

    else:
        _progress(f"Warning: unknown tech={tech} task={task}, emitting empty set")

    return b""


# ─────────────────────────────────────────────────────────────────────────────
# LIVE DMD — fetch snapshots on-demand, run exact DMD, stream mode animations
# ─────────────────────────────────────────────────────────────────────────────
def query_dmd_live(p):
    """
    Fetch N snapshots from any supported dataset using user-specified parameters,
    run exact DMD (truncated SVD, r modes), and emit image_frame events for the
    requested mode.

    Supported sources: "dyamond" (single variable, single face) and "ecco".

    The rendered frames are plain heatmaps (no quiver) so any scalar variable
    works without special U+V stacking.
    """
    source      = p.get("source", "dyamond")
    tech        = p.get("tech", "dmd")   # "dmd" | "rdmd" | "hosvd"
    n_frames    = min(int(p.get("n_frames", 20)), 30)
    n_modes_req = int(p.get("n_modes", 8))   # clamped after n_snapshots is known

    snapshot_shape = None
    cols      = []
    ts_real   = []
    n_ts_full = 0
    n_snapshots = 4   # resolved per-source below

    # ── Per-frame disk cache ───────────────────────────────────────────────
    # Fetched arrays are cached as .npy files keyed by source/variable/quality/
    # depth/timestep.  Cache hits load from disk in ~1 ms vs ~150 ms from NSDF.
    # The cache persists across runs so repeat analyses on the same window are instant.
    import os as _os
    _FRAME_CACHE = _os.path.join(_os.getcwd(), "outputs", "frame_cache")
    _os.makedirs(_FRAME_CACHE, exist_ok=True)

    def _fc_get(key):
        p_ = _os.path.join(_FRAME_CACHE, key + ".npy")
        if _os.path.exists(p_):
            try:
                return np.load(p_)
            except Exception:
                return None
        return None

    def _fc_put(key, arr):
        try:
            np.save(_os.path.join(_FRAME_CACHE, key + ".npy"), arr)
        except Exception:
            pass  # never fail a run due to cache write

    # ── Parallel snapshot helpers ──────────────────────────────────────────
    _N_WORKERS = 32          # 32 concurrent HTTP connections to NSDF
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _clean_arr(arr):
        arr2d = np.squeeze(np.array(arr, dtype=np.float32))
        if arr2d.ndim > 2:
            arr2d = arr2d.mean(axis=0)
        return np.nan_to_num(arr2d, nan=0.0).astype(np.float32)

    # ── Fetch snapshots ────────────────────────────────────────────────────
    if source == "dyamond":
        from analysis.core.utils import load_dyamond_geos
        variable = p.get("variable", "u")
        face     = int(p.get("face", 0))
        t_start  = int(p.get("t_start", 0))
        t_end    = int(p.get("t_end", 160))
        z_level  = int(p.get("z_level", 50))
        quality  = int(p.get("quality", -6))
        # Clamp: coarser quality = smaller array = faster fetch (DMD finds modes regardless)
        quality = min(quality, -5)
        cmap     = _DYAMOND_CMAP.get(variable, "RdBu_r")
        dt_min   = 30.0

        _progress(f"Loading DYAMOND GEOS {variable} face={face}")
        db = load_dyamond_geos(variable, face)
        all_ts      = list(db.getTimesteps())
        n_ts_full   = len(all_ts)
        ts_in_range = [t for t in all_ts if t_start <= t <= t_end]
        _n_before = len(ts_in_range)
        ts_in_range = _cap_uniform(ts_in_range, _DMD_SNAPSHOT_CAP)
        if len(ts_in_range) < _n_before:
            _progress(f"Subsampled {_n_before} → {len(ts_in_range)} snapshots (uniform) for DMD")

        _dyn_prefix = f"dyn_{variable}_f{face}_z{z_level}_q{quality}"

        def _fetch_dyn(t):
            key = f"{_dyn_prefix}_t{t}"
            cached = _fc_get(key)
            if cached is not None:
                return t, cached
            arr = _clean_arr(_dyamond_read(db, int(t), z_level, quality))
            _fc_put(key, arr)
            return t, arr

        ts_real = _probe_and_plan(_fetch_dyn, ts_in_range, n_ts_full, source="dyamond")
        n_snapshots = len(ts_real)

        _progress(f"Fetching {n_snapshots} steps (z={z_level} q={quality}, {_N_WORKERS} workers, cache={_FRAME_CACHE})")
        results = [None] * n_snapshots
        done_count = 0
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            fmap = {pool.submit(_fetch_dyn, t): i for i, t in enumerate(ts_real)}
            for fut in as_completed(fmap):
                i = fmap[fut]
                t_val, arr2d = fut.result()
                results[i] = arr2d
                done_count += 1
                _progress(f"  {done_count}/{n_snapshots}  ({_dyamond_date(int(t_val))})")
        cols = [r.ravel() for r in results]
        snapshot_shape = results[0].shape

        date_fn  = _dyamond_date
        data_desc = (
            f"DYAMOND GEOS · {variable} · face {face} · z={z_level} · q={quality} · "
            f"{_dyamond_date(ts_real[0])} → {_dyamond_date(ts_real[-1])}"
        )

    elif source == "ecco":
        from analysis.core.utils import load_dyamond_llc2160
        variable    = p.get("variable", "theta")
        t_start     = int(p.get("t_start", 0))
        t_end       = int(p.get("t_end", 100))
        depth_level = int(p.get("depth_level", 0))
        quality     = int(p.get("quality", -8))
        quality = min(quality, -8)   # keep arrays small
        cmap        = _ECCO_CMAP.get(variable, "cividis")
        dt_min      = 60.0

        # Try LLC2160 (HTTPS, reliable) then fall back to ECCO LLC4320 (Pelican)
        _progress(f"Loading ocean dataset (LLC2160 → ECCO fallback) for {variable}")
        db, _db_src = load_dyamond_llc2160(variable)
        _progress(f"  source={_db_src}")
        all_ts      = list(db.getTimesteps())
        n_ts_full   = len(all_ts)
        ts_in_range = [t for t in all_ts if t_start <= t <= t_end]
        _n_before = len(ts_in_range)
        ts_in_range = _cap_uniform(ts_in_range, _DMD_SNAPSHOT_CAP)
        if len(ts_in_range) < _n_before:
            _progress(f"Subsampled {_n_before} → {len(ts_in_range)} snapshots (uniform) for DMD")
        meters = DEPTH_LEVELS_METERS[depth_level] if 0 <= depth_level < len(DEPTH_LEVELS_METERS) else 0.0

        def _ocean_read_arr(t, q):
            if _db_src == "ecco_fallback":
                return db.db.read(time=int(t), z=[depth_level, depth_level + 1], quality=q)
            return db.read(time=int(t), z=[depth_level, depth_level + 1], quality=q)

        _ecco_prefix = f"ecco_{variable}_d{depth_level}_q{quality}"

        def _fetch_ecco(t):
            key = f"{_ecco_prefix}_t{t}"
            cached = _fc_get(key)
            if cached is not None:
                return t, cached, quality
            last_exc = None
            for q in range(quality, 1):
                try:
                    arr = _clean_arr(_ocean_read_arr(t, q))
                    _fc_put(key, arr)
                    return t, arr, q
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(f"ocean t={t} failed all qualities {quality}..0") from last_exc

        ts_real = _probe_and_plan(_fetch_ecco, ts_in_range, n_ts_full, source="ecco")
        n_snapshots = len(ts_real)

        _progress(f"Fetching {n_snapshots} steps (depth={depth_level} q={quality}, {_N_WORKERS} workers, cache={_FRAME_CACHE})")
        results = [None] * n_snapshots
        done_count = 0
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            fmap = {pool.submit(_fetch_ecco, t): i for i, t in enumerate(ts_real)}
            for fut in as_completed(fmap):
                i = fmap[fut]
                t_val, arr2d, used_q = fut.result()
                results[i] = arr2d
                done_count += 1
                t_label = _ecco_date(int(t_val))
                _progress(f"  {done_count}/{n_snapshots}  ({t_label})"
                          + (f"  [q={used_q}]" if used_q != quality else ""))
        cols = [r.ravel() for r in results]
        snapshot_shape = results[0].shape

        src_name = "ECCO LLC4320" if _db_src == "ecco_fallback" else "LLC2160"
        date_fn  = _ecco_date
        data_desc = (
            f"{src_name} · {variable} · L{depth_level} ({meters:.0f} m) · q={quality} · "
            f"{date_fn(ts_real[0])} → {date_fn(ts_real[-1])}"
        )

    elif source == "cmip6":
        from analysis.core.utils import load_cmip6, cmip6_field_name
        variable = p.get("variable", "tas")
        model    = p.get("model", "ACCESS-CM2")
        scenario = p.get("scenario", "historical")
        t_start  = int(p.get("t_start", 0))
        t_end    = int(p.get("t_end", 200))
        quality  = -4          # coarse but fast; CMIP6 has no deeper pyramid
        cmap     = _CMIP6_CMAP.get(variable, "viridis")
        dt_min   = 1440.0      # CMIP6 = daily (1440 min / step)

        _progress(f"Loading CMIP6 {variable} {model} {scenario}")
        db  = load_cmip6()
        fld = cmip6_field_name(variable, model, scenario)
        all_ts      = list(db.getTimesteps())
        n_ts_full   = len(all_ts)
        ts_in_range = [t for t in all_ts if t_start <= t <= t_end]
        _n_before = len(ts_in_range)
        ts_in_range = _cap_uniform(ts_in_range, _DMD_SNAPSHOT_CAP)
        if len(ts_in_range) < _n_before:
            _progress(f"Subsampled {_n_before} → {len(ts_in_range)} snapshots (uniform) for DMD")

        _cmip6_prefix = f"cmip6_{variable}_{model}_{scenario}_q{quality}"

        def _fetch_cmip6(t):
            key = f"{_cmip6_prefix}_t{t}"
            cached = _fc_get(key)
            if cached is not None:
                return t, cached
            arr = np.squeeze(np.array(db.read(field=fld, time=int(t), quality=quality),
                                      dtype=np.float32))
            fin = arr[np.isfinite(arr)]
            fill = float(fin.mean()) if fin.size > 0 else 0.0
            clean = np.where(np.isnan(arr), fill, arr).astype(np.float32)
            _fc_put(key, clean)
            return t, clean

        ts_real = _probe_and_plan(_fetch_cmip6, ts_in_range, n_ts_full, source="cmip6")
        n_snapshots = len(ts_real)

        _progress(f"Fetching {n_snapshots} CMIP6 steps (q={quality}, {_N_WORKERS} workers, cache={_FRAME_CACHE})")
        results = [None] * n_snapshots
        done_count = 0
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            fmap = {pool.submit(_fetch_cmip6, t): i for i, t in enumerate(ts_real)}
            for fut in as_completed(fmap):
                i = fmap[fut]
                t_val, arr2d = fut.result()
                results[i] = arr2d
                done_count += 1
                _progress(f"  {done_count}/{n_snapshots}  ({_cmip6_date(int(t_val))})")
        cols = [r.ravel() for r in results]
        snapshot_shape = results[0].shape

        date_fn  = _cmip6_date
        data_desc = (
            f"CMIP6 · {variable} · {model} · {scenario} · "
            f"{_cmip6_date(ts_real[0])} → {_cmip6_date(ts_real[-1])}"
        )

    else:
        raise ValueError(f"Unknown source for live DMD: {source!r}")

    if len(cols) < 3:
        raise RuntimeError("Need at least 3 snapshots to run DMD.")

    # Clamp n_modes now that we know the actual snapshot count
    n_modes = min(n_modes_req, len(cols) - 2)

    # Effective snapshot spacing in REAL minutes. _cap_uniform spaces the kept
    # timesteps `stride` native-steps apart, so the true dt between consecutive
    # DMD snapshots is stride × dt_min — not dt_min. Every period/frequency label
    # must use dt_eff_min or it is wrong by the stride factor.
    _ts_arr         = np.asarray(ts_real, dtype=float)
    step_steps      = float(np.median(np.diff(_ts_arr))) if _ts_arr.size >= 2 else 1.0
    dt_eff_min      = max(step_steps, 1.0) * dt_min
    nyquist_min     = 2.0 * dt_eff_min
    window_span_min = (float(_ts_arr[-1] - _ts_arr[0]) * dt_min) if _ts_arr.size >= 2 else 0.0

    # ── Data size summary (reported again at Tucker stage for the metrics strip) ─
    _H, _W      = snapshot_shape
    frame_kb    = _H * _W * 4 / 1024          # float32, one depth slice
    fetched_mb  = len(cols) * frame_kb / 1024
    full_est_mb = n_ts_full * frame_kb / 1024 if n_ts_full > 0 else 0
    coverage_pct = len(cols) / max(n_ts_full, 1) * 100

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Tucker HOSVD compression (always runs before any analysis)
    #
    # Reshape the flat snapshot matrix X (n_space × T) into a 3-D tensor
    # (H × W × T), then run Tucker HOSVD to obtain:
    #   U_h  (H  × rs)   — spatial row factor
    #   U_w  (W  × rs)   — spatial column factor
    #   U_t  (T  × rt)   — temporal factor
    #   G    (rs × rs × rt) — compressed core tensor
    #
    # Compressed time-series matrix X_tucker (rs² × T) = G_flat @ U_t.T
    # is what DMD / mrDMD operates on — far smaller than the raw H*W × T matrix.
    # Spatial reconstruction:  U_h @ vec.reshape(rs,rs) @ U_w.T → (H,W)
    # ════════════════════════════════════════════════════════════════════════
    import torch
    import tensorly as tl
    from tensorly.decomposition import tucker as tl_tucker

    # Build flat snapshot matrix — float32 is sufficient for Tucker compression
    X = np.column_stack(cols).astype(np.float32)   # (n_space, T)
    T = X.shape[1]
    H, W = snapshot_shape

    # Clean any residual NaN/Inf — ocean masks and land-fill values can slip
    # through per-frame cleaning (coastal edges, fill values outside grid).
    # Must happen before Tucker: cuSOLVER rejects matrices containing NaN
    # (CUSOLVER_STATUS_INVALID_VALUE).
    _n_bad = int(np.sum(~np.isfinite(X)))
    if _n_bad > 0:
        _progress(f"  Cleaning {_n_bad} non-finite values in X ({_n_bad / X.size * 100:.3f}% of elements)")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove the per-pixel temporal mean so Tucker, reconstruction error, the
    # rank sweep, and DMD all operate on the fluctuations (anomalies) the modes
    # actually model. Otherwise the near-rank-1 mean state dominates the energy
    # and recon error collapses to ~0% regardless of how well dynamics are captured.
    X = X - X.mean(axis=1, keepdims=True)

    # Tucker ranks — spatial rank raised to 32 max (GPU handles it); temporal = user's n_modes
    rs = max(4, min(n_modes * 4, 32, H, W))   # spatial rank (was 16; richer spatial modes)
    rt = min(n_modes, T - 1)                   # temporal rank (= user's mode count)

    # ── GPU-accelerated Tucker HOSVD ─────────────────────────────────────────
    # Use PyTorch CUDA backend when available (RTX 4090 → typically 10-30× faster
    # than CPU numpy for tensors in the 400×500×500 range).
    _use_gpu = torch.cuda.is_available()
    _device  = "cuda" if _use_gpu else "cpu"

    # ── Spatial downsampling when tensor exceeds VRAM budget ─────────────────
    # Tucker is a compression step — modest spatial downsampling preserves the
    # dominant spatiotemporal modes and is invisible in the final mode animations.
    _VRAM_BUDGET  = 18 * 1024**3    # 18 GB — leave 6 GB for SVD workspace / activations
    _CPU_BUDGET   =  8 * 1024**3    # 8 GB  — conservative limit for CPU numpy Tucker
    _raw_bytes    = H * W * T * 4   # float32
    _budget       = _VRAM_BUDGET if _use_gpu else _CPU_BUDGET
    if _raw_bytes > _budget:
        import math as _math
        _stride = max(2, _math.ceil((_raw_bytes / _budget) ** 0.5))
        H_ds    = max(rs + 1, H // _stride)
        W_ds    = max(rs + 1, W // _stride)
        _progress(
            f"  Tensor {_raw_bytes/1e9:.1f} GB > budget {_budget/1e9:.0f} GB — "
            f"spatial stride {_stride}× → ({H}×{W}) → ({H_ds}×{W_ds})"
        )
        # Downsample the spatial dimensions, keep all timesteps
        X_ds   = X.reshape(H, W, T)[::_stride, ::_stride, :]
        H, W   = H_ds, W_ds
        snapshot_shape = (H, W)
        X      = np.ascontiguousarray(X_ds.reshape(H * W, T))
        # Recompute spatial rank for new (possibly smaller) spatial dims
        rs     = max(4, min(n_modes * 4, 32, H, W))
        _progress(f"  Adjusted Tucker rank: [{rs}, {rs}, {rt}]")

    # Rank sweep uses max spatial rank and pure HOSVD (n_iter_max=0) so factors are
    # ordered by energy and truncation gives exact sub-rank errors.
    # Normal runs use n_iter_max=10 (ALS polish) at the user-specified rank.
    _tucker_is_sweep = (tech == "rank_sweep")
    if _tucker_is_sweep:
        rs = min(32, H, W)          # maximum spatial rank for full sweep
        rt = min(20, T - 1)         # wide temporal range for temporal sweep
    _tucker_n_iter = 0 if _tucker_is_sweep else 10

    _progress(f"Stage 3 — Tucker HOSVD ({_device.upper()}): ({H}×{W}×{T}) → rank [{rs},{rs},{rt}]"
              + (" [HOSVD ordered, n_iter=0]" if _tucker_is_sweep else ""))

    def _run_tucker_gpu():
        tl.set_backend("pytorch")
        try:
            X_t = torch.tensor(X.reshape(H, W, T), device="cuda")
            # Secondary NaN guard: cuSOLVER silently fails on NaN even if numpy
            # looks clean, due to floating-point edge cases on GPU transfer.
            _bad = (~torch.isfinite(X_t)).sum().item()
            if _bad > 0:
                _progress(f"  GPU tensor: {int(_bad)} non-finite after transfer — zeroing")
                X_t = torch.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
            c_t, f_t = tl_tucker(X_t, rank=[rs, rs, rt], init="svd", n_iter_max=_tucker_n_iter)
            c  = c_t.cpu().numpy().astype(np.float64)
            fs = [f.cpu().numpy().astype(np.float64) for f in f_t]
            del X_t, c_t, f_t
            torch.cuda.empty_cache()
            return c, fs
        finally:
            tl.set_backend("numpy")

    def _run_tucker_cpu():
        tl.set_backend("numpy")
        X_np = X.reshape(H, W, T).astype(np.float64)
        c_t, f_t = tl_tucker(X_np, rank=[rs, rs, rt], init="svd", n_iter_max=_tucker_n_iter)
        return c_t.astype(np.float64), [f.astype(np.float64) for f in f_t]

    if _use_gpu:
        try:
            core, factors = _run_tucker_gpu()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as _gpu_err:
            _estr = str(_gpu_err).lower()
            if "out of memory" in _estr:
                _progress(f"  GPU OOM — falling back to CPU Tucker")
                torch.cuda.empty_cache()
                core, factors = _run_tucker_cpu()
            elif "cusolver" in _estr or "invalid_value" in _estr or "nan" in _estr:
                # cuSOLVER rejects NaN/Inf even after our cleaning pass.
                # Try switching to the MAGMA backend first; fall back to CPU.
                _progress(f"  cuSOLVER error — trying MAGMA linalg backend")
                torch.cuda.empty_cache()
                try:
                    torch.backends.cuda.preferred_linalg_library("magma")
                    core, factors = _run_tucker_gpu()
                except Exception as _magma_err:
                    _progress(f"  MAGMA also failed ({_magma_err!s:.80}) — falling back to CPU Tucker")
                    core, factors = _run_tucker_cpu()
                finally:
                    torch.backends.cuda.preferred_linalg_library("default")
            else:
                raise
    else:
        core, factors = _run_tucker_cpu()
    U_h, U_w, U_t = factors[0], factors[1], factors[2]   # (H,rs) (W,rs) (T,rt)

    # Compressed snapshot matrix in Tucker latent space
    G_flat   = core.reshape(rs * rs, rt)     # (rs², rt)
    X_tucker = G_flat @ U_t.T               # (rs², T)  ← DMD runs on this

    raw_elems  = H * W * T
    comp_elems = rs * rs * rt + H * rs + W * rs + T * rt
    ratio      = raw_elems / max(comp_elems, 1)
    tucker_kb  = comp_elems * 8 / 1024

    # Reconstruction error via Pythagorean theorem — avoids rebuilding the full
    # tensor (which would allocate another ~1 GB just for the error metric).
    # For Tucker: ||X - X_rec||_F = sqrt(||X||_F² - ||core||_F²)
    # X is float32 (H*W, T) — Frobenius norm is identical to float64; no reshape needed.
    x_norm    = float(np.linalg.norm(X))
    core_norm = float(np.linalg.norm(core))
    recon_err = float(np.sqrt(max(0.0, x_norm ** 2 - core_norm ** 2)) / max(x_norm, 1e-10))

    _progress(
        f"Tucker done: {ratio:.0f}x compression  "
        f"({fetched_mb:.1f} MB  →  {tucker_kb:.1f} KB latent)  "
        f"anomaly recon error {recon_err:.2%}"
    )

    # Coverage / sampling diagnostics for the resolved-band readout.
    undersampled = bool(coverage_pct < 5.0)

    # Emit as a dedicated event so the frontend can display a persistent metrics strip
    _emit_event({
        "type": "tucker_stats",
        "compression_ratio": round(float(ratio), 1),
        "reconstruction_error": round(float(recon_err), 4),
        "fetched_mb": round(float(fetched_mb), 2),
        "tucker_kb": round(float(tucker_kb), 1),
        "coverage_pct": round(float(coverage_pct), 2),
        "n_samples": int(T),
        "n_ts_full": int(n_ts_full),
        "frame_h": int(H),
        "frame_w": int(W),
        "rank": [int(rs), int(rs), int(rt)],
        "dt_eff_min": round(float(dt_eff_min), 2),
        "nyquist_period_min": round(float(nyquist_min), 2),
        "window_span_min": round(float(window_span_min), 2),
        "undersampled": undersampled,
    })

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Rank sweep (tech == "rank_sweep")
    #
    # Since Tucker ran with n_iter_max=0 (HOSVD), the core slices are ordered
    # by energy along each mode axis.  Truncating core[:r, :r, :] gives the
    # exact Tucker error for spatial rank r (Pythagorean: ||X-X_r||² = ||X||² - ||core[:r,:r,:]||²).
    # Similarly for temporal: core[:, :, :rt].
    # ════════════════════════════════════════════════════════════════════════
    if _tucker_is_sweep:
        # Tucker ran with init="svd", n_iter_max=0 — this is pure HOSVD.
        # Mode factors are computed via SVD of each mode-k unfolding, so the
        # core tensor axes are ordered by energy (largest singular values first).
        # Truncating core[:r, :r, :rt] therefore gives the exact HOSVD error
        # at spatial rank r via the Pythagorean identity — no re-running Tucker.
        _progress(f"Stage 4 — HOSVD rank sweep  (core shape {list(core.shape)}, "
                  f"testing every integer rank 1..{rs} spatial, 1..{rt} temporal)")

        x_norm_sq = x_norm ** 2

        # Every integer rank — computing ||core[:r,:r,:]|| is a simple sum, ~µs each
        spatial_ranks  = list(range(1, rs + 1))
        temporal_ranks = list(range(1, rt + 1))

        # ── Spatial sweep: fix temporal at rt, vary spatial r 1..rs ─────
        _progress(f"  Spatial sweep: all {rs} ranks")
        spatial_errors = []
        for r in spatial_ranks:
            trunc_norm_sq = float(np.sum(core[:r, :r, :] ** 2))
            err = float(np.sqrt(max(0.0, x_norm_sq - trunc_norm_sq)) / max(x_norm, 1e-10))
            spatial_errors.append(err)

        # ── Temporal sweep: fix spatial at rs, vary temporal rt 1..rt ───
        _progress(f"  Temporal sweep: all {rt} ranks")
        temporal_errors = []
        for r_t in temporal_ranks:
            trunc_norm_sq = float(np.sum(core[:, :, :r_t] ** 2))
            err = float(np.sqrt(max(0.0, x_norm_sq - trunc_norm_sq)) / max(x_norm, 1e-10))
            temporal_errors.append(err)

        # ── Compression ratio at each rank (the advantage Tucker buys) ───
        _raw = H * W * T
        spatial_compression = [
            _raw / max(r * r * rt + H * r + W * r + T * rt, 1) for r in spatial_ranks
        ]
        temporal_compression = [
            _raw / max(rs * rs * r_t + H * rs + W * rs + T * r_t, 1) for r_t in temporal_ranks
        ]

        # ── Knee finder: max perpendicular distance from chord ───────────
        def _find_knee(ranks_arr, errors_arr):
            r = np.array(ranks_arr, dtype=float)
            e = np.array(errors_arr, dtype=float)
            if len(r) < 3:
                return ranks_arr[-1]
            # Normalise both axes to [0,1] then find max distance from chord
            r_n = (r - r[0]) / max(r[-1] - r[0], 1e-10)
            e_n = (e - e[-1]) / max(e[0] - e[-1], 1e-10)   # flip so curve goes down-right
            chord = np.array([r_n[-1] - r_n[0], e_n[-1] - e_n[0]])
            chord_len = np.linalg.norm(chord) + 1e-10
            # Signed perpendicular distances
            dists = np.abs((r_n - r_n[0]) * chord[1] - (e_n - e_n[0]) * chord[0]) / chord_len
            return ranks_arr[int(np.argmax(dists))]

        knee_s = _find_knee(spatial_ranks, spatial_errors)
        knee_t = _find_knee(temporal_ranks, temporal_errors)
        knee_s_err = spatial_errors[spatial_ranks.index(knee_s)] * 100
        knee_t_err = temporal_errors[temporal_ranks.index(knee_t)] * 100
        _progress(f"  Knee: spatial rank={knee_s} (error {knee_s_err:.2f}%)  "
                  f"temporal rank={knee_t} (error {knee_t_err:.2f}%)")

        # ── Emit raw data so frontend could render its own chart ─────────
        _emit_event({
            "type": "rank_sweep_result",
            "spatial_ranks":  spatial_ranks,
            "spatial_errors": [round(e, 5) for e in spatial_errors],
            "spatial_compression": [round(c, 1) for c in spatial_compression],
            "temporal_ranks": temporal_ranks,
            "temporal_errors": [round(e, 5) for e in temporal_errors],
            "temporal_compression": [round(c, 1) for c in temporal_compression],
            "knee_spatial": knee_s,
            "knee_temporal": knee_t,
            "data_desc": data_desc,
        })

        # ── Plot ─────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#c9d1d9", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            ax.grid(True, color="#21262d", linewidth=0.8)

        # — Spatial sweep —
        ax = axes[0]
        s_pct = [e * 100 for e in spatial_errors]
        ax.plot(spatial_ranks, s_pct, "o-", color="#58a6ff", linewidth=2, markersize=4, label="reconstruction error")
        ax.axvline(x=knee_s, color="#f78166", linewidth=1.5, linestyle="--", alpha=0.9)
        ax.annotate(
            f"knee  r={knee_s}\n{knee_s_err:.1f}% error",
            xy=(knee_s, knee_s_err),
            xytext=(knee_s + max(1, rs * 0.08), knee_s_err + max(1.0, s_pct[0] * 0.07)),
            color="#f78166", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#f78166", lw=0.8),
        )
        ax.set_xlabel("Spatial rank  rs", color="#c9d1d9", fontsize=10)
        ax.set_ylabel("Anomaly reconstruction error  (%)", color="#58a6ff", fontsize=10)
        ax.tick_params(axis="y", colors="#58a6ff")
        ax.set_title("Spatial rank: error ↓ vs compression ↑", color="#e6edf3", fontsize=11, pad=6)
        ax.set_ylim(bottom=0, top=max(s_pct) * 1.12)
        ax.set_xlim(left=0)
        axc = ax.twinx()
        axc.plot(spatial_ranks, spatial_compression, "s--", color="#d29922", lw=1.4, ms=3, alpha=0.9)
        axc.set_ylabel("Compression ratio  (×, log)", color="#d29922", fontsize=10)
        axc.set_yscale("log")
        axc.tick_params(axis="y", colors="#d29922", labelsize=8)
        axc.annotate(
            f"r={knee_s}: {spatial_compression[spatial_ranks.index(knee_s)]:.0f}× @ {knee_s_err:.1f}%",
            xy=(knee_s, spatial_compression[spatial_ranks.index(knee_s)]),
            xytext=(knee_s + max(1, rs * 0.1), spatial_compression[spatial_ranks.index(knee_s)]),
            color="#d29922", fontsize=8,
        )

        # — Temporal sweep —
        ax = axes[1]
        t_pct = [e * 100 for e in temporal_errors]
        ax.plot(temporal_ranks, t_pct, "o-", color="#3fb950", linewidth=2, markersize=4, label="reconstruction error")
        ax.axvline(x=knee_t, color="#f78166", linewidth=1.5, linestyle="--", alpha=0.9)
        ax.annotate(
            f"knee  r={knee_t}\n{knee_t_err:.1f}% error",
            xy=(knee_t, knee_t_err),
            xytext=(knee_t + max(0.3, rt * 0.08), knee_t_err + max(1.0, t_pct[0] * 0.07)),
            color="#f78166", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#f78166", lw=0.8),
        )
        ax.set_xlabel("Temporal rank  rt", color="#c9d1d9", fontsize=10)
        ax.set_ylabel("Anomaly reconstruction error  (%)", color="#3fb950", fontsize=10)
        ax.tick_params(axis="y", colors="#3fb950")
        ax.set_title("Temporal rank: error ↓ vs compression ↑", color="#e6edf3", fontsize=11, pad=6)
        ax.set_ylim(bottom=0, top=max(t_pct) * 1.12)
        ax.set_xlim(left=0)
        axc = ax.twinx()
        axc.plot(temporal_ranks, temporal_compression, "s--", color="#d29922", lw=1.4, ms=3, alpha=0.9)
        axc.set_ylabel("Compression ratio  (×, log)", color="#d29922", fontsize=10)
        axc.set_yscale("log")
        axc.tick_params(axis="y", colors="#d29922", labelsize=8)
        axc.annotate(
            f"r={knee_t}: {temporal_compression[temporal_ranks.index(knee_t)]:.0f}× @ {knee_t_err:.1f}%",
            xy=(knee_t, temporal_compression[temporal_ranks.index(knee_t)]),
            xytext=(knee_t + max(0.3, rt * 0.1), temporal_compression[temporal_ranks.index(knee_t)]),
            color="#d29922", fontsize=8,
        )

        fig.suptitle(
            f"Tucker: rank vs reconstruction error vs compression  |  {data_desc}",
            color="#e6edf3", fontsize=10, y=1.01,
        )
        fig.subplots_adjust(wspace=0.32)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        sweep_png = buf.getvalue()

        data_url = _png_data_url(sweep_png)
        _emit_event({"type": "image", "dataUrl": data_url})

        # ── Persist to disk so the website can show past sweeps ──────────
        import time as _time
        import json as _json_rs
        _sweep_dir = _os.path.join(_os.getcwd(), "outputs", "rank_sweeps")
        _os.makedirs(_sweep_dir, exist_ok=True)
        _ts   = int(_time.time())
        _sid  = f"{_ts}_{source}_{variable}"
        _meta = {
            "id":             _sid,
            "timestamp":      _ts,
            "date_label":     __import__("datetime").datetime.utcfromtimestamp(_ts).strftime("%Y-%m-%d %H:%M UTC"),
            "source":         source,
            "variable":       variable,
            "data_desc":      data_desc,
            "quality":        quality,
            "n_timesteps":    T,
            "frame_shape":    [H, W],
            "spatial_ranks":  spatial_ranks,
            "spatial_errors": [round(float(e), 6) for e in spatial_errors],
            "temporal_ranks": temporal_ranks,
            "temporal_errors":[round(float(e), 6) for e in temporal_errors],
            "knee_spatial":   int(knee_s),
            "knee_temporal":  int(knee_t),
            "data_url":       data_url,   # embedded so the frontend needs no extra request
        }
        _json_path = _os.path.join(_sweep_dir, f"{_sid}.json")
        with open(_json_path, "w", encoding="utf-8") as _jf:
            _json_rs.dump(_meta, _jf)
        _progress(f"Rank sweep saved → outputs/rank_sweeps/{_sid}.json")
        _progress(f"Rank sweep done. Suggested ranks: spatial={knee_s}, temporal={knee_t}")
        return b""

    def _reconstruct(vec_c):
        """Map Tucker-compressed vector (rs²,) back to full spatial array (H×W)."""
        return (U_h @ vec_c.real.astype(np.float64).reshape(rs, rs) @ U_w.T).astype(np.float32)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4a — Tucker HOSVD visualisation (ALL rt modes simultaneously)
    # Each panel: spatial_k * temporal_k[t] — shows the principal space-time
    # patterns sorted by explained variance (like PCA but on the tensor).
    # ════════════════════════════════════════════════════════════════════════
    if tech == "hosvd":
        core_norms = np.array([float(np.linalg.norm(core[:, :, i])) for i in range(rt)])
        total_norm_sq = max(float(np.sum(core_norms ** 2)), 1e-30)

        frames_to_show = min(n_frames, T)
        t_idx = np.round(np.linspace(0, T - 1, frames_to_show)).astype(int)

        # Pre-compute all rt mode animations
        hosvd_anims: list = []
        for mi in range(rt):
            spatial_c    = core[:, :, mi]
            spatial_full = (U_h @ spatial_c @ U_w.T).astype(np.float32)
            temporal_m   = U_t[:, mi].astype(np.float32)
            explained    = core_norms[mi] ** 2 / total_norm_sq

            frames_mi = [spatial_full * temporal_m[int(ti)] for ti in t_idx]
            fin_mi    = np.concatenate([f.ravel() for f in frames_mi])
            fin_mi    = fin_mi[np.isfinite(fin_mi)]
            vmax_mi   = float(np.percentile(np.abs(fin_mi), 99)) if fin_mi.size > 0 else 1.0

            hosvd_anims.append((frames_mi, vmax_mi, float(explained), mi))
            _progress(f"  Tucker mode {mi}: expl {explained:.1%}  vmax={vmax_mi:.3g}")

        suptitle = f"Tucker HOSVD all modes | {ratio:.0f}x compression | {data_desc}"

        for fi in range(frames_to_show):
            arrays    = [ha[0][fi]  for ha in hosvd_anims]
            vmaxes    = [ha[1]      for ha in hosvd_anims]
            subtitles = [f"mode {ha[3]}\nexpl {ha[2]:.1%}" for ha in hosvd_anims]
            t_abs     = ts_real[int(t_idx[fi])]
            date_str  = date_fn(int(t_abs))
            png = _render_all_modes_frame(
                arrays, subtitles, f"{suptitle} | {date_str}",
                cmap, vmaxes, fi, frames_to_show,
            )
            _emit_frame(fi, frames_to_show, date_str, png)
        return b""

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4b — TRUE multi-resolution DMD (mrDMD, Kutz et al. 2016)
    #
    # Algorithm:
    #   Level 0 — DMD on full window X_tucker.
    #             Separate SLOW modes (period > T/2 steps) from fast ones.
    #             Reconstruct slow background and subtract → residual.
    #   Level 1 — Split residual into two half-windows.
    #             DMD on each half = mesoscale dynamics.
    #   (Level 2 — would split halves again = fast synoptic; skipped here)
    #
    # Each level emits an all-modes grid animation so you see the spatial
    # structure and frequency of the dynamics at each scale separately.
    # ════════════════════════════════════════════════════════════════════════
    if tech == "rdmd":

        def _dmd_window(X_w):
            """DMD on Tucker-compressed window. Returns (eigs, Phi, b) sorted by amplitude."""
            if X_w.shape[1] < 3:
                return None, None, None
            X1_w, X2_w = X_w[:, :-1], X_w[:, 1:]
            U_, s_, Vh_ = np.linalg.svd(X1_w, full_matrices=False)
            r_  = min(n_modes, len(s_), X1_w.shape[1] - 1)
            if r_ < 1:
                return None, None, None
            A_  = (U_[:, :r_].conj().T @ X2_w) @ Vh_[:r_, :].conj().T / s_[:r_]
            eigs_, W_ = np.linalg.eig(A_)
            Phi_  = X2_w @ Vh_[:r_, :].conj().T / s_[:r_] @ W_
            b_    = np.linalg.lstsq(Phi_, X_w[:, 0], rcond=None)[0]
            ord_  = np.argsort(np.abs(b_))[::-1]
            return eigs_[ord_], Phi_[:, ord_], b_[ord_]

        def _reconstruct_window(eigs_, Phi_, b_, T_w, mask=None):
            """Reconstruct data matrix from selected modes (default: all)."""
            idx_ = range(len(eigs_)) if mask is None else np.where(mask)[0]
            X_r  = np.zeros((Phi_.shape[0], T_w), dtype=complex)
            for k in idx_:
                pows = np.array([eigs_[k] ** t for t in range(T_w)])
                X_r += np.outer(Phi_[:, k] * b_[k], pows)
            return X_r.real

        # ── Level 0: full window (mean-subtracted anomalies) ─────────────────
        X_mean_mrdmd = X_tucker.mean(axis=1, keepdims=True)
        X_tucker_mrdmd = X_tucker - X_mean_mrdmd
        _progress(f"mrDMD Level 0: full window ({X_tucker_mrdmd.shape[0]}x{T}), {ratio:.0f}x compression (anomalies)")
        eigs0, Phi0, b0 = _dmd_window(X_tucker_mrdmd)
        if eigs0 is None:
            raise RuntimeError("mrDMD level-0 failed (too few snapshots)")

        # Frequency threshold: "slow" = period longer than half the window
        # i.e. |angle(lambda)| < 2*pi / (T/2)
        freq_thresh = 2.0 * np.pi / max(T / 2, 2)
        slow_mask   = np.abs(np.angle(eigs0)) <= freq_thresh
        n_slow      = int(slow_mask.sum())
        _progress(f"  {n_slow} slow modes (period > T/2), "
                  f"{int((~slow_mask).sum())} fast — subtracting slow background")

        X_slow     = _reconstruct_window(eigs0, Phi0, b0, T, slow_mask)
        X_residual = X_tucker_mrdmd - X_slow

        # ── Level 1: two half-windows on the residual ────────────────────────
        mid = T // 2
        _progress(f"mrDMD Level 1: two half-windows on residual (mid={mid})")
        eigs1a, Phi1a, b1a = _dmd_window(X_residual[:, :mid + 1])
        eigs1b, Phi1b, b1b = _dmd_window(X_residual[:, mid:])

        # ── Collect all levels ───────────────────────────────────────────────
        levels_info = []
        if n_slow > 0:
            levels_info.append({
                "label":  "L0 slow background",
                "eigs":   eigs0[slow_mask],
                "Phi":    Phi0[:, slow_mask],
                "b":      b0[slow_mask],
                "ts":     ts_real,
            })
        if eigs1a is not None:
            levels_info.append({
                "label":  "L1 mesoscale (first half)",
                "eigs":   eigs1a,
                "Phi":    Phi1a,
                "b":      b1a,
                "ts":     ts_real[:mid + 1],
            })
        if eigs1b is not None:
            levels_info.append({
                "label":  "L1 mesoscale (second half)",
                "eigs":   eigs1b,
                "Phi":    Phi1b,
                "b":      b1b,
                "ts":     ts_real[mid:],
            })

        if not levels_info:
            raise RuntimeError("mrDMD: no levels produced")

        frames_per_level = max(4, n_frames // len(levels_info))
        total_out     = frames_per_level * len(levels_info)
        frame_counter = 0

        # ── Slow/fast energy split + redesigned spectrum (Level 0) ───────────
        slow_energy_frac = float(
            np.linalg.norm(X_slow) ** 2 / max(np.linalg.norm(X_tucker_mrdmd) ** 2, 1e-12)
        )
        _progress(f"  Slow branch carries {slow_energy_frac:.0%} of the variance")
        _progress("Rendering mrDMD spectrum (Level 0)")
        spec_png = _render_dmd_spectrum(eigs0, b0, dt_eff_min, f"mrDMD L0 | {data_desc}")
        _emit_event({"type": "dmd_spectrum", "dataUrl": _png_data_url(spec_png)})

        _day = 1440.0
        levels_payload = []

        # Animate each level + collect structured per-mode metadata for cards
        for lv in levels_info:
            lv_label  = lv["label"]
            eigs_l    = lv["eigs"]
            Phi_l     = lv["Phi"]
            b_l       = lv["b"]
            n_modes_l = Phi_l.shape[1]
            amp_tot_l = float(np.abs(b_l).sum()) or 1.0

            _progress(f"  Animating {lv_label}: {n_modes_l} modes x {frames_per_level} frames")

            lv_anims: list = []
            lv_modes: list = []
            for ki in range(n_modes_l):
                eig_i    = eigs_l[ki]
                amp_i    = complex(b_l[ki])
                mag_i    = float(abs(eig_i))
                angle_i  = float(np.angle(eig_i))
                period_i = _period_human(angle_i, dt_min=dt_eff_min)
                growth_i = float(np.log(max(mag_i, 1e-12)) / (dt_eff_min / _day) * 100.0)
                mode_c_i = Phi_l[:, ki]

                patt  = _reconstruct(mode_c_i * amp_i)
                pfin  = patt[np.isfinite(patt)]
                pvmax = float(np.percentile(np.abs(pfin), 99)) if pfin.size else 1.0
                thumb = _render_equirectangular(patt, cmap="RdBu_r", vmin=-pvmax, vmax=pvmax)
                lv_modes.append({
                    "idx": ki,
                    "period_label": period_i,
                    "period_min": (round(2.0 * np.pi / abs(angle_i) * dt_eff_min, 1)
                                   if abs(angle_i) > 1e-8 else None),
                    "growth_per_day": round(growth_i, 2),
                    "behavior": _dmd_behavior(mag_i),
                    "amplitude_share": round(float(np.abs(b_l[ki]) / amp_tot_l), 4),
                    "pattern": _png_data_url(thumb),
                })

                ps_i = 2.0 * np.pi / abs(angle_i) if abs(angle_i) > 1e-8 else float("inf")
                t_a  = (np.linspace(0, ps_i, frames_per_level, endpoint=False)
                        if np.isfinite(ps_i) and ps_i < frames_per_level * 5
                        else np.arange(frames_per_level, dtype=float))
                frames_ki = [_reconstruct((mode_c_i * amp_i * eig_i ** t).real) for t in t_a]
                fin_ki = np.concatenate([f.ravel() for f in frames_ki])
                fin_ki = fin_ki[np.isfinite(fin_ki)]
                vmax_ki = float(np.percentile(np.abs(fin_ki), 99)) if fin_ki.size > 0 else 1.0
                lv_anims.append((frames_ki, vmax_ki, mag_i, period_i, ki))

            levels_payload.append({"label": lv_label, "modes": lv_modes})

            for fi in range(frames_per_level):
                arrays_    = [la[0][fi]  for la in lv_anims]
                vmaxes_    = [la[1]      for la in lv_anims]
                subtitles_ = [f"mode {la[4]}\n|λ|={la[2]:.3f} T={la[3]}" for la in lv_anims]
                phase      = fi / max(frames_per_level, 1)
                suptitle_  = f"Tucker+mrDMD | {lv_label} | {data_desc} | phase {phase:.2f}"
                png = _render_all_modes_frame(
                    arrays_, subtitles_, suptitle_, "RdBu_r",
                    vmaxes_, frame_counter, total_out,
                )
                _emit_frame(frame_counter, total_out,
                            f"{lv_label} · phase {phase:.2f}", png)
                frame_counter += 1

        _emit_event({
            "type": "mrdmd_levels",
            "levels": levels_payload,
            "slow_energy_frac": round(slow_energy_frac, 4),
        })

        return b""

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4c — Exact DMD on Tucker-compressed ANOMALIES
    # Subtract the temporal mean before DMD so modes show oscillatory patterns
    # rather than the mean state (which would dominate mode 0 and leave all
    # other modes visually empty).
    # ════════════════════════════════════════════════════════════════════════
    _progress(f"Stage 4 — Tucker+DMD on ({X_tucker.shape[0]}×{T}), "
              f"r={n_modes}, {ratio:.0f}× compression")

    # Mean subtraction — isolate dynamics from background state
    X_mean   = X_tucker.mean(axis=1, keepdims=True)   # (rs², 1)
    X_anom   = X_tucker - X_mean                       # anomalies
    _progress(f"  Mean subtracted — DMD operates on anomalies")

    X1_t = X_anom[:, :-1]
    X2_t = X_anom[:, 1:]

    U_t2, s_t, Vh_t = np.linalg.svd(X1_t, full_matrices=False)
    r   = min(n_modes, len(s_t))
    A_red = (U_t2[:, :r].conj().T @ X2_t) @ Vh_t[:r, :].conj().T / s_t[:r]
    eigenvalues, W_t = np.linalg.eig(A_red)
    Phi_t = X2_t @ Vh_t[:r, :].conj().T / s_t[:r] @ W_t   # (rs², r)
    b_t   = np.linalg.lstsq(Phi_t, X_anom[:, 0], rcond=None)[0]

    _progress(f"DMD complete. {r} modes. |lambda|: {np.abs(eigenvalues).round(3).tolist()}")

    # Sort by amplitude; exclude explosively growing numerical artefacts (|λ|>1.1)
    stable  = np.abs(eigenvalues) <= 1.1
    order_t = np.argsort(np.where(stable, np.abs(b_t), 0.0))[::-1]
    eigenvalues = eigenvalues[order_t]
    Phi_t       = Phi_t[:, order_t]
    b_t         = b_t[order_t]

    # Drop negligible modes (< 1% of total amplitude) so blank panels/cards don't
    # clutter the analysis; always keep at least the top 3.
    amp_all   = np.abs(b_t)
    amp_total = float(amp_all.sum()) or 1.0
    n_keep    = max(3, int((amp_all / amp_total >= 0.01).sum()))
    n_keep    = min(n_keep, len(eigenvalues))
    eigenvalues = eigenvalues[:n_keep]
    Phi_t       = Phi_t[:, :n_keep]
    b_t         = b_t[:n_keep]
    _progress(f"Keeping {n_keep} modes (≥1% amplitude) of {r} computed")

    _day = 1440.0

    # ── Per-mode metadata + pattern thumbnail + animation frames ──────────
    mode_meta:  list = []
    mode_anims: list = []
    for ki in range(n_keep):
        eig_i    = eigenvalues[ki]
        amp_i    = complex(b_t[ki])
        mag_i    = float(abs(eig_i))
        angle_i  = float(np.angle(eig_i))
        period_i = _period_human(angle_i, dt_min=dt_eff_min)
        growth_i = float(np.log(max(mag_i, 1e-12)) / (dt_eff_min / _day) * 100.0)
        behav_i  = _dmd_behavior(mag_i)
        share_i  = float(np.abs(b_t[ki]) / amp_total)
        mode_c_i = Phi_t[:, ki]

        # Spatial pattern (phase 0) thumbnail — symmetric RdBu_r
        patt  = _reconstruct(mode_c_i * amp_i)
        pfin  = patt[np.isfinite(patt)]
        pvmax = float(np.percentile(np.abs(pfin), 99)) if pfin.size else 1.0
        thumb = _render_equirectangular(patt, cmap="RdBu_r", vmin=-pvmax, vmax=pvmax)

        mode_meta.append({
            "idx": ki,
            "period_label": period_i,
            "period_min": (round(2.0 * np.pi / abs(angle_i) * dt_eff_min, 1)
                           if abs(angle_i) > 1e-8 else None),
            "growth_per_day": round(growth_i, 2),
            "behavior": behav_i,
            "amplitude_share": round(share_i, 4),
            "pattern": _png_data_url(thumb),
        })

        # Animation frames spanning one oscillation period
        ps_i  = 2.0 * np.pi / abs(angle_i) if abs(angle_i) > 1e-8 else float("inf")
        t_a_i = (np.linspace(0, ps_i, n_frames, endpoint=False)
                 if np.isfinite(ps_i) and ps_i < n_frames * 5
                 else np.arange(n_frames, dtype=float))
        frames_i = [_reconstruct((mode_c_i * amp_i * eig_i ** t).real) for t in t_a_i]
        fin_i  = np.concatenate([f.ravel() for f in frames_i])
        fin_i  = fin_i[np.isfinite(fin_i)]
        vmax_i = float(np.percentile(np.abs(fin_i), 99)) if fin_i.size > 0 else 1.0
        mode_anims.append((frames_i, vmax_i, mag_i, period_i, ki))
        _progress(f"  mode {ki}: T={period_i}  {behav_i}  amp={share_i:.0%}")

    _emit_event({"type": "dmd_modes", "modes": mode_meta})

    # ── Spectrum: complex plane + period/growth (physical) ────────────────
    _progress("Rendering DMD spectrum")
    spec_png = _render_dmd_spectrum(eigenvalues, b_t, dt_eff_min, data_desc)
    _emit_event({"type": "dmd_spectrum", "dataUrl": _png_data_url(spec_png)})

    # ── Reconstruction check: original vs DMD(kept modes) vs residual ─────
    _progress("Rendering reconstruction check")
    orig_sp  = _reconstruct(X_anom[:, 0])
    recon_sp = _reconstruct(Phi_t @ b_t)
    resid_sp = orig_sp - recon_sp
    recon_png = _render_triptych(
        [orig_sp, recon_sp, resid_sp],
        ["Original anomaly (t0)", f"DMD reconstruction ({n_keep} modes)", "Residual"],
        f"DMD reconstruction check  |  {data_desc}",
    )
    _emit_event({"type": "dmd_reconstruction", "dataUrl": _png_data_url(recon_png)})

    # ── Mode animation grid (phase-labelled, kept modes only) ─────────────
    suptitle = f"Tucker+DMD modes | {data_desc} | {ratio:.0f}x compression"
    _progress(f"Rendering {n_frames} mode-grid frames ({n_keep} panels each)")
    for fi in range(n_frames):
        arrays    = [ma[0][fi] for ma in mode_anims]
        vmaxes    = [ma[1]     for ma in mode_anims]
        subtitles = [f"mode {ma[4]}  |λ|={ma[2]:.3f}\nT={ma[3]}" for ma in mode_anims]
        phase     = fi / max(n_frames, 1)
        png = _render_all_modes_frame(
            arrays, subtitles, f"{suptitle} | phase {phase:.2f}",
            "RdBu_r", vmaxes, fi, n_frames,
        )
        _emit_frame(fi, n_frames, f"phase {phase:.2f}", png)

    return b""
