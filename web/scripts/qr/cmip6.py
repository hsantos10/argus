"""qr.cmip6 — NEX-GDDP CMIP6 query."""
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


def query_cmip6(p):
    from analysis.core.utils import load_cmip6, cmip6_field_name
    variable = p.get("variable", "tas")
    model = p.get("model", "ACCESS-CM2")
    scenario = p.get("scenario", "historical")
    t_start = int(p.get("t_start", 0))
    t_end = int(p.get("t_end", 10))
    render_mode = p.get("render_mode", "grid")
    cmap = _CMIP6_CMAP.get(variable, "viridis")

    _progress(f"Loading CMIP6 variable={variable} model={model} scenario={scenario} (quality=N/A)")
    db = load_cmip6()
    field = cmip6_field_name(variable, model, scenario)
    _progress(f"Resolved field name: {field}")
    _progress("Listing timesteps")
    all_ts = list(db.getTimesteps())

    if render_mode == "globe":
        t = _midpoint_timestep(all_ts, t_start, t_end)
        _progress(f"Reading midpoint timestep {t}")
        arr = np.squeeze(np.array(db.read(field=field, time=int(t)), dtype=np.float32))
        return _render_equirectangular(arr, cmap=cmap)

    ts = _subsample(all_ts, t_start, t_end, cap=6)
    _progress(f"Subsampling {len(ts)} of {len(all_ts)} timesteps")
    frames, titles = [], []
    for i, t in enumerate(ts):
        _progress(f"Reading timestep {t} ({i + 1}/{len(ts)})")
        arr = np.squeeze(np.array(db.read(field=field, time=int(t)), dtype=np.float32))
        frames.append(arr)
        titles.append(f"t={t}")
    return _render_grid(
        frames,
        titles,
        f"CMIP6 | {variable} | {model} | {scenario}",
        cmap=cmap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
