"""FastAPI backend for SciVis 2026 LLC4320 volume visualization.

Loads sub-volumes on demand via OpenVisus / openvisuspy from the OSDF Pelican
endpoints listed on https://sciviscontest2026.github.io/examples/#access-llc4320
and serves them as raw float32 binary buffers to the Three.js frontend.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

import openvisuspy as ovp
import OpenVisus as ov


# ─── Dataset registry ─────────────────────────────────────────────────────────
#
# Every dataset the globe can render, LLC mosaics and cubed-sphere alike.
#   grid "llc"   — single lat-lon-like mosaic IDX, read via get_dataset()
#   grid "cubed" — 6 per-face IDX files (DYAMOND GEOS), reprojected to
#                  equirectangular server-side in /api/globe_volume
#
# LLC2160 is the same mosaic layout as LLC4320 at exactly half resolution
# (17280×12960 vs 8640×6480, verified), so the LLC4320 longitude-roll fraction
# and latitude LUT both apply to it unchanged.

_DYAMOND_LLC2160_BASE = (
    "https://nsdf-climate3-origin.nationalresearchplatform.org:50098"
    "/nasa/nsdf/climate3/dyamond/"
)

# At full LLC4320 res (W=17280) lon=−180° starts at x≈10464.
LLC_LON_ROLL_FRAC = 10464 / 17280

ECCO_TIME_INFO = {"t0_iso": "2011-09-10T00:00:00Z", "dt_hours": 1}
DYAMOND_TIME_INFO = {"t0_iso": "2020-01-20T00:00:00Z", "dt_hours": 1}

DATASETS: dict[str, dict[str, Any]] = {
    # ── ECCO LLC4320 ocean ────────────────────────────────────────────────
    "theta": {
        "label": "Temperature (theta)",
        "url": "pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/theta/theta_llc4320_x_y_depth.idx",
        "group": "ecco", "grid": "llc", "depth_levels": 90,
        "time_info": ECCO_TIME_INFO,
    },
    "salt": {
        "label": "Salinity (salt)",
        "url": "pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/salt/salt_llc4320_x_y_depth.idx",
        "group": "ecco", "grid": "llc", "depth_levels": 90,
        "time_info": ECCO_TIME_INFO,
    },
    "w": {
        "label": "Vertical velocity (w)",
        "url": "pelican://osg-htc.org/nasa/nsdf/climate2/llc4320/idx/w/w_llc4320_x_y_depth.idx",
        "group": "ecco", "grid": "llc", "depth_levels": 90,
        "time_info": ECCO_TIME_INFO,
    },
    # NOTE: ECCO u/v are not published on the NSDF federation (probed
    # climate1-3 with u/v/uvel/vvel patterns on 2026-07-03 — all absent).

    # ── DYAMOND LLC2160 ocean ─────────────────────────────────────────────
    "dyamond_theta": {
        "label": "DYAMOND Temperature (theta)",
        "url": _DYAMOND_LLC2160_BASE + "mit_output/llc2160_theta/llc2160_theta.idx",
        "group": "dyamond_ocean", "grid": "llc", "depth_levels": 90,
        "time_info": DYAMOND_TIME_INFO,
    },
    "dyamond_salt": {
        "label": "DYAMOND Salinity (salt)",
        "url": _DYAMOND_LLC2160_BASE + "mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
        "group": "dyamond_ocean", "grid": "llc", "depth_levels": 90,
        "time_info": DYAMOND_TIME_INFO,
    },
    "dyamond_w": {
        "label": "DYAMOND Vert. velocity (w)",
        "url": _DYAMOND_LLC2160_BASE + "mit_output/llc2160_w/llc2160_w.idx",
        "group": "dyamond_ocean", "grid": "llc", "depth_levels": 90,
        "time_info": DYAMOND_TIME_INFO,
    },

    # ── DYAMOND GEOS atmosphere (cubed-sphere, 6 faces × 1440² × 52) ──────
    "geos_t": {
        "label": "GEOS Air temperature (t)",
        "var": "t",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_u": {
        "label": "GEOS Zonal wind (u)",
        "var": "u",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_v": {
        "label": "GEOS Meridional wind (v)",
        "var": "v",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_fcld": {
        "label": "GEOS Cloud fraction (fcld)",
        "var": "fcld",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_qv": {
        "label": "GEOS Specific humidity (qv)",
        "var": "qv",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_w": {
        "label": "GEOS Vertical wind (w)",
        "var": "w",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_ql": {
        "label": "GEOS Cloud liquid water (ql)",
        "var": "ql",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_qi": {
        "label": "GEOS Cloud ice water (qi)",
        "var": "qi",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
    "geos_co2": {
        "label": "GEOS Carbon dioxide (co2)",
        "var": "co2",
        "group": "dyamond_atm", "grid": "cubed", "depth_levels": 52,
        "time_info": DYAMOND_TIME_INFO,
    },
}

# ── CMIP6 NEX-GDDP (single IDX dataset, uniform 0.25° lat-lon, 2D daily) ──────
# All variables/scenarios/models live as FIELDS of one dataset; the globe uses
# one fixed model (ACCESS-CM2, r1i1p1f1 — verified present with all scenarios
# 2026-07-08). Grid: 1440×600, lat −59.875…89.875 (NO coverage south of 60°S),
# native lon 0.125…359.875°E. Land-only downscaling — oceans are fill values.
CMIP6_TIME_INFO = {"t0_iso": "1950-01-01T00:00:00Z", "dt_hours": 24}
CMIP6_MODEL = "ACCESS-CM2"
CMIP6_MEMBER_GRID = "r1i1p1f1_gn"
CMIP6_SCENARIOS = ("historical", "ssp126", "ssp245", "ssp370", "ssp585")
_CMIP6_VARS = {
    "cmip6_tas":     ("tas",     "CMIP6 Air temperature (tas)"),
    "cmip6_tasmax":  ("tasmax",  "CMIP6 Max temperature (tasmax)"),
    "cmip6_tasmin":  ("tasmin",  "CMIP6 Min temperature (tasmin)"),
    "cmip6_pr":      ("pr",      "CMIP6 Precipitation (pr)"),
    "cmip6_sfcwind": ("sfcWind", "CMIP6 Surface wind (sfcWind)"),
    "cmip6_hurs":    ("hurs",    "CMIP6 Rel. humidity (hurs)"),
    "cmip6_huss":    ("huss",    "CMIP6 Spec. humidity (huss)"),
    "cmip6_rlds":    ("rlds",    "CMIP6 Longwave down (rlds)"),
    "cmip6_rsds":    ("rsds",    "CMIP6 Shortwave down (rsds)"),
}
for _key, (_var, _label) in _CMIP6_VARS.items():
    DATASETS[_key] = {
        "label": _label,
        "var": _var,
        # grid "latlon": plain uniform lat-lon equirect — read directly, no LLC
        # longitude roll (data is rolled half-width instead) and no cubed
        # reprojection. Partial latitude coverage handled by the lat LUT.
        "url": None,  # loaded via get_cmip6_db() (single shared IDX dataset)
        "group": "cmip6", "grid": "latlon", "depth_levels": 1,
        "time_info": CMIP6_TIME_INFO,
    }


_db_cache: dict[str, Any] = {}
_db_error_cache: dict[str, str] = {}
_db_lock = threading.Lock()


# ─── CMIP6 dataset ────────────────────────────────────────────────────────────

_CMIP6_URL = "https://atlantis.sci.utah.edu/mod_visus?dataset=nex-gddp-cmip6"

_cmip6_db: Any = None
_cmip6_db_error: str | None = None
_cmip6_lock = threading.Lock()


def get_cmip6_db():
    global _cmip6_db, _cmip6_db_error
    with _cmip6_lock:
        if _cmip6_db_error:
            raise HTTPException(status_code=502, detail=_cmip6_db_error)
        if _cmip6_db is not None:
            return _cmip6_db
        try:
            _cmip6_db = ov.LoadDataset(_CMIP6_URL)
            if _cmip6_db is None:
                raise RuntimeError("LoadDataset returned None")
            _cmip6_db_error = None
            return _cmip6_db
        except Exception as exc:
            _cmip6_db_error = f"CMIP6 server unreachable: {exc}"
            raise HTTPException(status_code=502, detail=_cmip6_db_error) from exc


# ─── DYAMOND GEOS atmosphere datasets ─────────────────────────────────────────

_DYAMOND_GEOS_TMPL = (
    "https://nsdf-climate3-origin.nationalresearchplatform.org:50098"
    "/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{VAR}/{var}_face_{face}_depth_52_time_0_10269.idx"
)

_dyamond_atm_cache: dict[str, Any] = {}
_dyamond_atm_error_cache: dict[str, str] = {}
_dyamond_atm_lock = threading.Lock()


def get_dyamond_atm_dataset(variable: str, face: int):
    cache_key = f"{variable}_{face}"
    with _dyamond_atm_lock:
        if (err := _dyamond_atm_error_cache.get(cache_key)):
            raise HTTPException(status_code=502, detail=err)
        if (db := _dyamond_atm_cache.get(cache_key)):
            return db
        url = _DYAMOND_GEOS_TMPL.format(VAR=variable.upper(), var=variable.lower(), face=face)
        try:
            db = ov.LoadDataset(url)
            _dyamond_atm_cache[cache_key] = db
            _dyamond_atm_error_cache.pop(cache_key, None)
            return db
        except Exception as exc:
            detail = f"failed to load DYAMOND ATM '{variable}' face {face}: {exc}"
            _dyamond_atm_error_cache[cache_key] = detail
            raise HTTPException(status_code=502, detail=detail) from exc


# ─── Cubed-sphere → equirectangular reprojection ──────────────────────────────
# Face layout verified against GEOS c1440 data on 2026-07-04 (seam-energy
# minimization over all dihedral orientations + visual coastline check):
# data faces 0,1,3,4 are equatorial (gnomonic centers at lon 0°, 90°E, 180°,
# 90°W), face 2 holds the North Pole, face 5 the South Pole. Each face also
# needs a fixed dihedral re-orientation before sampling.

# cube position order: E0 (lon 0°), E1 (90°E), E2 (180°), E3 (90°W), N, S
_CUBE_POS_TO_FACE = (0, 1, 3, 4, 2, 5)
# dihedral op per cube position: fliplr if op>=4, then rot90(k=op%4)
_CUBE_POS_OP = (0, 0, 1, 1, 2, 5)

_cubed_sampler_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_cubed_sampler_lock = threading.Lock()


def _cubed_sampler(out_w: int, out_h: int):
    """Per-output-pixel (cube_position, s, t) gnomonic sampling grids, cached."""
    with _cubed_sampler_lock:
        hit = _cubed_sampler_cache.get((out_w, out_h))
    if hit is not None:
        return hit

    lat = np.deg2rad(np.linspace(-90.0, 90.0, out_h, dtype=np.float64))[:, None]
    lon = np.deg2rad(np.linspace(-180.0, 180.0, out_w, endpoint=False, dtype=np.float64))[None, :]
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat) * np.ones_like(lon)

    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    pos = np.where(
        (az >= ax) & (az >= ay), np.where(z > 0, 4, 5),
        np.where(ax >= ay, np.where(x > 0, 0, 2), np.where(y > 0, 1, 3)),
    ).astype(np.int32)

    # face normal and local east direction per equatorial cube position
    normals = {0: (1, 0, 0), 1: (0, 1, 0), 2: (-1, 0, 0), 3: (0, -1, 0),
               4: (0, 0, 1), 5: (0, 0, -1)}
    east = {0: (0, 1, 0), 1: (-1, 0, 0), 2: (0, -1, 0), 3: (1, 0, 0)}
    s = np.zeros_like(x)
    t = np.zeros_like(x)
    for p in range(6):
        m = pos == p
        n = normals[p]
        d = np.maximum(x * n[0] + y * n[1] + z * n[2], 1e-9)
        if p < 4:
            e = east[p]
            sp = (x * e[0] + y * e[1]) / d
            tp = z / d
        else:  # polar: s along +X, t along +Y (dihedral op absorbs rotation)
            sp = x / d
            tp = y / d
        s = np.where(m, sp, s)
        t = np.where(m, tp, t)

    result = (pos, s.astype(np.float32), t.astype(np.float32))
    with _cubed_sampler_lock:
        _cubed_sampler_cache[(out_w, out_h)] = result
    return result


def _equirect_from_cubed_faces(
    face_arrays: list, out_w: int = 720, out_h: int = 360
) -> np.ndarray:
    """Reproject 6 cubed-sphere face arrays to equirectangular [out_h, out_w]."""
    # Faces are logically square (1440²); quality can downsample the two axes
    # unevenly, so resample everything to S×S before re-orienting.
    S = 0
    for arr in face_arrays:
        if arr is None:
            continue
        S = max(S, arr.shape[-2] if arr.ndim >= 2 else 1, arr.shape[-1])
    if S == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)

    # stack in CUBE POSITION order, re-oriented so row = t (south→north on
    # equatorial faces) and col = s
    stack = np.zeros((6, S, S), dtype=np.float32)
    for p in range(6):
        arr = face_arrays[_CUBE_POS_TO_FACE[p]]
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim > 2:
            a = np.squeeze(a)
        if a.ndim > 2:
            a = a.mean(axis=tuple(range(a.ndim - 2)))
        if a.shape != (S, S):
            yi = np.linspace(0, a.shape[0] - 1, S).astype(np.int32)
            xi = np.linspace(0, a.shape[1] - 1, S).astype(np.int32)
            a = a[yi[:, None], xi[None, :]]
        op = _CUBE_POS_OP[p]
        if op >= 4:
            a = np.fliplr(a)
        a = np.rot90(a, k=op % 4)
        stack[p] = np.nan_to_num(a, nan=0.0)

    pos, s, t = _cubed_sampler(out_w, out_h)

    px = np.clip((s + 1.0) * 0.5 * (S - 1), 0.0, S - 1.0)
    py = np.clip((t + 1.0) * 0.5 * (S - 1), 0.0, S - 1.0)
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, S - 1)
    y1 = np.clip(y0 + 1, 0, S - 1)
    fx2 = (px - x0).astype(np.float32)
    fy2 = (py - y0).astype(np.float32)

    out = (stack[pos, y0, x0] * (1 - fx2) * (1 - fy2)
         + stack[pos, y0, x1] * fx2 * (1 - fy2)
         + stack[pos, y1, x0] * (1 - fx2) * fy2
         + stack[pos, y1, x1] * fx2 * fy2)
    return out.astype(np.float32)


def format_dataset_load_error(key: str, url: str, exc: Exception) -> str:
    # openvisuspy currently assumes a Pelican redirect Location header exists.
    # If the federation responds without it, requests raises KeyError('location').
    if isinstance(exc, KeyError) and str(exc).strip("'\"").lower() == "location":
        return (
            f"failed to load dataset '{key}': Pelican director for '{url}' did not "
            "return a redirect Location header; the dataset endpoint may be "
            "unavailable or misconfigured"
        )
    return f"failed to load dataset '{key}' from '{url}': {exc}"


def get_dataset(key: str):
    if key not in DATASETS:
        raise HTTPException(status_code=404, detail=f"unknown dataset '{key}'")
    if "url" not in DATASETS[key]:
        raise HTTPException(
            status_code=400,
            detail=f"dataset '{key}' is cubed-sphere; only /api/globe_volume and /api/metadata support it",
        )
    with _db_lock:
        load_error = _db_error_cache.get(key)
        if load_error is not None:
            raise HTTPException(status_code=502, detail=load_error)

        db = _db_cache.get(key)
        if db is None:
            url = DATASETS[key]["url"]
            try:
                db = ovp.LoadDataset(url)
            except Exception as exc:
                detail = format_dataset_load_error(key, url, exc)
                _db_error_cache[key] = detail
                raise HTTPException(status_code=502, detail=detail) from exc

            _db_cache[key] = db
            _db_error_cache.pop(key, None)
        return db


def get_full_dims(db) -> tuple[int, int, int]:
    # logic_box is ((x0,y0,z0), (x1,y1,z1))
    box = db.getLogicBox()
    p1, p2 = box[0], box[1]
    return int(p2[0]), int(p2[1]), int(p2[2])


# ─── Latitude lookup table (lat → normalized mosaic row) ─────────────────────
#
# LLC4320 mosaic rows are NOT uniformly spaced in latitude (the middle row sits
# at ~26.6°S). The shader needs a 1D LUT mapping latitude (−90…90) to the
# normalized row index of the data array, built from the lat/lon grid file.

LATLON_PATH = Path(__file__).parent / "llc4320_latlon.nc"
LATLUT_SIZE = 2048

_latlut_cache: np.ndarray | None = None
_latlut_lock = threading.Lock()


def build_lat_lut() -> np.ndarray:
    import xarray as xr

    ds = xr.open_dataset(LATLON_PATH)
    lat2d = ds["latitude"]
    ny = lat2d.sizes["y"]
    nx = lat2d.sizes["x"]
    # Rows are ~constant latitude below the Arctic cap junction; take the
    # median over a spread of columns to smooth out the wavy northern edge.
    sample = lat2d.isel(x=slice(0, nx, max(1, nx // 64))).values  # [ny, ~64]
    lat_rows = np.median(sample, axis=1).astype(np.float64)
    # Enforce monotonicity so np.interp is well-defined.
    lat_rows = np.maximum.accumulate(lat_rows)

    rows_norm = np.arange(ny, dtype=np.float64) / (ny - 1)
    # Parameterise by sin(lat) so the shader can look up with p.y/r directly,
    # avoiding one asin() transcendental call per ray-march step.
    target_sinlat = np.linspace(-1.0, 1.0, LATLUT_SIZE)
    target_lat = np.degrees(np.arcsin(target_sinlat))
    lut = np.interp(target_lat, lat_rows, rows_norm, left=-1.0, right=-1.0)
    return lut.astype(np.float32)


def get_lat_lut() -> np.ndarray:
    global _latlut_cache
    with _latlut_lock:
        if _latlut_cache is None:
            if not LATLON_PATH.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"lat/lon grid file not found at {LATLON_PATH}",
                )
            try:
                _latlut_cache = build_lat_lut()
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"failed to build latitude LUT: {exc}"
                ) from exc
        return _latlut_cache


app = FastAPI(title="SciVis 2026 LLC4320 backend", version="0.1.0")


@app.on_event("startup")
async def _preload_datasets() -> None:
    """Eagerly load all dataset handles in the background so the first browser
    request doesn't stall while OpenVisus/Pelican initializes."""
    import asyncio, concurrent.futures

    def _load_all() -> None:
        for key, cfg in DATASETS.items():
            try:
                if cfg["grid"] == "cubed":
                    get_dyamond_atm_dataset(cfg["var"], 0)  # faces 1-5 load on demand
                elif cfg["grid"] == "latlon":
                    get_cmip6_db()  # one shared IDX handle for all cmip6_* keys
                else:
                    get_dataset(key)
            except Exception as exc:
                print(f"[startup] preload '{key}' failed: {exc}", flush=True)
            else:
                print(f"[startup] preloaded '{key}'", flush=True)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load_all)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "https://localhost:3000", "https://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=[
        "X-Shape", "X-Min", "X-Max", "X-Time", "X-Quality", "X-Z0", "X-Z1", "X-Size", "X-TimeLabels",
        "x-shape", "x-min", "x-max", "x-time", "x-quality", "x-z0", "x-z1", "x-size", "x-timelabels",
    ],
)



@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    return [
        {
            "key": k,
            "label": v["label"],
            "group": v["group"],
            "grid": v["grid"],
            "depth_levels": v["depth_levels"],
            "url": v.get("url") or "",
        }
        for k, v in DATASETS.items()
    ]


@app.get("/api/metadata/{key}")
def metadata(key: str) -> dict[str, Any]:
    cfg = DATASETS.get(key)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset '{key}'")
    if cfg["grid"] == "cubed":
        db = get_dyamond_atm_dataset(cfg["var"], 0)
        box = db.getLogicBox()
        fw, fh, fz = int(box[1][0]), int(box[1][1]), int(box[1][2])
        # Nominal equirectangular dims for a cubed-sphere of this face size.
        dims = (4 * fw, 2 * fh, fz)
        timesteps = [int(t) for t in db.getTimesteps()]
    elif cfg["grid"] == "latlon":
        db = get_cmip6_db()
        box = db.getLogicBox()
        dims = (int(box[1][0]), int(box[1][1]), 1)
        timesteps = [int(t) for t in db.getTimesteps()]
    else:
        db = get_dataset(key)
        dims = get_full_dims(db)
        timesteps = [int(t) for t in db.getTimesteps()]
    return {
        "key": key,
        "label": cfg["label"],
        "url": cfg.get("url") or "",
        "group": cfg["group"],
        "grid": cfg["grid"],
        "depth_levels": cfg["depth_levels"],
        "dims": list(dims),  # [X, Y, Z]
        "timesteps": timesteps,
        "default_quality": -15,
        "time_info": {
            **cfg["time_info"],
            "n_timesteps": len(timesteps),
        },
    }


@app.get("/api/globe/{key}")
def globe(
    key: str,
    time: int = 0,
    depth: int = Query(0, ge=0, le=89),
    quality: int = Query(-4, ge=-20, le=0),
) -> Response:
    """Return a single lat×lon depth slice as float32 binary, lon-rolled to −180°."""
    db = get_dataset(key)
    fx, fy, fz = get_full_dims(db)

    depth = max(0, min(fz - 1, depth))

    try:
        data = db.db.read(
            time=time,
            quality=quality,
            x=[0, fx],
            y=[0, fy],
            z=[depth, depth + 1],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenVisus read failed: {e}") from e

    arr = np.asarray(data, dtype=np.float32).squeeze()  # [Y, X]

    # Roll longitude axis: at full res (W=17280) lon=−180° is at x≈10464.
    # Proportionally adjust for the actual downsampled width.
    actual_w = arr.shape[1]
    roll_left = round(10464 * actual_w / 17280)
    arr = np.roll(arr, -roll_left, axis=1)

    # Replace non-finite values with 0 (land / missing → transparent in shader)
    arr = np.where(np.isfinite(arr), arr, 0.0)

    finite = arr[arr != 0.0]
    if finite.size > 0:
        vmin = float(np.percentile(finite, 1))
        vmax = float(np.percentile(finite, 99))
    else:
        vmin, vmax = 0.0, 1.0

    body = arr.tobytes(order="C")
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{arr.shape[0]},{arr.shape[1]}",
            "X-Min": str(vmin),
            "X-Max": str(vmax),
            "Cache-Control": "public, max-age=300",
        },
    )


@app.get("/api/latlut")
def lat_lut(
    grid: str = Query("llc4320"),
    lat0: float = Query(-90.0),
    lat1: float = Query(90.0),
) -> Response:
    """1D LUT: sin(latitude) (−1…1, uniform) → normalized data row index (0…1).

    grid=llc4320|llc2160 — LLC mosaic mapping from llc4320_latlon.nc (LLC2160 is
    the same layout at half resolution, so the normalized LUT is identical).
    grid=equirect — analytic mapping for uniform-latitude rows (reprojected
    cubed-sphere volumes and plain lat-lon datasets). Optional lat0/lat1 bound
    the grid's coverage (e.g. CMIP6 NEX-GDDP spans only −60…90°); latitudes
    outside coverage map to −1, which the shader treats as no-data.
    Entries outside grid coverage are −1. float32 binary.
    """
    if grid in ("llc4320", "llc2160", "llc"):
        lut = get_lat_lut()
    elif grid == "equirect":
        sinlat = np.linspace(-1.0, 1.0, LATLUT_SIZE)
        lat = np.degrees(np.arcsin(sinlat))
        lut = ((lat - lat0) / max(lat1 - lat0, 1e-6)).astype(np.float32)
        lut[(lat < lat0) | (lat > lat1)] = -1.0
    else:
        raise HTTPException(status_code=400, detail=f"unknown grid '{grid}'")
    return Response(
        content=lut.tobytes(order="C"),
        media_type="application/octet-stream",
        headers={
            "X-Size": str(lut.size),
            "Cache-Control": "public, max-age=86400",
        },
    )


# ─── Cubed-sphere globe volumes ──────────────────────────────────────────────
# Reprojection is the expensive step, so completed float32 volumes are kept in
# a small LRU keyed by the full request tuple (timeline scrubbing re-hits it).

_cubed_vol_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
_CUBED_VOL_CACHE_MAX = 12
_cubed_vol_lock = threading.Lock()


def _read_cubed_globe_volume(
    key: str, cfg: dict[str, Any], time: int, z0: int, z1: int, quality: int
) -> np.ndarray:
    """Read 6 GEOS faces for levels z0…z1 and reproject to an equirectangular
    float32 volume [Z, H, W] with non-finite values zeroed."""
    ck = (key, time, z0, z1, quality)
    with _cubed_vol_lock:
        cached = _cubed_vol_cache.get(ck)
        if cached is not None:
            _cubed_vol_cache.move_to_end(ck)
            return cached

    var = cfg["var"]

    def read_face(face: int) -> np.ndarray:
        db = get_dyamond_atm_dataset(var, face)
        data = db.read(time=time, quality=quality, z=[z0, z1 + 1])
        a = np.asarray(data, dtype=np.float32).squeeze()
        if a.ndim == 2:
            a = a[np.newaxis]
        return a

    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            faces = list(ex.map(read_face, range(6)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GEOS face read failed: {e}") from e

    # Quality may downsample z as well as x/y; use what actually came back.
    nz = min(f.shape[0] for f in faces)
    # Faces are logically square; quality may downsample axes unevenly, so
    # size the output from the best-resolved spatial axis.
    fh = max(max(f.shape[1], f.shape[2]) for f in faces)
    # A face spans 90° of latitude, so equirect rows ≈ 2× face rows. W = 2H,
    # capped so the wide axis stays within WebGL2 MAX_3D_TEXTURE_SIZE (2048).
    out_h = int(min(2 * fh, 1024))
    out_w = 2 * out_h
    levels = [
        _equirect_from_cubed_faces([faces[f][zi] for f in range(6)], out_w, out_h)
        for zi in range(nz)
    ]
    vol = np.stack(levels)
    vol = np.where(np.isfinite(vol), vol, 0.0).astype(np.float32)

    with _cubed_vol_lock:
        _cubed_vol_cache[ck] = vol
        while len(_cubed_vol_cache) > _CUBED_VOL_CACHE_MAX:
            _cubed_vol_cache.popitem(last=False)
    return vol


@app.get("/api/globe_volume/{key}")
def globe_volume(
    key: str,
    time: int = 0,
    z0: int = Query(0, ge=0),
    z1: int = Query(10, ge=0),
    quality: int = Query(-8, ge=-20, le=0),
    norm_min: float | None = Query(None),
    norm_max: float | None = Query(None),
    scenario: str = Query("ssp585"),
) -> Response:
    """Return depth layers z0…z1 (inclusive) as a uint8 block, shape [Z, Y, X].

    LLC mosaics are read directly; cubed-sphere (GEOS) datasets are reprojected
    to equirectangular server-side; plain lat-lon datasets (CMIP6) are read as a
    single 2D field (`scenario` selects the CMIP6 experiment; ignored for other
    grids). Values are normalized to 1…255 over the block's 1–99 percentile
    range; 0 marks land / missing.
    """
    cfg = DATASETS.get(key)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset '{key}'")

    fz = cfg["depth_levels"]
    z0 = max(0, min(fz - 1, z0))
    z1 = max(z0, min(fz - 1, z1))

    if cfg["grid"] == "cubed":
        vol = _read_cubed_globe_volume(key, cfg, time, z0, z1, quality)
    elif cfg["grid"] == "latlon":
        # CMIP6 NEX-GDDP: one shared IDX dataset; variable × scenario picks the
        # field. 2D (no z axis), daily timesteps, 1440×600 — small enough that
        # quality downsampling is unnecessary.
        if scenario not in CMIP6_SCENARIOS:
            raise HTTPException(status_code=400, detail=f"unknown scenario '{scenario}'")
        field = f"{cfg['var']}_day_{CMIP6_MODEL}_{scenario}_{CMIP6_MEMBER_GRID}"
        db = get_cmip6_db()
        # The IDX time axis is absolute day ordinals in a NOLEAP (365-day)
        # calendar: first step 711750 = 1950-01-01, 55115 steps = 151×365
        # → last = 2100-12-31. The API takes a relative index 0…55114.
        tsteps = db.getTimesteps()
        t_first = int(tsteps[0])
        t_abs = t_first + max(0, min(len(tsteps) - 1, time))
        try:
            data = db.read(field=field, time=t_abs)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"CMIP6 read failed ({field}, t={time}): {e}"
            ) from e
        vol = np.asarray(data, dtype=np.float32).squeeze()
        if vol.ndim != 2:
            raise HTTPException(status_code=500, detail=f"unexpected CMIP6 shape {vol.shape}")
        vol = vol[np.newaxis]  # [1, H, W]
        # Native longitude starts at 0.125°E; the shader's texture u=0 is
        # lon −180°, so roll half the width (NOT the LLC mosaic fraction).
        vol = np.roll(vol, vol.shape[-1] // 2, axis=-1)
        vol = np.where(np.isfinite(vol), vol, 0.0)
        z0 = z1 = 0
    else:
        db = get_dataset(key)
        fx, fy, _ = get_full_dims(db)

        # Single read for the full depth range — avoids N serial round-trips to
        # the remote IDX server (one db.read() per layer = 16× the latency).
        try:
            data = db.db.read(
                time=time,
                quality=quality,
                x=[0, fx],
                y=[0, fy],
                z=[z0, z1 + 1],
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"OpenVisus read failed: {e}"
            ) from e

        vol = np.asarray(data, dtype=np.float32)
        # Normalise to [Z, Y, X] — OpenVisus may return extra singleton dims.
        vol = vol.squeeze()
        if vol.ndim == 2:
            vol = vol[np.newaxis]  # single layer requested

        # WebGL2 MAX_3D_TEXTURE_SIZE is 2048 per axis on most GPUs.
        # Stride-downsample if either spatial dimension exceeds that limit.
        MAX_GPU_DIM = 2048
        h, w = vol.shape[-2], vol.shape[-1]
        if w > MAX_GPU_DIM or h > MAX_GPU_DIM:
            sx = (w + MAX_GPU_DIM - 1) // MAX_GPU_DIM
            sy = (h + MAX_GPU_DIM - 1) // MAX_GPU_DIM
            vol = vol[..., ::sy, ::sx]

        # Longitude roll — same mosaic fraction for LLC4320 and LLC2160.
        actual_w = vol.shape[-1]
        roll_left = round(LLC_LON_ROLL_FRAC * actual_w)
        vol = np.roll(vol, -roll_left, axis=-1)
        vol = np.where(np.isfinite(vol), vol, 0.0)

    if norm_min is not None and norm_max is not None:
        vmin, vmax = float(norm_min), float(norm_max)
    else:
        finite = vol[vol != 0.0]
        if finite.size > 0:
            vmin = float(np.percentile(finite, 1))
            vmax = float(np.percentile(finite, 99))
        else:
            vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)
    quant = np.where(vol == 0.0, 0, 1 + np.round(norm * 254.0)).astype(np.uint8)

    return Response(
        content=quant.tobytes(order="C"),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{quant.shape[0]},{quant.shape[1]},{quant.shape[2]}",
            "X-Min": str(vmin),
            "X-Max": str(vmax),
            "X-Z0": str(z0),
            "X-Z1": str(z1),
            "Cache-Control": "public, max-age=300",
        },
    )


@app.get("/api/volume/{key}")
def volume(
    key: str,
    time: int = 0,
    quality: int = Query(-15, ge=-20, le=0),
    x0: int | None = None,
    x1: int | None = None,
    y0: int | None = None,
    y1: int | None = None,
    z0: int | None = None,
    z1: int | None = None,
) -> Response:
    db = get_dataset(key)
    fx, fy, fz = get_full_dims(db)

    x0 = 0 if x0 is None else max(0, min(fx, x0))
    x1 = fx if x1 is None else max(x0 + 1, min(fx, x1))
    y0 = 0 if y0 is None else max(0, min(fy, y0))
    y1 = fy if y1 is None else max(y0 + 1, min(fy, y1))
    z0 = 0 if z0 is None else max(0, min(fz, z0))
    z1 = fz if z1 is None else max(z0 + 1, min(fz, z1))

    try:
        data = db.db.read(
            time=time,
            quality=quality,
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenVisus read failed: {e}") from e

    arr = np.asarray(data)
    body = arr.tobytes(order="C")

    headers = {
        "X-Shape": ",".join(str(s) for s in arr.shape),  # Z,Y,X shape from OpenVisus
        "X-Time": str(time),
        "X-Quality": str(quality),
        "Cache-Control": "public, max-age=300",
    }

    return Response(content=body, media_type="application/octet-stream", headers=headers)
