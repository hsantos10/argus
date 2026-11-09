"""
utils.py — SciViz2026 shared foundation
========================================
Provides URL constants, dataset loaders, tensor builders, math utilities,
and output saving for all task notebooks.

All dataset loaders return handles with the same two-level API:
  db.read(time=t, ...)       — DYAMOND GEOS, CMIP6, LLC2160 (ov.LoadDataset)
  db.db.read(time=t, ...)    — ECCO (wrapped ov.LoadDataset via _EccoDirectProxy)
"""

import os
import time
import warnings
import numpy as np
import OpenVisus as ov
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout


class _EccoDirectProxy:
    """Wraps an ov.LoadDataset to present the .db.read(...) interface
    so existing callers that expect ovp.LoadDataset style need no changes."""

    class _DbProxy:
        def __init__(self, ov_db):
            self._ov_db = ov_db

        def read(self, **kwargs):
            return self._ov_db.read(**kwargs)

    def __init__(self, ov_db):
        self._ov_db = ov_db
        self.db = self._DbProxy(ov_db)

    def getTimesteps(self):
        return self._ov_db.getTimesteps()

    def getLogicBox(self):
        return self._ov_db.getLogicBox()

def _load_dataset_with_timeout(url: str, timeout: float = 12.0):
    """Load an OpenVisus dataset, raising RuntimeError if it hangs longer than `timeout` seconds."""
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(ov.LoadDataset, url)
    try:
        result = fut.result(timeout=timeout)
        ex.shutdown(wait=False)
        return result
    except _FuturesTimeout:
        ex.shutdown(wait=False)  # don't block on the stuck thread
        raise RuntimeError(f"LoadDataset timed out after {timeout:.0f}s — server unreachable: {url[:80]}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. URL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_LLC4320_BASE = "https://nsdf-climate1-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate1/llc4320/idx"
LLC4320_URLS = {
    "theta": f"{_LLC4320_BASE}/theta/theta_llc4320_x_y_depth.idx",
    "salt":  f"{_LLC4320_BASE}/salt/salt_llc4320_x_y_depth.idx",
}

NEX_GDDP_URL = "https://atlantis.sci.utah.edu/mod_visus?dataset=nex-gddp-cmip6"

_DYAMOND_GEOS_BASE = (
    "https://nsdf-climate3-origin.nationalresearchplatform.org:50098"
    "/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{VAR}/{var}_face_{face}_depth_52_time_0_10269.idx"
)

_DYAMOND_LLC2160_BASE_URL = (
    "https://nsdf-climate3-origin.nationalresearchplatform.org:50098"
    "/nasa/nsdf/climate3/dyamond/"
)

# Per-variable paths from the contest reference notebook
# (ieee_scivis_dyamond_ocean.ipynb)
_LLC2160_PATHS = {
    "theta": "mit_output/llc2160_theta/llc2160_theta.idx",
    "w":     "mit_output/llc2160_w/llc2160_w.idx",
    "u":     "mit_output/llc2160_arco/visus.idx",
    "salt":  "mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
    "v":     "mit_output/llc2160_v/v_llc2160_x_y_depth.idx",
}

# Variables that have an ECCO LLC4320 fallback if LLC2160 is unreachable
_LLC2160_ECCO_FALLBACK = {"theta", "salt"}


def dyamond_geos_url(var: str, face: int) -> str:
    """Return the DYAMOND GEOS URL for a given variable and cubed-sphere face (0–5)."""
    return _DYAMOND_GEOS_BASE.format(VAR=var.upper(), var=var.lower(), face=face)


def dyamond_llc2160_url(var: str):
    """Return the DYAMOND LLC2160 ocean URL for a given variable, or None if unknown."""
    path = _LLC2160_PATHS.get(var.lower())
    if path is None:
        return None
    return _DYAMOND_LLC2160_BASE_URL + path


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_ecco(variable: str):
    """
    Load an ECCO LLC4320 dataset.

    Returns an _EccoDirectProxy handle.
    Data access: db.db.read(time=t, z=[z0, z1], quality=q)

    Parameters
    ----------
    variable : str
        One of 'theta', 'salt'.
    """
    if variable not in LLC4320_URLS:
        raise ValueError(
            f"ECCO variable '{variable}' is not available. "
            f"Supported: {list(LLC4320_URLS)}"
        )
    url = LLC4320_URLS[variable]
    print(f"[load_ecco] Loading {variable} from {url[:70]}...")
    inner_db = _load_dataset_with_timeout(url, timeout=15.0)
    dims = inner_db.getLogicBox()[1]
    ts   = inner_db.getTimesteps()
    print(f"  dims={dims}  timesteps={len(ts)}")
    return _EccoDirectProxy(inner_db)


def load_dyamond_geos(variable: str, face: int):
    """
    Load a DYAMOND GEOS atmosphere dataset.

    Returns an ov.LoadDataset handle.
    Data access: db.read(time=t, z=[z0, z1], quality=q)

    Parameters
    ----------
    variable : str
        E.g. 'u', 'v', 't', 'co', 'co2'.
    face : int
        Cubed-sphere face index, 0–5.
    """
    url = dyamond_geos_url(variable, face)
    print(f"[load_dyamond_geos] Loading {variable} face={face} from {url[:70]}...")
    db = _load_dataset_with_timeout(url, timeout=8.0)
    dims = db.getLogicBox()[1]
    ts   = db.getTimesteps()
    print(f"  dims={dims}  timesteps={len(ts)}")
    return db


def load_dyamond_llc2160(variable: str):
    """
    Load DYAMOND LLC2160 ocean data.

    Tries the per-variable URL from the contest reference notebook first.
    If that fails and the variable is one of {theta, salt, w}, falls back to
    ECCO LLC4320 (same spatial structure, different time period — Sep2011–Nov2012
    instead of Jan2020).

    Returns (db, source) where source is 'llc2160' or 'ecco_fallback'.
    Data access for llc2160:      db.read(time=t, z=[z0,z1], quality=q)   [ov]
    Data access for ecco_fallback: db.db.read(time=t, z=[z0,z1], quality=q) [ovp]
    """
    url = dyamond_llc2160_url(variable)
    if url is not None:
        try:
            print(f"[load_dyamond_llc2160] Trying LLC2160 URL for {variable}...")
            print(f"  {url[:80]}...")
            db = _load_dataset_with_timeout(url, timeout=8.0)
            _ = db.getLogicBox()
            dims = db.getLogicBox()[1]
            ts   = db.getTimesteps()
            print(f"  LLC2160 OK: dims={dims}  timesteps={len(ts)}")
            return db, "llc2160"
        except Exception as e:
            print(f"  LLC2160 URL failed: {e}")

    if variable in _LLC2160_ECCO_FALLBACK:
        warnings.warn(
            f"[load_dyamond_llc2160] LLC2160 unreachable for '{variable}'. "
            "Falling back to ECCO LLC4320 (same grid, temporal coverage Sep2011–Nov2012).",
            RuntimeWarning,
            stacklevel=2,
        )
        db = load_ecco(variable)
        return db, "ecco_fallback"

    raise RuntimeError(
        f"LLC2160 URL for '{variable}' failed and no ECCO fallback exists for this variable."
    )


def load_cmip6():
    """
    Load NEX GDDP CMIP6 dataset.

    Returns an ov.LoadDataset handle.
    Data access: db.read(field=field_name, time=t)

    Fields follow the pattern: '{var}_day_{MODEL}_{scenario}_r1i1p1f1_gn'
    """
    last_exc = None
    for attempt in range(3):
        try:
            db = ov.LoadDataset(NEX_GDDP_URL)
            if db is None:
                raise RuntimeError("LoadDataset returned None")
            dims   = db.getLogicBox()[1]
            ts     = db.getTimesteps()
            fields = db.getFields()
            print(f"  dims={dims}  timesteps={len(ts)}  fields={len(fields)}")
            return db
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.5)
    raise RuntimeError(
        "CMIP6 data source is currently unreachable — the OpenVisus server "
        "(atlantis.sci.utah.edu/mod_visus) is not responding. Please try the "
        "DYAMOND or ECCO datasets instead."
    ) from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# 3. TENSOR BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_tensor_ecco(db, timesteps, depth_range=(0, 10), quality=-6):
    """
    Build a 4-D float32 tensor from an ECCO LLC4320 dataset.

    Shape: (T, D, lat, lon)
      T   = len(timesteps)
      D   = depth_range[1] - depth_range[0]
      lat = spatial height at given quality
      lon = spatial width  at given quality

    Parameters
    ----------
    db          : ovp.LoadDataset handle (use db.db.read)
    timesteps   : list of integer time indices
    depth_range : (z_start, z_end) — passed as z=[z_start, z_end] to read()
    quality     : int, -15 (coarsest) to 0 (full resolution)
    """
    T = len(timesteps)
    # Probe first slice to get spatial shape
    probe = db.db.read(time=timesteps[0], z=list(depth_range), quality=quality)
    probe = np.array(probe, dtype=np.float32)
    shape = (T,) + probe.shape          # (T, D, lat, lon)
    tensor = np.empty(shape, dtype=np.float32)
    tensor[0] = probe

    for i, t in enumerate(timesteps[1:], start=1):
        slice_ = db.db.read(time=t, z=list(depth_range), quality=quality)
        tensor[i] = np.array(slice_, dtype=np.float32)
        if (i + 1) % 5 == 0 or i == T - 1:
            print(f"  [build_tensor_ecco] {i+1}/{T} timesteps loaded", end="\r")
    print()
    return tensor


def build_tensor_cmip6(db, field, timesteps):
    """
    Build a 3-D float32 tensor from NEX GDDP CMIP6.

    Shape: (T, lat, lon)

    Parameters
    ----------
    db        : ov.LoadDataset handle (use db.read)
    field     : str, e.g. 'tas_day_ACCESS-CM2_historical_r1i1p1f1_gn'
    timesteps : list of integer time indices
    """
    T = len(timesteps)
    probe = np.squeeze(np.array(db.read(field=field, time=timesteps[0]), dtype=np.float32))
    tensor = np.empty((T,) + probe.shape, dtype=np.float32)
    tensor[0] = probe

    for i, t in enumerate(timesteps[1:], start=1):
        arr = np.squeeze(np.array(db.read(field=field, time=t), dtype=np.float32))
        tensor[i] = arr
        if (i + 1) % 5 == 0 or i == T - 1:
            print(f"  [build_tensor_cmip6] {i+1}/{T} timesteps loaded", end="\r")
    print()
    return tensor


def build_snapshot_matrix_dyamond(db, timesteps, z_level=50, quality=-4):
    """
    Build a DMD snapshot matrix from a DYAMOND GEOS dataset.

    Shape: (space, T) where space = lat * lon at the given quality.
    NaN values are replaced with 0 before returning.

    Parameters
    ----------
    db        : ov.LoadDataset handle (use db.read)
    timesteps : list of integer time indices
    z_level   : int, vertical level index (0 = top, higher = closer to surface depends on var)
    quality   : int, resolution quality
    """
    T = len(timesteps)
    probe = np.array(db.read(time=timesteps[0], z=[z_level, z_level + 1], quality=quality),
                     dtype=np.float32)
    probe = np.squeeze(probe).ravel()
    space = probe.shape[0]

    X = np.empty((space, T), dtype=np.float32)
    X[:, 0] = np.nan_to_num(probe, nan=0.0)

    for i, t in enumerate(timesteps[1:], start=1):
        arr = np.array(db.read(time=t, z=[z_level, z_level + 1], quality=quality), dtype=np.float32)
        X[:, i] = np.nan_to_num(np.squeeze(arr).ravel(), nan=0.0)
        if (i + 1) % 10 == 0 or i == T - 1:
            print(f"  [build_snapshot_matrix_dyamond] {i+1}/{T} timesteps loaded", end="\r")
    print()
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def nan_to_zero(arr: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf with 0. Returns a float32 copy."""
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalize arr to [0, 1].
    If the range is 0 (constant array), returns zeros.
    """
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def cmip6_field_name(var: str, model: str, scenario: str,
                     member: str = "r1i1p1f1", grid: str = "gn") -> str:
    """
    Construct a CMIP6 field name in the NEX GDDP OpenVisus dataset.

    Pattern: '{var}_day_{model}_{scenario}_{member}_{grid}'

    Parameters
    ----------
    var      : e.g. 'tas', 'hurs'
    model    : e.g. 'ACCESS-CM2', 'BCC-CSM2-MR'
    scenario : e.g. 'historical', 'ssp245', 'ssp585'
    """
    return f"{var}_day_{model}_{scenario}_{member}_{grid}"


def subsample_timesteps(all_ts: list, n: int, start_idx: int = 0) -> list:
    """
    Select n evenly-spaced timestep indices from all_ts, starting at start_idx.

    Parameters
    ----------
    all_ts    : list of all available timestep indices (from db.getTimesteps())
    n         : number of timesteps to select
    start_idx : first index into all_ts to consider
    """
    available = all_ts[start_idx:]
    if n >= len(available):
        return list(available)
    step = len(available) // n
    return [available[i * step] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 5. OUTPUT SAVING
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def save_outputs(name: str, **arrays) -> str:
    """
    Save named arrays to outputs/{name}.npz (compressed).

    For complex arrays, also saves real/imag as separate float32 keys so the
    3D Apple Vision Pro pipeline can consume them without complex dtype support.

    Returns the full path of the saved file.

    Usage
    -----
    save_outputs("task3_ocean_circulation",
                 spatial_modes=modes, depth_modes=dm, temporal_modes=tm)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.npz")

    save_dict = {}
    for key, arr in arrays.items():
        arr = np.array(arr)
        if np.iscomplexobj(arr):
            save_dict[key]              = arr.astype(np.complex64)
            save_dict[f"{key}_real"]    = arr.real.astype(np.float32)
            save_dict[f"{key}_imag"]    = arr.imag.astype(np.float32)
        else:
            save_dict[key] = arr.astype(np.float32)

    np.savez_compressed(path, **save_dict)
    size_mb = os.path.getsize(path) / 1024**2
    print(f"[save_outputs] Saved '{name}' → {path}  ({size_mb:.1f} MB)")
    print(f"  Keys: {list(save_dict.keys())}")
    return path
