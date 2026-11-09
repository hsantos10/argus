"""
query_runner.py - load a SciViz contest dataset, render one PNG, write bytes to stdout.

Usage: stdin = JSON params, stdout = raw PNG bytes, stderr = PROGRESS lines.
The calling Node route handler caches output by input hash.

Thin dispatch entry point. The actual query/render logic lives in the qr/ package:
  qr.common  — shared constants, helpers, matplotlib render functions
  qr.dyamond — DYAMOND GEOS atmosphere/ocean/coupled queries
  qr.ecco    — ECCO LLC4320 ocean query
  qr.cmip6   — NEX-GDDP CMIP6 query
  qr.dmd     — animation + DMD (precomputed and live) queries
"""
import os
import sys
import traceback

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))   # web/scripts/
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, "../.."))  # repo root
sys.path.insert(0, _SCRIPTS_DIR)   # for qr.*
sys.path.insert(0, _REPO_ROOT)     # for analysis.*, utils, etc.

from qr.common import _parse_params, _progress
from qr.dyamond import (
    query_dyamond, query_dyamond_ocean, query_dyamond_coupled,
    query_face_z_sweep, query_face_cross_section, query_face_all_panel,
)
from qr.ecco import query_ecco
from qr.cmip6 import query_cmip6
from qr.dmd import query_animate, query_dmd, query_dmd_live


def main():
    # The only thing allowed on real stdout is the final binary PNG. Dataset
    # loaders and libraries print() diagnostics to stdout, which would corrupt a
    # single-image response or — on multi-event (dmd_live/animate/faces) runs —
    # make stdout non-empty and trick the Node layer into emitting a bogus
    # trailing image. Redirect stray stdout to stderr for the whole query.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        p = _parse_params()
        dataset = p.get("dataset")
        _progress(f"Dispatching dataset={dataset}")
        if dataset == "dyamond":
            if bool(p.get("ocean", False)):
                png = query_dyamond_ocean(p)
            elif bool(p.get("coupled", False)):
                png = query_dyamond_coupled(p)
            elif bool(p.get("animate", False)):
                png = query_animate(p)
            elif bool(p.get("z_sweep", False)):
                png = query_face_z_sweep(p)
            elif bool(p.get("cross_section", False)):
                png = query_face_cross_section(p)
            elif bool(p.get("all_panel", False)):
                png = query_face_all_panel(p)
            else:
                png = query_dyamond(p)
        elif dataset == "ecco":
            if bool(p.get("animate", False)):
                png = query_animate(p)
            else:
                png = query_ecco(p)
        elif dataset == "cmip6":
            if bool(p.get("animate", False)):
                png = query_animate(p)
            else:
                png = query_cmip6(p)
        elif dataset == "dmd":
            png = query_dmd(p)
        elif dataset == "dmd_live":
            png = query_dmd_live(p)
        else:
            raise ValueError(f"Unknown dataset: {dataset!r}")
        _progress(f"Writing {len(png)} bytes to stdout")
        buf = getattr(real_stdout, "buffer", None)
        if buf is not None:
            buf.write(png)
        else:
            real_stdout.write(png)
        real_stdout.flush()
    except Exception as e:
        # Intentional RuntimeErrors carry a clean, user-facing message (e.g.
        # "CMIP6 data source unreachable", "Fetch aborted") — surface just that.
        # Unexpected errors keep the full traceback for debugging.
        if isinstance(e, RuntimeError):
            sys.stderr.write(str(e))
        else:
            sys.stderr.write(f"{e}\n{traceback.format_exc()}")
        sys.stderr.flush()
        # os._exit bypasses the normal shutdown path and kills non-daemon
        # background threads immediately (e.g. stuck ov.LoadDataset calls).
        import os as _os
        _os._exit(1)
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    main()
    # OpenVisus and Pelican clients spin up non-daemon threads that survive
    # past main() and prevent normal Python exit. Force termination so the
    # Node streaming layer sees the process close and doesn't hit its timeout.
    import os as _os
    _os._exit(0)
