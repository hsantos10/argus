# LLC4320 ocean backend

FastAPI service that streams raw sub-volumes of the SciVis 2026 ECCO LLC4320
dataset to the Three.js GPU globe. Wraps `openvisuspy` and the OSDF Pelican
endpoints from <https://sciviscontest2026.github.io/examples/#access-llc4320>.

---

## Prerequisites

- Python environment: `conda env create -f environment.yml && conda activate argus` (from the repo root)
- The lat/lon grid file described below

---

## Required data file

> **Download `llc4320_latlon.nc`:**
> <https://drive.google.com/file/d/18-gKsEAtoL6vXlhT1UpkZsrL2M3g9E2R/view?usp=sharing>

This is the LLC4320 lat/lon grid (~158 MB). It is **not tracked in git**.
Place the downloaded file at `backend/llc4320_latlon.nc` (i.e. the same
folder as `main.py`).

The backend will start without it but the `/api/latlut` endpoint — which the
GPU globe calls on first load — will return HTTP 503 until the file is present.

---

## Run

```sh
# From the backend/ directory
uvicorn main:app --host 127.0.0.1 --port 8000

# Or from the repo root
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/datasets` | `[{key, label, url}]` for `theta`, `salt`, `w` |
| `GET` | `/api/metadata/{key}` | Dims, timestep list, time info (T0, dt) |
| `GET` | `/api/latlut` | 1-D lat→row LUT (float32 binary) — requires `llc4320_latlon.nc` |
| `GET` | `/api/globe_volume/{key}` | Depth-block uint8 volume for GPU rendering |
| `GET` | `/api/globe/{key}` | Single lat×lon depth slice (float32 binary) |
| `GET` | `/api/volume/{key}` | Arbitrary sub-volume (float32 binary) |

### Key query parameters

- `time` — integer timestep index (0–10311 for ECCO; dataset covers Sep 2011 – Nov 2012 hourly)
- `quality` — OpenVisus resolution offset: `0` = full res, `-15` = very coarse. `-8` is a good interactive default.
- `z0`, `z1` — depth layer range (0–89)

Response headers for binary endpoints: `X-Shape` (numpy shape), `X-Min`, `X-Max`, `X-Z0`, `X-Z1`.

### Quality / resolution reference

| `quality` | Approx. spatial resolution | Good for |
|-----------|---------------------------|----------|
| `-15` | ~83 × 52 px | Fastest exploration |
| `-12` | ~333 × 208 px | Low quality |
| `-8` | ~1080 × 833 px | Interactive (default) |
| `-4` | ~1440 × 1620 px* | High detail |

*At quality `-4` the raw data exceeds WebGL2's 3D texture limit (2048 px/axis).
The backend auto-strides down to fit — effective resolution is still higher than `-8`.

---

## Smoke test

```sh
curl http://localhost:8000/api/health
curl http://localhost:8000/api/datasets
curl -D - http://localhost:8000/api/metadata/theta
curl -D - -o /tmp/vol.bin \
  "http://localhost:8000/api/globe_volume/theta?time=0&quality=-8&z0=0&z1=15"
```
