# WeatherNext2-typhoon-Krovanh

Case study: how well does Google DeepMind's **WeatherNext 2** (FGN) forecast
the track of **Typhoon Krovanh (2026, JMA No. 24 / T2624)** when initialized
from **ERA5** just before the storm formed?

Only the storm centre is compared. The reference is the JMA typhoon position
table (best track).

## Status

- [x] Project skeleton
- [ ] ERA5 download and WeatherNext 2 input construction (Mac)
- [ ] Inference on RunPod H100 (uv virtual environment, no Docker)
- [ ] Storm-centre tracking with the tracker bundled in `weathernext`
- [ ] Comparison with the JMA best track (once the official CSV includes 2624)

## Experiment summary

| Item | Value |
|---|---|
| Model | WeatherNext2_<2025, checkpoint model1 (0.25 deg, 13 levels) |
| Initial conditions | ERA5 (ERA5T) from the CDS API, two frames (t-6h, t) |
| Initialization times | 2026-08-31 18 UTC (TD stage), 2026-09-01 00 UTC (TS upgrade) |
| Lead time | 240 h (40 steps of 6 h) |
| Ensemble | 8 members |
| Reference | JMA typhoon position table, UTC |

Details: [docs/requirements.md](docs/requirements.md), [docs/design.md](docs/design.md).

## Layout

```
configs/krovanh.yaml      experiment config (storm, model, cases, output policy)
src/wn2_typhoon/          package: data / inference / analysis / utils
scripts/                  thin CLI entry points, one per pipeline step
runpod/                   setup and run scripts for the H100 pod
makefiles/                make targets grouped by area (see `make help`)
docs/                     requirements, design, license notes
tests/                    unit tests
data/, outputs/           local data and results (git-ignored)
```

## Quick start

Python 3.12 is required (a `weathernext` dependency, `gdm-xarray-jax`, needs >= 3.12);
`uv` picks it up from `.python-version`.

```bash
# Mac
uv sync --group analysis --group dev
make help

# RunPod H100 (Linux)
bash runpod/setup.sh
make smoke-gpu
```

## Data sources and attribution

* WeatherNext 2 model and code: Google DeepMind, https://github.com/google-deepmind/weathernext
* ERA5: Copernicus Climate Change Service (C3S), via the Climate Data Store
* Best track: Japan Meteorological Agency, https://www.data.jma.go.jp/typhoon/position_table/

## License

Not yet decided. See [docs/license-notes.md](docs/license-notes.md).
