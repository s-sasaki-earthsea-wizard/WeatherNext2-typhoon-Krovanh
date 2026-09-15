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

## Model and data provenance

Forecasts here are produced by a model this repository does not contain. The
weights are downloaded at run time from Google DeepMind's public bucket and are
used unmodified.

| What | Source | Terms |
|---|---|---|
| WeatherNext 2 code | [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext) v0.3.0 | Apache-2.0 |
| Checkpoint `WeatherNext2_<2025_model1.npz` | `gs://dm_graphcast/weathernext2/params/` | CC BY 4.0, (c) Google DeepMind |
| ERA5 / ERA5T initial conditions | Copernicus Climate Data Store | Copernicus licence; contains modified Copernicus Climate Change Service information 2026 |
| Typhoon best track | [JMA typhoon position table](https://www.data.jma.go.jp/typhoon/position_table/) | JMA website terms of use |

Any forecast figure or track file derived from those weights carries the CC BY
4.0 attribution to Google DeepMind. Neither the European Commission nor ECMWF
is responsible for any use made of the Copernicus information above.

## License

Apache-2.0 for the code in this repository; see [LICENSE](LICENSE) and
[NOTICE](NOTICE). Third-party terms that apply to the model weights and input
data are summarized above and in [docs/license-notes.md](docs/license-notes.md).
