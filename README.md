# WeatherNext2-typhoon-Krovanh

Case study: how well does Google DeepMind's **WeatherNext 2** (FGN) forecast
the track of **Typhoon Krovanh (2026, JMA No. 24 / T2624)** when initialized
from **ERA5** just before the storm formed?

Only the storm centre is compared. The reference is the JMA typhoon position
table (best track).

## Status

- [x] Project skeleton
- [x] Model input contract derived from the checkpoint config (`make model-spec`)
- [x] Local pipeline check, model to tracker (`make smoke-mini`)
- [x] ERA5 download and WeatherNext 2 input construction (Mac)
- [x] Ensemble rollout and storm-centre tracking, run end to end on the Mac
      against the 1 deg checkpoint
- [x] Inference on RunPod H100 (uv virtual environment, no Docker): five
      initialization times, 8 members each
- [x] JMA reference track, preliminary and post-analysis read into one schema
      (`make fetch-besttrack`)
- [x] Comparison of forecast tracks against it, per case (`make evaluate-all`)
- [x] Comparison across initialization times, spread against skill, genesis
      and lifetime (`make compare-cases`)
- [x] Track maps on OpenStreetMap tiles (`BASEMAP=osm`) and GeoJSON export
      of every track for QGIS
- [ ] Re-run against the JMA post-analysis table when storm 2624 appears in it

## Experiment summary

| Item | Value |
|---|---|
| Model | WeatherNext2_<2025, checkpoint model1 (0.25 deg, 13 levels) |
| Initial conditions | ERA5 (ERA5T) from the CDS API, two frames (t-6h, t) |
| Initialization times | five, 6 h apart: 2026-08-31 12 and 18 UTC (before formation), 2026-09-01 00 UTC (formation), 06 and 12 UTC (after) |
| Lead time | 240 h (40 steps of 6 h) |
| Ensemble | 8 members |
| Compute | RunPod H100 80 GB, 30 GB volume, results pulled to the NAS |
| Stored output | regional crop 15-50N 115-150E, all levels (2.6 GB per case) |
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
make setup          # uv sync --group cpu --group analysis --group dev
make help

# RunPod H100 (Linux)
bash runpod/setup.sh
make smoke-gpu
```

### What the model wants as input

The input variable list is never written down in this repository: it is read
from the task config of the checkpoint itself, which ships inside the
`weathernext` package, so it needs no weights, GPU or network access.

```bash
make model-spec     # 6 pressure-level + 7 single-level + 2 static + 4 computed
```

WeatherNext2 takes 19 inputs on 13 pressure levels over two frames (t-6h, t).
Notably it needs no precipitation and no solar radiation on input, and it is
the only bundled checkpoint that asks for 100 m winds. See
[docs/design.md](docs/design.md).

### Checking the pipeline before paying for a GPU

```bash
make smoke-mini     # 1 deg Mini checkpoint, two 6 h steps, under a minute
```

This runs the real path end to end -- config, the public sample forecast,
input extraction, rollout, and the cyclone tracker both seeded and in
cyclogenesis mode -- on a checkpoint small enough for a laptop. It validates
the plumbing, not the science: WeatherNext2 at 0.25 deg needs an H100. The
same target run on the pod exercises the GPU attention path, which is why it
is not called `smoke-cpu`.

### Getting the inputs

```bash
make download-era5-all   # 7 CDS requests, ~0.45 GB, shared across both cases
make prepare-inputs CASE=init-2026-08-31T18
```

The download asks for one frame per request: the CDS expands year/month/day/time
as a cross product, so a pair of frames straddling midnight would otherwise
return four. Frames are cached by timestamp and the two cases share the
2026-08-31 18 UTC one. `prepare_inputs` checks the result against the
checkpoint's contract -- variable set, ascending latitude, longitudes in
[0, 360), level order, frame spacing, which fields may contain NaN -- and
refuses to write a file that would only fail once the weights were loaded.

### The reference track

```bash
make fetch-besttrack    # post-analysis if it is out, preliminary otherwise
```

The JMA publishes each storm twice. The post-analysis CSV is the archival
record but lags the season by about 3.7 months, so storm 2624 is not expected
before the turn of the year; the preliminary PDF is out now. Both are read into
one schema, so the comparison starts on the preliminary values and is re-run
against the CSV later without changing anything downstream.

### Results on the NAS

Forecast output is pulled off the pod to `$RESULTS_ROOT` (the NAS by default,
see `.env.example`) rather than accumulating on the billed volume.
`runpod/sync.sh pull` symlinks `outputs/<case>` at the pulled directory, so the
analysis targets run in the working copy without a second copy of the data.

### Comparing the forecasts with the reference

```bash
make evaluate CASE=init-2026-09-01T00   # one case
make evaluate-all                       # every case with tracker output
make compare-cases                      # the cases side by side
make evaluate-all BASEMAP=osm           # track maps on OpenStreetMap tiles
```

`evaluate` reduces a case's tracker output to one track per member (the
tracker reports every storm on the globe), scores it against the JMA track,
and writes six tables, four figures and two GeoJSON files to
`outputs/<case>/analysis/`, plus one track map per member under `members/`.
The ensemble track map (`tracks.png`) puts every member on one map; the member
maps draw one member at a time over the ensemble mean and the JMA track, with
both tracks dotted and dated at 00Z so a member that is on the right path but
late shows it, and `tracks-by-member.png` lays them out four rows by two.
`compare-cases` reads those tables for every
evaluated case and writes `outputs/comparison/`: error against lead time and
against valid time, spread against skill, genesis timing, track end and
minimum pressure per member, and where each member's track ended, plus small
multiples of the tracks and one GeoJSON pair holding every case.

The GeoJSON files (`tracks-lines.geojson`, `tracks-points.geojson`) drop
straight onto a QGIS map: one line per member plus the reference, and one
point per 6 h position with its lead time, pressure and wind. Lines and points
are separate files because QGIS splits a mixed-geometry file into sub-layers
and asks which to load.

`BASEMAP=osm` needs the network on the first draw; tiles are cached under
`data/cache/tiles/` and the figure's credit line gains the OpenStreetMap
attribution. The default Natural Earth rendering is the cleaner one for print.

## Model and data provenance

Forecasts here are produced by a model this repository does not contain. The
weights are downloaded at run time from Google DeepMind's public bucket and are
used unmodified.

| What | Source | Terms |
|---|---|---|
| WeatherNext 2 code | [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext) v0.3.0 | Apache-2.0 |
| Checkpoint `WeatherNext2_<2025_model1.npz` | `gs://dm_graphcast/weathernext2/params/` | CC BY 4.0, (c) Google DeepMind |
| Checkpoint `WeatherNextCyclones_Mini_<2024.npz` (local pipeline check only) | `gs://dm_graphcast/weathernext2/params/` | CC BY 4.0, (c) Google DeepMind |
| ERA5 / ERA5T initial conditions | Copernicus Climate Data Store | Copernicus licence; contains modified Copernicus Climate Change Service information 2026 |
| Typhoon best track | [JMA typhoon position table](https://www.data.jma.go.jp/typhoon/position_table/) | JMA website terms of use |
| Map tiles (optional, `BASEMAP=osm`) | [OpenStreetMap](https://www.openstreetmap.org) standard tiles | (c) OpenStreetMap contributors, ODbL; [tile usage policy](https://operations.osmfoundation.org/policies/tiles/) |

Any forecast figure or track file derived from those weights carries the CC BY
4.0 attribution to Google DeepMind. Neither the European Commission nor ECMWF
is responsible for any use made of the Copernicus information above.

## License

Apache-2.0 for the code in this repository; see [LICENSE](LICENSE) and
[NOTICE](NOTICE). Third-party terms that apply to the model weights and input
data are summarized above and in [docs/license-notes.md](docs/license-notes.md).
