# Design

## Environments

| Where | Does | uv groups |
|---|---|---|
| Mac | ERA5 download, input construction, best-track comparison, plots | base + analysis + dev |
| RunPod H100 (Linux) | Weight download, ensemble rollout, storm tracking | base + gpu |

The gpu group is Linux-only via environment markers so the Mac never resolves
CUDA wheels. ERA5 is downloaded on the Mac because CDS queue time should not
burn GPU hours.

## Data flow

```
Mac  CDS (ERA5T) --download_era5--> data/raw/era5/{pressure,single}_levels_<stamp>.nc
                 --prepare_inputs-> data/interim/<case>/inputs.nc   (~0.3 GB)
                 --sync.sh push---> pod:/workspace/.../data/interim/<case>/
pod  run_inference: per member, rollout (global, streamed, never stored)
                                -> track     -> outputs/<case>/tracks.csv
                                -> crop      -> outputs/<case>/member-XX.zarr
                 --sync.sh pull--> Mac:staging/<case>/   (pod stops here)
                                -> NAS:/Volumes/EW-NAS-Atoll/.../outputs/<case>/
Mac  run_tracker  --> outputs/<case>/tracks-retracked.csv   (optional, no GPU)
     fetch_besttrack --> data/raw/jma/{T2624.pdf, table2026.csv}
     evaluate        --> outputs/<case>/{errors.csv, figures/}
```

Raw ERA5 frames are named after the timestamp they hold and shared by every
case, because the two Krovanh cases have the 2026-08-31 18 UTC frame in
common. `outputs/<case>` in the working copy is a symlink to the NAS, so the
analysis steps read what the pull wrote without a second copy.

## Model input contract

Read from `config.task` of the fiddle config bundled with weathernext
(`weathernext/weathernext2/configs/<name>.json`), so it needs neither weights
nor a GPU: `make model-spec`. `src/wn2_typhoon/model_spec.py` classifies the
variables; the only table in the repository is the model-name to CDS-name
mapping, and an input that does not resolve through it is a hard error.

WeatherNext2 `<2025` asks for `input_duration: 12h` (two frames at t-6h and t)
on 13 WeatherBench pressure levels, and 19 input variables:

| Group | Count | Variables | CDS dataset |
|---|---|---|---|
| Pressure level | 6 x 13 | temperature, geopotential, u/v_component_of_wind, vertical_velocity, specific_humidity | `reanalysis-era5-pressure-levels` |
| Single level | 7 | 2m_temperature, mean_sea_level_pressure, 10m_u/v_component_of_wind, sea_surface_temperature, 100m_u/v_component_of_wind | `reanalysis-era5-single-levels` |
| Static | 2 | geopotential_at_surface (CDS name: `geopotential`), land_sea_mask | `reanalysis-era5-single-levels`, one timestamp |
| Computed | 4 | year_progress_sin/cos, day_progress_sin/cos | none; `data_utils.add_derived_vars` |

Consequences worth recording:

* **No precipitation and no solar radiation on input.** `total_precipitation_6hr`
  is a target only, and WeatherNext 2 does not use `toa_incident_solar_radiation`
  at all, so the CDS request needs no accumulated fields.
* **100 m winds are specific to WeatherNext2.** WeatherNextCyclones and its Mini
  variant take 17 inputs and omit them, which is why the list is derived per
  checkpoint rather than written down once.
* `sea_surface_temperature` is NaN over land in ERA5. That is the convention the
  model was trained with, so the NaNs are kept.
* The 31 targets include 17 `cyclone_*` fields the model predicts directly; the
  bundled direct tracker consumes those rather than deriving centres from MSLP.

## Attention implementation by backend

All three configs ship `attention_type: splash_mha` with `mask_type: lazy`,
which is the TPU Pallas kernel. The upstream demo overrides it only when the
backend is GPU, so CPU runs need the same override:

| Backend | attention_type |
|---|---|
| tpu | splash_mha (config default) |
| gpu | triblockdiag_mha (slower, more memory) |
| cpu | triblockdiag_mha |

## Working around weathernext 0.3.0

Three upstream behaviours cost time to find, so they are recorded here. All are
handled in `src/wn2_typhoon/`, none needs a fork.

| Symptom | Cause | What we do |
|---|---|---|
| `uv sync` pulls an unknown `colabtools` 0.0.1 | `install_requires` names `colabtools`, but the PyPI project of that name is not Google's, and nothing under `weathernext/` imports it | `override-dependencies` in pyproject.toml drops it |
| The model fails on a CPU backend | every config ships `attention_type: splash_mha`, a TPU Pallas kernel, and the upstream demo overrides it for GPU only | `ATTENTION_TYPE_BY_BACKEND` in `inference/load_model.py` |
| `DirectTracker(..., initial_storms_df=None)` raises `KeyError: lead_time` | the documented "pure cyclogenesis" path assigns a bare `pd.DataFrame()` and then indexes it by column name | pass a typed empty frame, `tracker.empty_initial_storms()` |
| The tracker raises `ValueError: Encountered all NA values` | `enforce_physical_consistency_on_quadrants_and_winds` calls `DataFrame.idxmax(axis=1)` expecting NaN for an all-NaN row; pandas raises from 2.1 on, and a storm seeded from an observed position has no quadrant radii at t=0 | `TRACKER_OVERRIDES` turns that step off; it only rewrites wind-radius columns, never the centre, pressure or maximum wind |

Pinning `pandas < 2.1` would restore the old `idxmax` behaviour but is not
available: the rest of the stack needs numpy 2.

## Local pipeline check

`make smoke-mini` runs the whole inference path on the Mac with the 1 deg
`WeatherNextCyclones_Mini_<2024` checkpoint and the public sample forecast:
input contract, sample validation, input extraction, rollout, and the tracker
in both seeded and cyclogenesis-only modes. On an M5 with 32 GiB it takes well
under a minute for two 6 h steps, which makes it a cheap gate before the pod.

It checks plumbing, not skill. The Mini checkpoint is a different model at a
different resolution, and it takes 17 inputs rather than WeatherNext2's 19.

## Memory and storage

A WeatherNext2 forecast frame carries 103 two-dimensional fields: 6 upper-air
variables on 13 levels plus 25 single-level targets, of which 17 are the
`cyclone_*` fields the tracker reads. At 0.25 deg that is 4.15 MB per field,
so:

| Item | Size |
|---|---|
| One checkpoint | 0.74 GB |
| ERA5 input per case (2 frames, 172 fields, compressed NetCDF) | ~0.3 GB |
| Full forecast per member (40 steps, global) | 17.1 GB, never held whole |
| Peak host memory per member (tracker fields + crop) | ~3 GB |
| Regional crop per member (15-50N, 115-150E, 141 x 141) | 0.33 GB |
| Retained per case (8 members plus tracks) | 2.6 GB |
| uv venv + cache on the volume | ~6 GB |

Members are generated one at a time and each rollout step is reduced the
moment it arrives, so a full global member never exists at once. Two things
are kept from each step:

* the 17 cyclone fields the tracker reads, globally, because the storm can be
  anywhere: 2.8 GB accumulated over the rollout
* the regional crop, every variable and level: 0.33 GB

Everything else is dropped, which holds a member near 3 GB instead of 17 GB.
After the rollout the tracker consumes the global cyclone fields in memory and
only the crop and the track table are written.

The NaN target template that sizes the rollout is **not** part of the input
file: materialising 40 steps of it would cost another 17 GB. It is built on the
pod from the input coordinates, dask-backed and chunked one step at a time,
which is enough because `rollout` slices one chunk and calls `.compute()` on
it. `tests/test_inputs.py` pins that a two-frame file plus that template splits
into exactly what the upstream helper produces from a full one.

### Why the global cyclone fields are not written

They would make re-tracking trivial, and the training targets in the public
sample are 94 per cent NaN and 99.6 per cent exact zero, which compresses about
180-fold. Model output does not behave that way: it is dense small values
everywhere, measured at a compression ratio of 1.09, so storing it would cost
2.6 GB per member and 21 GB per case. The crop already carries all 17 cyclone
fields, so `run_tracker` pads it back onto a global grid with zeros instead --
the tracker rejects any grid that is not the full [0, 360) in longitude. On the
1 deg check that reproduced the global tracker's positions exactly, which holds
as long as the storm stays well inside the region.

**Network volume: 30 GB.** Worst case on the volume is the venv and cache
(~6 GB), Python (0.1 GB), one checkpoint (0.74 GB), both cases' inputs
(1.4 GB) and one case of output (2.6 GB): under 11 GB, with the rest as
headroom for a second case in flight or an extra checkpoint. Results are
pulled to the NAS after each case rather than accumulating.

Keep `UV_CACHE_DIR` on the same filesystem as the venv so uv hardlinks wheels
instead of copying them; the CUDA wheels are the bulk of the 6 GB and paying
for them twice would be wasteful. `uv cache prune` after a successful sync
reclaims what is no longer referenced.

## RunPod layout

Everything under the volume mount so a stopped pod keeps state:

```
/workspace/
  WeatherNext2-typhoon-Krovanh/   # git clone, .venv inside
  .uv-cache/                      # UV_CACHE_DIR, same filesystem for hardlinks
  .uv-python/                     # UV_PYTHON_INSTALL_DIR
  weights/                        # checkpoint cache
```

Pods are created per run, so the ssh alias in `~/.ssh/config` is rewritten each
time a new pod comes up; `RUNPOD_HOST` selects which alias. The top-level
Makefile reads `.env` before the `?=` defaults in `makefiles/*.mk`, so setting
it there is enough and no target needs it on the command line.

## Moving data

The `ssh.runpod.io` proxy carries interactive sessions but not scp or rsync, so
the pod is created with **ssh over an exposed TCP port** and the alias points
at that host and port directly. `runpod/sync.sh` then uses plain rsync:

```
push:         data/interim/<case>/     -> pod:.../data/interim/<case>/
pull-stage:   pod:.../outputs/<case>/  -> staging/<case>/
pull-publish: staging/<case>/          -> $RESULTS_ROOT/outputs/<case>/
pull:         both pull legs, in order
```

The pull is two legs because `RESULTS_ROOT` is an SMB mount on the NAS reached
over Tailscale, and one rsync from the pod to the NAS keeps the pod rented for
the whole of that leg while it computes nothing. The first leg lands the case
-- 2.3 GB in 5002 files, 458 KB on average, measured 2026-09-16 -- on the Mac's
SSD, after which the pod can be stopped; the second leg runs between the Mac
and the NAS with the pod already off. Whether the split is faster end to end is
not measured (#5); what it changes for certain is how long the pod is billed.

Staging is `staging/<case>/` in the working copy, git-ignored. It is there
rather than under `/tmp` so that a `RESULTS_ROOT` which is itself local is on
the same filesystem, keeping the second leg off the network entirely, and so
that nothing outside this repository can delete a case that is waiting to be
published. The second leg is rsync too, not `mv`: it resumes, `--append-verify`
checks what it skips, and it merges into a destination that already holds an
earlier pull. The staged copy is removed only once that rsync has returned
success, so a second leg that fails -- share unmounted, Tailscale down --
leaves the case on the SSD and `make runpod-pull-publish CASE=...` finishes it
without a pod. The writability walk over `RESULTS_ROOT`'s ancestors therefore
gates the second leg alone; an unmounted NAS must not block the leg that is
costing money.

`RESULTS_ROOT` defaults to `/Volumes/EW-NAS-Atoll/Projects/personal-dev/
WeatherNext2-typhoon-Krovanh` on the NAS. The working copy's `outputs/` is a
symlink to it, so the analysis targets on the Mac read the pulled results in
place with no second copy. The second leg creates that symlink once it
succeeds, always pointing at `RESULTS_ROOT` and never at the staging copy,
which is gone by then. Nothing breaks when the NAS is unmounted beyond the
analysis step failing to find its input.

## Storm-centre reference

The comparison needs the JMA position table, which comes in two forms with very
different schedules.

| | Preliminary (T2624.pdf) | Post-analysis (table2026.csv) |
|---|---|---|
| Available | now | around the turn of the year |
| Coverage | formation to loss of typhoon status | whole life including the depression stage |
| Rows for 2624 | 49 | none yet |
| Cadence | 3-hourly JST | 6-hourly UTC |
| Maximum wind | m/s | knots, 0 below the 34 kt threshold |
| Grade code | absent | present |
| Parsing | `pypdf` layout mode | CSV, Shift_JIS |

`data/jma_besttrack.py` reads both into one schema -- time in UTC, lat, lon,
`pressure_hpa`, `wind_kt`, `grade`, `remark`, `source` -- so the comparison can
start on the preliminary values and be re-run against the CSV without any
change downstream. `make fetch-besttrack` tries the CSV, explains what it found
instead, and falls back; when 2624 appears the same command picks it up.

Two details the schema has to reconcile. The CSV writes a maximum wind of 0
below 34 kt, which is a threshold rather than a measurement, so it is read as
missing. And the two files disagree on time resolution, which a merge would
silently turn into an empty join, so both loaders normalise to `datetime64[ns]`.

Wind radii are not parsed. The releases describe them differently -- storm- and
gale-force radii by compass direction against 50 kt and 30 kt ellipse axes --
so they would not survive the swap, and this project compares centres.

Measured on 2026-09-16 the CSV stops at storm 2605 (19 May 2026) although the
file itself was re-published on 2026-09-09, so the post-analysis lags roughly
3.7 months. The comparison is therefore built against the preliminary values,
labelled as preliminary, and re-run against the CSV when 2624 appears. Only the
CSV carries a position before formation, which is why the pre-formation case
cannot be seeded today.

### The analysis is not the truth either

ERA5 and the JMA analysis do not put the storm in the same place, and the gap
is a floor on any forecast error worth reporting. At the 2026-09-01 00 UTC
initialization the downloaded ERA5 has its mean sea level pressure minimum at
22.75N 131.75E and 998.4 hPa, against the JMA 22.6N 131.9E and 996 hPa: about
20 km apart and 2.4 hPa shallower, with a maximum 10 m wind of 15.8 m/s against
the reported 18 m/s. Under-deepening tropical cyclones is a known property of
0.25 deg reanalysis. Position errors of this order at short lead times say more
about the initial state than about the model.
