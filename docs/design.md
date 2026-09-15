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
CDS (ERA5T) --download_era5--> data/raw/era5/<case>/*.nc
            --prepare_inputs--> data/interim/<case>/inputs.nc      (~0.8 GiB)
            --push to pod----> /workspace/.../data/interim/<case>/
pod: run_inference (per member) --> run_tracker --> outputs/<case>/
            --pull to Mac----> outputs/<case>/{tracks.csv, member-XX/*.zarr}
Mac: fetch_besttrack --> data/raw/jma/table2026.csv
     evaluate        --> outputs/<case>/{errors.csv, figures/}
```

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

`make smoke-cpu` runs the whole inference path on the Mac with the 1 deg
`WeatherNextCyclones_Mini_<2024` checkpoint and the public sample forecast:
input contract, sample validation, input extraction, rollout, and the tracker
in both seeded and cyclogenesis-only modes. On an M5 with 32 GiB it takes well
under a minute for two 6 h steps, which makes it a cheap gate before the pod.

It checks plumbing, not skill. The Mini checkpoint is a different model at a
different resolution, and it takes 17 inputs rather than WeatherNext2's 19.

## Memory and storage

Calibration from the public 0.25 deg sample file (32 frames, 12.5 GiB):
about 0.4 GiB per frame for all variables. Hence:

| Item | Size |
|---|---|
| One checkpoint | 0.7 GiB |
| ERA5 input per case (2 frames) | ~0.8 GiB |
| Full forecast per member (40 steps) | ~16 GiB, kept in RAM only |
| Retained per case (8 members: tracks, global surface vars, regional crop) | ~12 GiB uncompressed |
| uv venv + cache on the volume | ~10 GiB |
| Total for 2 cases | ~40 GiB |

Members are generated one at a time (rollout -> tracker -> save subsets ->
free) so host RAM stays around 20 GiB.

**Network volume: 100 GB.** That is the ~40 GiB above plus room for two or
three extra initialization times at ~12 GiB each. Attach it when creating the
pod; everything that must survive a pod restart lives under its mount point.

## RunPod layout

Everything under the volume mount so a stopped pod keeps state:

```
/workspace/
  WeatherNext2-typhoon-Krovanh/   # git clone, .venv inside
  .uv-cache/                      # UV_CACHE_DIR
  .uv-python/                     # UV_PYTHON_INSTALL_DIR
  weights/                        # checkpoint cache
```

Pods are created per run, so the `runpod` alias in `~/.ssh/config` is rewritten
each time a new pod comes up; `RUNPOD_HOST` in `.env` selects the alias.

Open item: ssh.runpod.io is a proxy without scp/rsync support; file transfer
uses the pod's direct TCP port or tar over ssh (runpod/sync.sh).
