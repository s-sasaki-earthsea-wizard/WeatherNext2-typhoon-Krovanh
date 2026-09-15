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

* Two frames at t-6h and t; 0.25 deg; 13 pressure levels.
* Variable names follow the GraphCast/GenCast convention in the sample
  datasets under gs://dm_graphcast/weathernext2/dataset/. The exact input,
  forcing and static lists are read from the checkpoint task config at
  runtime and validated against the ERA5 request before any GPU time is spent.

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
