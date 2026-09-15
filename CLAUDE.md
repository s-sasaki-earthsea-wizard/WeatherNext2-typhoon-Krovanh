# CLAUDE.md

Project-specific guidance for this repository. Global conventions
(language, commit style, docstrings) come from the user's global CLAUDE.md.

## What this project is

Forecast Typhoon Krovanh (2026, JMA 2624) with WeatherNext 2 (FGN) from ERA5
initial conditions and compare the storm-centre track with the JMA best track.
Requirements: docs/requirements.md. Design: docs/design.md.

## Current phase

Phase 0 done: skeleton and initial commit (2026-09-16).
Phase 1 in progress (2026-09-16): the model input contract is derived from the
checkpoint task config (`make model-spec`), the whole inference path runs on
the Mac CPU with the 1 deg Mini checkpoint (`make smoke-mini`), and the pod
decisions are settled (docs/requirements.md 9-13). Remaining before the pod:
ERA5 download, input construction, and turning run_inference/run_tracker into
the per-member loop the pod needs. The current `rollout.predict` collects all
members in memory, which is fine for the Mini smoke run but would need 137 GB
at 0.25 deg.
Phase 2: inference + tracking for the two cases.
Phase 3: comparison against the preliminary JMA table (T2624.pdf, available
now, readable with `pdftotext -layout`), re-run against the post-analysis CSV
when it reaches storm 2624. Measured 2026-09-16 the CSV stops at 2605, about
3.7 months behind, so expect 2624 around the turn of the year.

## Hard constraints

* Python 3.12 only: gdm-xarray-jax (pulled in by weathernext) requires >= 3.12,
  and the lock is resolved for 3.12. Do not lower it.
* RunPod has no Docker: environments are uv venvs. Keep the `gpu` dependency
  group Linux-only so the Mac lock never pulls CUDA wheels.
* Do not hardcode model variable lists; read them from the checkpoint task
  config and validate ERA5 inputs against it before spending GPU time. The
  classification lives in `src/wn2_typhoon/model_spec.py`; the only table there
  is the model-name to CDS-name mapping, and an unmapped input is a hard error.
* `attention_type` in every bundled config is `splash_mha`, a TPU Pallas
  kernel. CPU and GPU runs must override it to `triblockdiag_mha`; see
  `ATTENTION_TYPE_BY_BACKEND`. The upstream demo only covers the GPU case.
* Do not install `colabtools`. `weathernext` lists it but never imports it, and
  the PyPI project of that name is an unrelated third-party package. It is
  dropped via `override-dependencies` in pyproject.toml.
* Never store full global forecast fields (17.1 GB per member). Tracking runs
  on the global field in memory; only the regional crop in the config
  (`output.region`) and the track table are written. No global subset is kept.
* The NaN target template is built on the pod from the input coordinates, never
  shipped in the input file: 40 steps of it is another 17 GB.
* All timestamps in code and data are UTC. Convert to JST only in prose.
* ERA5 comes from the CDS API (ERA5T); ARCO-ERA5 lags by months.
* Reuse the tracker bundled in `weathernext.cyclones`; do not write one.
  Configure it through `TRACKER_OVERRIDES` in `inference/tracker.py`, and never
  pass `initial_storms_df=None` -- see the upstream quirks table in
  docs/design.md.

## Conventions specific to this repo

* One YAML config (configs/krovanh.yaml); a "case" is one init time.
  Adding an init time means appending to `cases`, nothing else.
* Scripts in scripts/ are thin: parse args, load config, call the package.
* make targets live in makefiles/*.mk with `## help` comments; keep them in
  sync when adding scripts.
* Session notes go to .claude-notes/ (git-ignored).
* Results live on the NAS under `$RESULTS_ROOT`, not in the working copy.
  `runpod/sync.sh pull` symlinks `outputs/<case>` at them.
* The pod's ssh alias must use an exposed TCP port; ssh.runpod.io carries no
  rsync. Rewrite the `runpod` block in ~/.ssh/config for each new pod.

## License and attribution

Apache-2.0 for our code. Model weights are CC BY 4.0 (Google DeepMind) and are
downloaded at run time, never committed. When a new checkpoint is used, add it
to the checkpoint list in NOTICE. Published figures and tracks must credit the
model, ERA5 (Copernicus) and the JMA best track. See docs/license-notes.md.
