# CLAUDE.md

Project-specific guidance for this repository. Global conventions
(language, commit style, docstrings) come from the user's global CLAUDE.md.

## What this project is

Forecast Typhoon Krovanh (2026, JMA 2624) with WeatherNext 2 (FGN) from ERA5
initial conditions and compare the storm-centre track with the JMA best track.
Requirements: docs/requirements.md. Design: docs/design.md.

## Current phase

Phase 0 done: skeleton and initial commit (2026-09-16).
Phase 1 next: ERA5 download + input construction on the Mac, then environment
setup and a smoke run on the RunPod H100.
Phase 2: inference + tracking for the two cases.
Phase 3: comparison, once the JMA post-analysis CSV includes storm 2624.

## Hard constraints

* Python 3.12 only: gdm-xarray-jax (pulled in by weathernext) requires >= 3.12,
  and the lock is resolved for 3.12. Do not lower it.
* RunPod has no Docker: environments are uv venvs. Keep the `gpu` dependency
  group Linux-only so the Mac lock never pulls CUDA wheels.
* Do not hardcode model variable lists; read them from the checkpoint task
  config and validate ERA5 inputs against it before spending GPU time.
* Never store full global forecast fields (~16 GiB per member). Keep tracks,
  global surface variables, and the regional crop defined in the config.
* All timestamps in code and data are UTC. Convert to JST only in prose.
* ERA5 comes from the CDS API (ERA5T); ARCO-ERA5 lags by months.
* Reuse the tracker bundled in `weathernext.cyclones`; do not write one.

## Conventions specific to this repo

* One YAML config (configs/krovanh.yaml); a "case" is one init time.
  Adding an init time means appending to `cases`, nothing else.
* Scripts in scripts/ are thin: parse args, load config, call the package.
* make targets live in makefiles/*.mk with `## help` comments; keep them in
  sync when adding scripts.
* Session notes go to .claude-notes/ (git-ignored).
