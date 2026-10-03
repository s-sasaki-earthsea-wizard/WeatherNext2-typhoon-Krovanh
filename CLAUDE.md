# CLAUDE.md

Project-specific guidance for this repository. Global conventions
(language, commit style, docstrings) come from the user's global CLAUDE.md.

## What this project is

Forecast Typhoon Krovanh (2026, JMA 2624) with WeatherNext 2 (FGN) from ERA5
initial conditions and compare the storm-centre track with the JMA best track.
Requirements: docs/requirements.md. Design: docs/design.md.

Issue #12 repeats the five cases on two control storms, DUJUAN (2625) and
PEIPAH (2515), to tell "the model cannot hold this kind of track" from
"Krovanh was a hard case". Each storm has its own config under configs/.

## Current phase

Phase 0 done: skeleton and initial commit (2026-09-16).
Phase 1 done (2026-09-16): the input contract is derived from the checkpoint
(`make model-spec`), the pipeline runs on the Mac (`make smoke-mini`), the pod
decisions are settled (docs/requirements.md 9-13), ERA5 is downloaded and
converted with a contract check (`make download-era5-all`, `make prepare-inputs`),
and `run_inference` / `run_tracker` are real. Both were exercised end to end on
the Mac against the Mini checkpoint, which is the same code path the pod runs.
Phase 2 done (2026-09-16) for the first two cases: 8 members each, tracked,
verified on the NAS, pod destroyed. Measurements are in .claude-notes/
session05.
Phase 2b (2026-09-17): three more init times were added, 12 h before formation
and 6 and 12 h after, so the five cases sit 6 h apart across genesis. ERA5 and
inputs for all five are cached locally; the three new cases still need GPU
time.
Phase 3 done (2026-09-17) for the two cases that have output: both JMA
releases load into one schema (`make fetch-besttrack`), and `make evaluate`
writes the position, pressure, genesis and lifetime tables plus three figures
to `outputs/<storm>/<case>/analysis/`. The reference is the preliminary table;
measured 2026-09-16 the post-analysis CSV stops at 2605, about 3.7 months
behind, so expect 2624 around the turn of the year, at which point re-running
`make fetch-besttrack && make evaluate-all` picks it up with no code change.
Phase 4 done (2026-09-18): `make evaluate-all` ran over all five cases and
`make compare-cases` writes the across-case tables and figures (skill against
initialization time in both the lead-time and the valid-time view, spread
against skill, genesis, lifetime, member fates) to `outputs/comparison/`
(now `outputs/2624-krovanh/comparison/`),
which is symlinked to the NAS like the cases. Every evaluated case and the
comparison also write GeoJSON for QGIS, and `BASEMAP=osm` puts OpenStreetMap
tiles under the track maps. Findings are in docs/design.md ("Across-case
comparison"). Nothing planned remains; a -24 h case to bound genesis timing
was offered and declined, and issue #9 (other models) is open.
2026-09-25: `make evaluate` also writes one track map per member
(`analysis/members/`) and a 2x4 grid of them (`tracks-by-member.png`), with
00Z positions dotted on both tracks; see docs/design.md ("Member track
maps").
2026-10-02: control storms (issue #12, docs/requirements.md 14-15).
`configs/dujuan.yaml` copies Krovanh's design; `configs/peipah.yaml` centres
its cases on the first depression row because its tropical-storm phase was
only 42 h. Inputs for both were built on the Mac and both were queued on
one pod the same day. The reference track and the comparison are now kept
per storm, and the comparison is cut at the end of the tropical-storm phase;
Krovanh's tables are byte-identical under both changes.

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
* Never store full global forecast fields (17.1 GB per member). `stream_member`
  reduces each rollout step as it arrives to the tracker's variables globally
  plus the regional crop, so a member peaks near 3 GB. Only the crop
  (`output.region`) and the track table are written. The global cyclone fields
  are not written either: model output does not compress (ratio 1.09, unlike
  the 180x of the training targets), so it would be 21 GB per case.
* The cyclone tracker only accepts a full [0, 360) longitude grid. Re-tracking
  a stored crop goes through `tracker.pad_to_global`; on the 1 deg check that
  reproduced the global tracker's positions exactly.
* The NaN target template is built on the pod from the input coordinates, never
  shipped in the input file: 40 steps of it is another 17 GB.
* All timestamps in code and data are UTC. Convert to JST only in prose.
* ERA5 comes from the CDS API (ERA5T); ARCO-ERA5 lags by months.
* Reuse the tracker bundled in `weathernext.cyclones`; do not write one.
  Configure it through `TRACKER_OVERRIDES` in `inference/tracker.py`, and never
  pass `initial_storms_df=None` -- see the upstream quirks table in
  docs/design.md.
* Tracking is cyclogenesis-only for every case; no observed position is ever
  handed to the tracker. Seeding was measured to change nothing past lead 0,
  and it exempts a track from the 2.5-day minimum-duration filter that
  cyclogenesis tracks are subject to. `observed_position` in the config is
  reference data. See docs/requirements.md decision 12.
* `select_storm` drops every row at or before init time. That is what lets
  the still-seeded `tracks.csv` of `init-2026-09-01T00` on the NAS evaluate
  exactly like the unseeded cases (its lead-0 row is the JMA position echoed
  back); do not re-track or rewrite that file for consistency's sake.
* The comparison covers the tropical-storm phase only. Every script cuts
  the reference with `jma_besttrack.tropical_phase` at the transition row
  from `config.tropical_end`: `depression_time` for a storm that weakened
  (Krovanh), `extratropical_time` for one that went extratropical (DUJUAN,
  PEIPAH), exactly one per storm. Never score against the uncut table: the
  post-analysis runs on as grade 6 rows, and a minimum taken over it is the
  extratropical low's (PEIPAH's 980 hPa; 992 hPa while tropical).
* Never draw an ensemble-mean track on an across-case figure. The members
  split into a Japan group and a continent group and the mean runs between
  them near the observed loop on a path no member took. Any ensemble-mean
  error is reported next to the count of members within 200 km and the
  member fates (`compare.member_fate`).
* Title a map panel with `plot._panel_title`, never `axes.set_title`. With
  the top gridline labels off (as `map_axes` sets them) matplotlib 3.11 and
  cartopy 0.25 put an automatic title at y=inf and silently drop it.
* Map tiles: only OpenStreetMap's standard tiles are wired in
  (`plot.map_axes`). Send the project user agent, keep the cache under
  `data/cache/tiles/` and put the OSM credit on the figure, which
  `credit_line(source, basemap)` does. CARTO's free tiles are key-gated now
  (every tile is watermarked "API KEY REQUIRED"), so do not add them.

## Conventions specific to this repo

* One YAML config per storm (`configs/<storm>.yaml`, chosen with `CONFIG=`);
  a "case" is one init time. Adding an init time means appending to `cases`,
  nothing else. `formation_time` means the TS upgrade in every config.
* Results live under one directory per storm, `outputs/<number>-<name>/`
  (`config.storm_dir`, e.g. `outputs/2624-krovanh/`), holding its cases
  (`config.case_dir`) and its `comparison/` (`config.comparison_dir`); the
  reference track is `data/interim/besttrack-<number>.csv`
  (`config.besttrack_path`). Everything derives from the config, including
  `runpod/run_all.sh` and `runpod/sync.sh`, so pass `CONFIG=` to every make
  target and script for a storm other than Krovanh.
* Scripts in scripts/ are thin: parse args, load config, call the package.
* make targets live in makefiles/*.mk with `## help` comments; keep them in
  sync when adding scripts.
* Session notes go to .claude-notes/ (git-ignored).
* Results live on the NAS under `$RESULTS_ROOT`, not in the working copy.
  `runpod/sync.sh pull` symlinks `outputs/<storm>` at them, one link per
  storm.
* The pod's ssh alias must use an exposed TCP port; ssh.runpod.io carries no
  rsync. Rewrite the `runpod` block in ~/.ssh/config for each new pod.

## License and attribution

Apache-2.0 for our code. Model weights are CC BY 4.0 (Google DeepMind) and are
downloaded at run time, never committed. When a new checkpoint is used, add it
to the checkpoint list in NOTICE. Published figures and tracks must credit the
model, ERA5 (Copernicus) and the JMA best track. See docs/license-notes.md.
