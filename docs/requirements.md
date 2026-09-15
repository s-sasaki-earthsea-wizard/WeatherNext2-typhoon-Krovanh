# Requirements

Decisions taken on 2026-09-16. Times are UTC unless marked JST.

## Goal

Quantify how well WeatherNext 2 (FGN), initialized from ERA5, reproduces the
track of Typhoon Krovanh (2026, JMA number 2624). Only the storm centre is
compared; no field-wise verification.

## Storm facts (JMA)

| Event | UTC | JST |
|---|---|---|
| TS upgrade (formation) | 2026-09-01 00 | 09-01 09 |
| Extratropical transition | 2026-09-07 00 | 09-07 09 |
| Dissipation | 2026-09-10 00 | 09-10 09 |

Formation position 22.6N 131.9E, 996 hPa. Minimum pressure 985 hPa, peak wind 45 kt.

## Decisions

1. **Two initialization times**, run as separate cases; more may be added
   depending on results: 2026-08-31 18 UTC (TD stage, just before formation)
   and 2026-09-01 00 UTC (TS upgrade).
2. **Forecast length 240 h** (40 steps of 6 h), covering the storm through
   dissipation.
3. **Model**: WeatherNext2_<2025, checkpoint model1, 8 ensemble members
   (defaults; not yet confirmed). The checkpoint is fine-tuned for HRES
   initial conditions; the ERA5 mismatch is accepted as part of the experiment.
4. **Initial conditions**: ERA5 (ERA5T) from the CDS API. ARCO-ERA5 is months
   behind and unusable for this period.
5. **Reference track**: JMA typhoon position table, official post-analysis CSV
   only. The comparison step starts after the CSV includes storm 2624.
6. **Metrics**: great-circle position error per lead time (members, ensemble
   mean, spread), central pressure difference, genesis timing for the
   pre-formation case. Maximum wind is reference-only (10-min mean vs gridded
   maximum).
7. **Order of work**: inference and tracking first, comparison later.
8. **Repository**: public on GitHub. License deferred pending review of the
   WeatherNext weights terms (docs/license-notes.md).
9. **Compute**: RunPod H100 80 GB, uv virtual environment, no Docker.
   A network volume holds the venv, weights, inputs and outputs.
