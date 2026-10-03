# Requirements

Decisions taken on 2026-09-16. Times are UTC unless marked JST.

## Goal

Quantify how well WeatherNext 2 (FGN), initialized from ERA5, reproduces the
track of Typhoon Krovanh (2026, JMA number 2624). Only the storm centre is
compared; no field-wise verification.

## Storm facts (JMA)

From the preliminary position table T2624.pdf.

| Event | UTC | JST | Position |
|---|---|---|---|
| TS upgrade (formation) | 2026-09-01 00 | 09-01 09 | 22.6N 131.9E, 996 hPa |
| Minimum pressure | 2026-09-03 00 | 09-03 09 | 26.3N 130.5E, 985 hPa, 23 m/s |
| Weakened to a depression | 2026-09-07 00 | 09-07 09 | 26.0N 132.0E |
| Dissipation | 2026-09-10 00 | 09-10 09 | press reports, not in the table |

Krovanh weakened back into a tropical depression; it did not undergo
extratropical transition, and an earlier note to that effect was wrong. The
storm never had a storm-force wind radius, and the whole observed track fits
in 22.3-29.8N, 126.5-132.6E: it looped around the Nansei islands rather than
recurving away.

## Decisions

1. **Five initialization times**, run as separate cases, 6 h apart and
   spanning 12 h either side of formation: 2026-08-31 12 and 18 UTC (TD
   stage), 2026-09-01 00 UTC (TS upgrade), and 2026-09-01 06 and 12 UTC.
   The first two were run on 2026-09-16; the other three were added on
   2026-09-17 once those results were in hand. Two points could show that
   skill depends on the initialization, but not how, and they could not show
   genesis timing at all: in cyclogenesis mode the tracker cannot report a
   storm before the first rollout step, so the 18 UTC case can only ever
   place genesis at or after the observed time. An initialization 12 h ahead
   is the shortest one that can resolve an early genesis.
2. **Forecast length 240 h** (40 steps of 6 h), covering the storm through
   dissipation.
3. **Model**: WeatherNext2_<2025, checkpoint model1, 8 ensemble members.
   Confirmed as the starting configuration; other checkpoints, the
   WeatherNextCyclones variants or a larger ensemble are only tried if the
   results are poor. The checkpoint is fine-tuned for HRES initial conditions;
   the ERA5 mismatch is accepted as part of the experiment.
4. **Initial conditions**: ERA5 (ERA5T) from the CDS API. ARCO-ERA5 is months
   behind and unusable for this period.
5. **Reference track**: JMA typhoon position table. The official post-analysis
   CSV is the target, but measured on 2026-09-16 it reaches only storm 2605
   (19 May 2026) while being re-published as recently as 2026-09-09, so the
   analysis itself runs about 3.7 months behind. Storm 2624 should appear
   around the turn of the year. The preliminary table (T2624.pdf) is available
   now, is machine readable with `pdftotext -layout`, and covers formation
   (1 Sep 09 JST) to the return to depression status (7 Sep 09 JST) at 3-hourly
   JST steps. Decision: build the comparison against the preliminary values,
   labelled as such, and swap in the CSV when it lands. The two differ in
   units (m/s against knots) and in coverage: only the CSV carries the
   depression stage before formation.
6. **Metrics**: great-circle position error per lead time (members, ensemble
   mean, spread), central pressure difference, genesis timing for the
   pre-formation case. Maximum wind is reference-only (10-min mean vs gridded
   maximum).
7. **Order of work**: inference and tracking first, comparison later.
8. **Repository**: public on GitHub, Apache-2.0. The model weights are
   downloaded at run time and never vendored here; their CC BY 4.0
   attribution is recorded in NOTICE and the README (docs/license-notes.md).
9. **Compute**: RunPod H100 80 GB PCIe (SXM only if throughput demands it),
   uv virtual environment, no Docker. A **30 GB network volume** holds the
   venv, uv cache, weights, inputs and the case being worked on; the sizing is
   derived in docs/design.md. Results are pulled to the NAS after each case
   rather than accumulating on the volume, so the volume never has to hold
   more than one case. The pod is created per run with ssh over an exposed TCP
   port (the ssh.runpod.io proxy supports no rsync), and the ssh alias in
   `~/.ssh/config` is rewritten each time.
10. **Stored output**: the regional crop only, at 15-50N 115-150E for Krovanh
    (the control storms carry their own, decision 14), all
    variables and all 13 levels. Global fields exist only in memory during
    tracking. The crop covers Taiwan, the Chinese coast, the East China Sea,
    the Nansei islands and the Japanese archipelago; the observed track fits
    well inside it. No global surface subset is kept: it is needed for the
    forecast, not for the question being asked.
11. **Results storage**: outputs live on the NAS at
    `/Volumes/EW-NAS-Atoll/Projects/personal-dev/WeatherNext2-typhoon-Krovanh/`,
    one directory per storm (`outputs/2624-krovanh/`, decided 2026-10-03 when
    the control storms arrived), each symlinked into `outputs/` of the
    working copy so analysis on the Mac needs no copying.
12. **Tracker seeding: none, in any case.** Every case is tracked in
    cyclogenesis mode, so no observed position ever enters a forecast track.
    The earlier decision seeded the formation case with the JMA centre; that
    was reversed on 2026-09-17 after measuring what the seed actually did.

    Re-tracking `init-2026-09-01T00` from the stored crop twice, once seeded
    and once not, gives **identical tracks past lead 0**: 0.0 km and 0.00 hPa
    at every step of all 8 members. Cyclogenesis mode also found the storm
    unaided in 8 of 8 members, 36-110 km from the JMA position, with the
    nearest other storm about 1640 km away. The seed contributed only its own
    lead-0 row, which is the JMA position echoed back with unit probability
    and no pressure, and which has to be excluded from any error curve.

    Two reasons to drop it rather than keep something harmless. The tracker
    discards cyclogenesis tracks shorter than 2.5 days but never discards a
    seeded track, so seeding the post-formation cases and not the others
    would censor the two groups differently, and the weakening question is
    precisely about short-lived storms. And seeding only from formation
    onwards would put a mode change at the boundary the five cases exist to
    measure, confounding a better analysis with being handed the answer.

    `observed_position` stays in the config as reference data: it is what the
    initial-state error is quoted against. `scripts/run_tracker.py
    --seed-position` re-runs a seeded track from the stored crop for anyone
    who wants to repeat the check, and `--keep-short-tracks` distinguishes a
    genesis miss from a storm that formed and died inside 2.5 days.
13. **Budget**: a 30 USD credit is available. The first case is run with one
    member to measure wall-clock before committing to the full ensemble.
    Anything beyond roughly 100 USD is discussed before spending.
14. **Control storms (2026-10-01 and 10-02, issue #12)**: the five cases are
    repeated on DUJUAN (2625) and PEIPAH (2515), each with its own config.
    They control for different things. PEIPAH matches Krovanh in grade (both
    tropical storms), genesis region and week; DUJUAN matches the pipeline
    (same year, ERA5T, preliminary reference) but reached typhoon grade.

    | | DUJUAN (2625) | PEIPAH (2515) |
    |---|---|---|
    | Cases, 6 h apart | 2026-09-16 00Z to 09-17 00Z | 2025-09-02 00Z to 09-03 00Z |
    | Centred on | TS upgrade, 09-16 12Z | first depression row, 09-02 12Z |
    | TS upgrade | 09-16 12Z, 16.4N 149.4E | 09-03 18Z, 28.5N 130.9E |
    | Peak while tropical | 960 hPa | 992 hPa |
    | Transition | extratropical 09-22 12Z, 40.0N 154.0E | extratropical 09-05 18Z, 35.5N 146.2E |
    | Initial conditions | ERA5T | consolidated ERA5 |
    | Reference | preliminary PDF | post-analysis CSV |
    | Stored crop | 5-55N 120-180E, ~5.5 GB per case | 10-55N 110-170E, ~5.0 GB per case |

    DUJUAN copies Krovanh's design case for case. PEIPAH's tropical-storm
    phase lasted only 42 h, so centring its cases on the TS upgrade would
    leave 36-60 h of lead before the transition; centred on the first
    depression row, every case forecasts the turn at the upgrade 18-42 h
    ahead and is compared up to lead 66-90 h. `formation_time` still means the
    TS upgrade in every config. The crops are wide on purpose: where the
    members go cannot be read before the run, and a model-side extratropical
    criterion would need the fields around the centre.
15. **The comparison covers the tropical-storm phase only (2026-10-02).**
    The reference is cut at JMA's transition row, kept, and nothing after it
    is scored: weakening to a depression for Krovanh (`depression_time`),
    extratropical transition for DUJUAN and PEIPAH (`extratropical_time`).
    The tracker still runs the full 240 h; forecast positions past the
    transition simply have nothing to be compared against. Keeping the
    transition row leaves Krovanh's comparison byte-identical, since its
    preliminary table ends there.

    Still open: whether the pre-formation depression rows of a post-analysis
    table are scored (binding now for PEIPAH, at the turn of the year for the
    2026 storms), and how the lifetime and member-fate tables should read for
    a storm that went extratropical, which the tracker can follow on.
