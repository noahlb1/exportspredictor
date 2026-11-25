# Base-year fold models and the scored UK panel

For spec #156, ticket #176, under [ADR 0001](../adr/0001-out-of-fold-uk-training.md), [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md), [ADR 0009](../adr/0009-test-year-read-once-per-frozen-version.md) and [ADR 0010](../adr/0010-market-reach-in-the-model-form.md). ITC's Realisation Test needs UK Export Potential at base years t = 2015–2019 from models that never saw a year after t. The artefact's fold models trained on 2012–2023, so they cannot provide it. This note records the run that trains those models and builds the panel the test reads.

**Status: partial.** Base years 2015–2018 are trained and cached, 20 of 25 models. Claude Code stopped the run twice because the machine was low on memory, both times while other processes used it; the run itself held about 4.5 GB:

- at about 12:05 on 25 Nov 2026, during 2017, beside a merger's pytest;
- at about 14:39, during 2019, when about 7 GB of node processes started.

The restart at 12:10 read 2015 and 2016 from the cache, in 5.5 s and 5.4 s, so the cache works as intended. A re-run now trains only 2019. See *Resuming*.

**2023 is not read, and the Test-year Ledger is untouched.** `build_panel` reads the feature files and Flows of 2012–2022 only. It raises `LeakageError` before reading anything if a year it needs (a panel year, a base year or Market Reach's t − 1) is 2023 or later. Its tests poison every Flow of 2023 and later in the fixture and get the same panel, and the ledger's write functions are patched to fail.

## What was run

- Code: `src/exportspredictor/base_years.py` (commit 593d2b6); tests: `tests/test_base_years.py`.
- Command, from the worktree, logging to `C:\CodingProjectsC\ep-notes\156\t176-run.log`:

  ```
  EXPORTSPREDICTOR_FEATURE_SET=exporter_count powershell -NoProfile -File scripts/wt.ps1 python -m exportspredictor.base_years
  ```

- **Configuration**: from the newest artefact, `20261123T232505Z` (feature set `exporter_count`): its tuned params (max_depth 12, eta 0.1, Tweedie power 1.266), **1,279 rounds unscaled**, non-UK sampling rate 0.5894, seed 0, its 25 features with their monotone constraints, and its Model Form. The Model Form is the Blend (w = 0.3) followed by Market Reach (β = 4), refitted for each base year from the UK grid of t − 1.
- Each base year trains through `evaluation.train_test_models` on the years 2012..t. The feature transform, including the tariff clip range, is refitted on those years.
- Scoring goes through `folds.score_out_of_fold` → `prediction.export_potential`, with Market Reach from `market_reach.fit`, as the served version scores.
- Models are cached under `data/derived/base_year_models/<t>_<key>`. The key is a hash of the training years, params, rounds, folds, non-UK rate, seed, features and feature set (`base_years.models_dir`).

## Records

| Base year | Training years | Cache key | Train (s) | Score (s) | UK pairs scored |
|---|---|---|---|---|---|
| 2015 | 2012–2015 | `2015_7e22ceefcb59` | 2,791 (46.5 min) | 24.8 | 1,156,050 |
| 2016 | 2012–2016 | `2016_dc4ee9508377` | 3,258 (54.3 min) | 28.4 | 1,156,050 |
| 2017 | 2012–2017 | `2017_3234d76c383a` | 2,765 (46.1 min) | 21.7 | 1,156,050 |
| 2018 | 2012–2018 | `2018_18eb918538ce` | 3,490 (58.2 min) | 45.4 | 1,156,050 |
| 2019 | 2012–2019 | not finished (the second low-memory stop came during it) | | | |

- Each fold model is about 300 MB on disk (1,279 rounds at depth 12).
- **Slow models**: none stood out within a base year. The cost grows with the training years, about 9–11 min per fold model. That is near the 10 min of a fold model on the full 2012–2022 data (`final-round-2022.md`), although these models train on fewer years. GPU use stayed low, at 15–50%, so building the training matrix on the CPU probably dominates.
- 2019 should take about 1–1.2 h. The spec's 2–4 h for all 25 models was an underestimate: the full set is about 5 h.

## Resuming

Run the command above again, with no suite or other heavy job beside it. Base years 2015–2018 load from the cache in seconds and score in under a minute each. Only 2019 trains. The run then writes:

- `data/derived/base_years/panel.parquet`, read by `base_years.load_panel()`;
- `panel.json` beside it, which holds this run's record (artefact version, configuration, each base year's cache key, seconds and rows, and the panel's row and pair counts).

Fill in the table above from `panel.json`.

## The panel

`hs6`, `iso3`, `year` (2012–2022), `export_potential`, `naive_baseline` and `actual_exports`, all in USD.

- Rows: every pair of the base years' UK grids, crossed with every year 2012–2022. Excluded Products are left out, as in the served version.
- `actual_exports` is 0 where there is no Flow.
- The two scores are filled only on base-year rows of pairs in that year's grid.
- The 2022 grid file is the Gap Period's union of pairs, but it only supplies 2022's Actual Exports, never extra pairs.

## Caveats for the Realisation Test

- **Relatedness** comes from the proximity matrix of the shipped feature files, fitted on 2012–2021. So a base-year feature carries co-export structure from a few later years. Rebuilding the features per base year was out of scope. Proximity describes which Products countries export together, not any UK pair's own Flows.
- **The training scope** (top 100 Exporters by mean GDP over 2012–2023, ADR 0004) is the shipped one. It is chosen on GDP, not trade.
- **The round count** is the served 1,279. The served version scaled it to 2012–2023's rows, and it is not scaled down for fewer years, as the spec's "same round count" asks.
