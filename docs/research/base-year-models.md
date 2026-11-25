# Base-year fold models, the scored UK panel and the Realisation Test

For spec #156, ticket #176, under [ADR 0001](../adr/0001-out-of-fold-uk-training.md), [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md), [ADR 0009](../adr/0009-test-year-read-once-per-frozen-version.md) and [ADR 0010](../adr/0010-market-reach-in-the-model-form.md). ITC's Realisation Test needs UK Export Potential at base years t = 2015–2019 from models that never saw a year after t. The artefact's fold models trained on 2012–2023, so they cannot provide it. This note records the run that trains those models and builds the panel the test reads, then the test's run and results (#178).

**Status: complete.** All five base years are trained and cached, 25 of 25 models, and the panel is written. The run took three starts. Claude Code stopped it twice because the machine was low on memory, both times while other processes used it; the run itself held about 4.5 GB:

- **First stop**, at about 12:05 on 25 Nov 2026, during 2017, beside a merger's pytest. 2015 and 2016 were already cached. The restart at 12:10 read them from the cache, in 5.5 s and 5.4 s, then trained 2017 and 2018.
- **Second stop**, at about 14:39, during 2019, when about 7 GB of node processes started. The orchestrator restarted it at 14:44. That start read 2015–2018 from the cache in 5.6–8.8 s each, trained only 2019 and wrote the panel. It took 3,805 s (63 min).

**2023 is not read, and the Test-year Ledger is untouched.** `build_panel` reads the feature files and Flows of 2012–2022 only. It raises `LeakageError` before reading anything if a year it needs (a panel year, a base year or Market Reach's t − 1) is 2023 or later. Its tests poison every Flow of 2023 and later in the fixture and get the same panel, and the ledger's write functions are patched to fail. `panel.json` records `"test_year_read": false`.

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

Train and score times are from the start that trained each base year (the log); a cached load takes 5–9 s.

| Base year | Training years | Cache key | Train (s) | Score (s) | UK pairs scored |
|---|---|---|---|---|---|
| 2015 | 2012–2015 | `2015_7e22ceefcb59` | 2,791 (46.5 min) | 24.8 | 1,156,050 |
| 2016 | 2012–2016 | `2016_dc4ee9508377` | 3,258 (54.3 min) | 28.4 | 1,156,050 |
| 2017 | 2012–2017 | `2017_3234d76c383a` | 2,765 (46.1 min) | 21.7 | 1,156,050 |
| 2018 | 2012–2018 | `2018_18eb918538ce` | 3,490 (58.2 min) | 45.4 | 1,156,050 |
| 2019 | 2012–2019 | `2019_e254627fbfbd` | 3,632 (60.5 min) | 27.7 | 1,156,050 |

- Training all 25 models took 15,936 s, about 4.4 h, against the spec's 2–4 h.
- Each fold model is about 300 MB on disk (1,279 rounds at depth 12).
- **Slow models**: none stood out within a base year. The cost grows with the training years, about 9–12 min per fold model. That is near the 10 min of a fold model on the full 2012–2022 data (`final-round-2022.md`), although these models train on fewer years. GPU use stayed low, at 15–50%, so building the training matrix on the CPU probably dominates.

## The panel

`data/derived/base_years/panel.parquet`, read by `base_years.load_panel()`, with the run's record in `panel.json` beside it (`base_years.load_panel_info()`: artefact version, the frozen version's settings, each base year's cache key, seconds and rows, and the panel's row and pair counts).

- **12,716,550 rows**: 1,156,050 UK pairs × 11 years (2012–2022).
- Columns (`realisation.PANEL_SCHEMA`): `hs6`, `iso3`, `year`, `export_potential_usd`, `naive_baseline_usd` and `actual_exports_usd`. The money columns were renamed to carry their unit after the run, and the panel rescored from the 25 cached models.
- The pairs are every pair of the base years' UK grids. Excluded Products are left out, as in the served version.
- `actual_exports_usd` is 0 where there is no Flow.
- The two scores are filled only on base-year rows of pairs in that year's grid.
- The 2022 grid file is the Gap Period's union of pairs, but it only supplies 2022's Actual Exports, never extra pairs.

## Caveats for the Realisation Test

- **The shipped feature files were reused**, not rebuilt per base year; rebuilding them was out of scope. Two of their inputs are slowly moving aggregates that see years after t:
  - **Relatedness** comes from the proximity matrix fitted on 2012–2021. So a base-year feature carries co-export structure from a few later years. Proximity describes which Products countries export together, not any UK pair's own Flows.
  - **The training scope** (top 100 Exporters by mean GDP over 2012–2023, ADR 0004) ranks Exporters on GDP over years after t, not on trade.

  Neither carries a target value: no Flow, Export Potential or Actual Exports from after t enters a base-year model's training rows or its scores. The leak is in which Exporters are in scope and how Products relate, and both move slowly, so it should be small. It still leans the test, if at all, in the model's favour.
- **The round count** is the served 1,279. The served version scaled it to 2012–2023's rows, and it is not scaled down for fewer years, as the spec's "same round count" asks.
- `panel.json` records `frozen_settings.training_years` as an empty list: each base year's own training years are under `base_year_models`.
- **This departs from the spec's "never saw years after the base year"** in those two aggregate inputs only. No target value or trade flow after t, and nothing of 2023, is read.

## The Realisation Test on the panel (#178)

ITC's Realisation Test (`exportspredictor.realisation`; ITC, *Export Potential Assessment methodology*, June 2023, pp. 17–22) was run once on this panel. It reads Actual Exports for 2012–2022, and its report was added to a new version of the newest artefact.

- Command, from the worktree `ep-wt/156-t178` at commit da9ee75, logging to `C:\CodingProjectsC\ep-notes\156\t178-realisation.log`:

  ```
  powershell -NoProfile -File scripts/wt.ps1 python -m exportspredictor.realisation
  ```

- **New version: `20261125T171812Z`**, copied from `20261123T232505Z`. The command took 32 s wall-clock; the report itself took 21.2 s (`seconds`), on the CPU.
- **Nothing else changed.** The new version's evaluation gains one block, `itc_comparison`. Every other block is equal to the source's, and the metadata differs only in `version` and `copied_from`, so the Configuration Fingerprint is unchanged. The other 15 files are the same, by hash, or by size for the fold models.
- **The test year was not read.** The Test-year Ledger is unchanged, and each base year's training cut-off equals the base year (2015–2019).
- **The main checkout's older API** (`dc8c3bb`, before spec #156) loads the new version. `/health` names it, and `/evaluation`, `/caveats` and `/opportunities/top` answer 200. Main's `Evaluation` schema allows extra keys, so it passes the raw `itc_comparison` block (all 12 rows) through, and its app ignores it.

### Results

Coefficients, with standard errors clustered by unit in brackets. Fixed effects are Market and year × Sector. ITC's figures are its static test with fixed effects: every exporter pooled, 2006–2018, at 76 sub-sectors (Tables 1, 2 and 4).

| Test | Term | ITC | Model, HS2 | Naive Baseline, HS2 | Model, HS6 | Naive Baseline, HS6 |
|---|---|---|---|---|---|---|
| Intensive margin | Unrealised Share | 0.615 | 0.658 (0.028) | 0.715 (0.026) | 1.270 (0.005) | 1.127 (0.004) |
| Intensive margin | past growth | −0.386 | −0.482 (0.006) | −0.480 (0.006) | −0.382 (0.001) | −0.380 (0.001) |
| Extensive margin | log Export Potential | 0.023 | 0.008 (0.0015) | 0.014 (0.0017) | 0.035 (0.0001) | 0.027 (0.0001) |
| New-flow size | log Export Potential | 0.243 | 0.161 (0.020) | 0.154 (0.019) | 0.313 (0.002) | 0.204 (0.002) |

Samples (unit-years, base years 2015–2019):

| Test | Level | Model | Naive Baseline | Left out by the Naive Baseline (score 0) |
|---|---|---|---|---|
| Intensive margin | HS2 | 71,823 | 71,708 | 174 of 75,507 trading |
| Intensive margin | HS6 | 1,489,078 | 1,481,990 | 18,069 of 1,719,360 trading |
| Extensive margin | HS2 | 32,485 | 29,691 | 2,794 of 32,493 not trading |
| Extensive margin | HS6 | 4,028,940 | 2,564,874 | 1,464,066 of 4,060,890 not trading |
| New-flow size | HS2 | 10,325 | 10,128 | starters among the above |
| New-flow size | HS6 | 619,592 | 584,926 | starters among the above |

The model scores every unit above zero, so its rows leave none out. At HS6, 31,950 not-trading pair-years are left out of both rows because the UK exported their Product nowhere that year.

**Reading, as a rough benchmark of sign and broad size:**

- At HS2, the level set beside ITC, the model agrees with ITC's sign on all three tests.
- **Intensive margin.** A larger Unrealised Share is associated with faster growth over the next three years: 0.658 against ITC's 0.615. Past growth is associated with slower growth: −0.482 against −0.386.
- **Extensive margin.** Higher Export Potential is associated with a non-trading Sector–Market starting to trade, but the slope is about a third of ITC's (0.008 against 0.023). A linear probability model's slope also depends on how often units start, which differs between the UK's Sectors and ITC's pooled sub-sectors.
- **New-flow size.** Higher Export Potential is associated with larger new flows, at about two-thirds of ITC's slope (0.161 against 0.243).
- **The Naive Baseline column goes beyond ITC's test**, which has no naive comparator. At HS2 its coefficients are similar or larger (0.715, 0.014, 0.154), so at Sector level the link between unrealised potential and later trade owes much to what the market-share rule already sees, and the Blend contains that rule. At HS6 the model's coefficients are the larger on all three tests, but there the Naive Baseline's extensive-margin sample is about two-thirds of the model's.
- These are associations across many pairs, not forecasts for any one, and not a validation of either tool.

### Further caveats

- **Part (c) was dropped**: the direct comparison with the EPM's own UK figures (#156 step-one comment, 25 Nov 2026).
  - The public site offers only a chart image (PNG), not a table.
  - ITC's Market Analysis Tools Terms and Conditions §12 forbid extracting MAT content by browser automation. §10 allows only limited and incidental quotation without ITC's prior written permission.
  - The EPM is an estimate for 2031, built on 2021–2025 trade data, while this test scores each base year itself.
  - (c) could be revived if a person obtains the data under a licence or with ITC's permission. Even then, only aggregate statistics should be published.
- **Each predictor runs on the units it scores above zero**, since a log needs it. The model's row is the ITC-comparable one. The two columns therefore compare different samples wherever the Naive Baseline scores 0, above all on the HS6 extensive margin, where 1,464,066 pair-years are left out.
- ITC pools every exporter with an exporter fixed effect, and projects about five years ahead with a decomposition. This test covers the UK only, at HS2 Sectors rather than 76 sub-sectors, and has no forward projection.
