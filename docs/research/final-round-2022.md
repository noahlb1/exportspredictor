# The final round on the 2022 validation year

For spec #155, under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md), [ADR 0001](../adr/0001-out-of-fold-uk-training.md) and [ADR 0009](../adr/0009-test-year-read-once-per-frozen-version.md). The model is about to ship. This round tries the ideas left that use only data we already have. Each idea runs through the experiment loop (`exportspredictor.loop`) as one or more Arms on fold 0 of 2022. It is kept only if its Selection Measure beats the best so far by more than the Win Threshold. Winners stack: each idea starts from the ones kept before it.

**2023 is not read during the round.** The loop's data refuses the test year. If anything is kept, all winners become one new frozen version, and 2023 is read once for it through the Test-year Ledger.

## Summary

One row per idea, in the order run. Later tickets add their rows here and their own `## ` section below. "Base" is the best so far as served at the start of the idea. "Gain" is the Arm's Selection Measure minus the base's.

| # | Idea (ticket) | Arms | Base | Best Arm | Gain | Win Threshold | GPU time | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | [Cap on Market Reach's upside](#1-a-cap-on-market-reachs-upside-163) (#163) | 1 (4 caps) | 0.6210 | cap 16×, 0.6204 | −0.0006 | 0.008869 | none (scored from cache, 5.5 min) | **Dropped** |
| 2 | [Sibling Demand, a UK-free demand proxy, as a feature](#3-sibling-demand-a-uk-free-demand-proxy-as-a-feature-164) (#164) | 1 | 0.6210 (retrained on `sibling_demand`) | w = 0.3, 0.6213 | +0.0003 | 0.008869 | 76.6 min (2 models; see the section) | **Dropped** |
| 3 | [A fresh hyperparameter search](#4-a-fresh-hyperparameter-search-166) (#166) | 14 in 5 stages | 0.6210 | `colsample_bytree` 0.6, 0.6215 | +0.0005 | 0.008869 | 134.6 min (14 models) | **Dropped** (every stage) |
| 4 | [Other model families: LightGBM and CatBoost](#5-other-model-families-lightgbm-and-catboost-167) (#167) | 2 (one experiment per family) | 0.6210 | LightGBM ⚑, 0.6205 (CatBoost 0.6191) | −0.0005 | 0.008869 | none: CPU only, 28.7 min (2 models) | **Dropped** (both) |
| 5 | [A wider Exporter scope](#6-a-wider-exporter-scope-169) (#169) | 1 (paired, one non-UK rate) | 0.6212 (retrained on `all_exporters`, top 100 Exporters) | every Exporter, 0.6212 | +0.0000 | 0.008869 | 20.6 min (2 models) | **Dropped** |
| 6 | [More training rows through GPU external memory](#7-more-training-rows-through-gpu-external-memory-170) (#170) | 1 (non-UK rate 1.0; the wider scope unsampled not run, as #169 dropped the scope) | 0.6210 | 132 M rows, 0.6209 | −0.0000 | 0.008869 | 493.6 min (1 model, 35 s a round) | **Dropped** |
| 7 | [Train on the Excluded Products](#8-the-leftover-research-leads-171) (#171) | 1 (paired, one non-UK rate) | 0.6214 (retrained on `excluded_products`, Excluded Products dropped) | trained on them, 0.6207 | −0.0007 | 0.008869 | 18.2 min (2 models) | **Dropped** |
| 8 | [Tariff 0 where an agreement's preference is unrecorded](#8-the-leftover-research-leads-171) (#171) | a scoring-only probe; the retrained Arm gated out | 0.6210 | the probe, 0.6209 | −0.0001 | gate +0.0044 | none (26 s, cached) | **Skipped** (retrained Arm not run) |

Not an idea, so not a row: [section 2](#2-wider-exporter-scope-feature-files-168) records the build of the `all_exporters` feature files (#168), which the wider Exporter scope experiment (#169, section 6) trains on. Section 8 also lists the leads #171 skipped, each with its reason.

## The base every idea starts from

Market Reach was adopted outside the loop, by #124's five-fold gate ([market-reach-2022.md](market-reach-2022.md), [ADR 0010](../adr/0010-market-reach-in-the-model-form.md)). So the loop's recorded best so far (`results["best"]`, the Blend at w = 0.3, 0.5771) carries no Market Reach. The served model also trains on the `exporter_count` files (#127 kept the gravity fill).

Each idea of this round is therefore decided against the **best so far as served**: `loop.served_base_arm(best)`, which adds Market Reach at β = 4 to a best so far without it. It runs on the feature files that `served_configuration` serves (`kept_feature_set`, now `exporter_count`), as a paired comparison (`Loop.run_experiment`'s `base_arm`). A kept Arm becomes `results["best"]`, with Market Reach and anything else it changed in its settings. After that, `served_base_arm` adds nothing.

The Win Threshold is `loop.current_win_threshold`, 0.008869 (#116).

## 1. A cap on Market Reach's upside (#163)

**The idea.** Market Reach multiplies each held-out UK pair's Export Potential by exp(β · offset) of its Market, at β = 4, then rescales the grid to keep its summed Export Potential. A Market with a large offset is scaled far above comparable Markets. The first form gave the Falkland Islands a multiplier of about 230×. The nested form fixed that defect (offset 0.35, about 4×), but the 95th percentile of offsets (+0.71) still gives about 17×, and Ireland (+0.91) about 38×. A cap on the multiplier limits how far one strong Market Reach estimate can lift a Market.

**The Arm.** A new scoring field, `Settings.market_reach_cap`, holds the multiplier exp(β · offset) at or below the cap, before the rescale constant. Its default, None, caps nothing, and the fixture test checks that no cap gives exactly the uncapped scores. It changes only scoring, so the Arm reuses the base's cached model and trains nothing.

**Pre-registered** (committed before any 2022 result was read):
- **Grid:** caps of 2, 4, 8 and 16 (`loop.MARKET_REACH_CAP_GRID`). At β = 4 these cap offsets above 0.17, 0.35, 0.52 and 0.69.
- **Choice:** the cap with the highest Selection Measure on fold 0 of 2022.
- **Base:** the best so far as served (Blend w = 0.3, Market Reach β = 4, no cap), scored from the model #127 trained on the `exporter_count` files (`e5943f03616c0cf7`). #127 measured it at 0.6168.
- **Gate:** keep the cap only if it beats the base by more than `current_win_threshold` (0.008869).
- **Command:** `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop market-reach-cap`. It writes `experiments["market reach cap"]` to the loop's `results.json`.

**Decision: drop the cap.** Every cap ranks the held-out UK Markets worse than no cap, and the measure falls steadily as the cap tightens. The best cap, 16×, scores **0.6204** against **0.6210** for the base. That is a change of **−0.0006**, against a Win Threshold of 0.008869. The best so far is unchanged: `results["best"]` is still the Blend without Market Reach, and the round's next idea starts from `served_base_arm` of it, as this one did.

**Results.** Fold 0 of 2022, 231,217 held-out UK pairs, 4,858 ranked Products, every figure under the tie-proof measure. Deviance is the Tweedie deviance on the validation rows. Both rows score the same cached model (`e5943f03616c0cf7`, 824 rounds), so the cap is the only difference.

| Arm | Cap | **Selection Measure** | Change | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance |
|---|---|---|---|---|---|---|---|---|
| Best so far as served (base) | none | **0.6210** | | 0.507 | 0.589 | 0.816 | 0.943 | 464.1 |
| Cap Market Reach | 2× | 0.6132 | −0.0078 | 0.507 | 0.581 | 0.815 | 0.941 | |
| | 4× | 0.6158 | −0.0052 | 0.506 | 0.584 | 0.815 | 0.941 | |
| | 8× | 0.6184 | −0.0026 | 0.504 | 0.587 | 0.815 | 0.944 | |
| | **16× (chosen)** | **0.6204** | **−0.0006** | 0.504 | 0.589 | 0.816 | 0.943 | 462.1 |

Without Market Reach, both rows score 0.5762 (#127's retrained base without Reach).

**Reading it.**
- **The upper tail of Market Reach carries signal.** Markets with large offsets are the many-Product Markets the UK serves broadly (Ireland, France, Germany, the United States). Holding their multiplier down mostly lowers "zero against non-zero" (0.589 to 0.581 at 2×). That is the extensive margin Market Reach was adopted for.
- **The Falklands case is already fixed.** The nested form brought its offset from 2.06 to 0.35 (ADR 0010, *Amendment*). A cap at 4× or below would have bound it, and both cost more than 0.005.
- **The base is 0.6210, not #127's 0.6168.** #127 scored the first (`share`) form. This run uses the nested form now served. The +0.0042 matches the nested form's extra gain on five folds (+0.0449 against +0.0406).
- As always, this is an association on one fold and one year. Capping ranks held-out UK Markets worse in 2022, which says nothing about why.

**Run.** `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop market-reach-cap`, on 2026-11-24, while #168's feature build was using the CPU. It took 333 s in all: 29 s of setup, 167 s to score the base and 132 s for the Arm's four caps. XGBoost warned of mismatched devices (the booster on the CPU, the input on `cuda:0`) and fell back to slower prediction through a DMatrix. That is harmless, but it is why scoring took minutes. No model was trained and no GPU training time was used. The experiment is recorded under `experiments["market reach cap"]` in the loop's `results.json`. 2023 was not read.

## 2. Wider Exporter scope: feature files (#168)

For spec #155, ticket #168, under [ADR 0004](../adr/0004-training-scope-and-sampling.md). v2 trains on the top 100 Exporters by mean GDP; v1 trained on the whole world. The wider scope tests whether that cut caused v2's −0.020 drop from v1. This section covers only the feature files. The experiment that trains on them is #169.

**The feature set is `all_exporters`**: `exporter_count` (#126: the Exporter count as a candidate feature, GDP filled for small Markets) with **every Exporter's rows**, 227 Exporters in place of 100. All Exporters were chosen over a larger top N because they test the hypothesis directly: v1's scope against v2's.

- Built with one command, on the CPU, in 40.4 minutes: `scripts/wt.ps1 python -m exportspredictor.features --set all_exporters --parts 8`. The set's scope comes from `features.FEATURE_SET_SCOPES` (`scope.ALL_EXPORTERS`).
- Peak RAM was **11.6 GB** (the build process's peak working set, sampled every 5 s). Fitting Product proximity alone reached 9.2 GB; building one year in 8 groups of Markets reached 11.6 GB. Other programs held about 22 GB of the 32 GB when it started. The dense Exporter × Importer × Product grid is never formed: Zero Cells come only from Exporters and Importers active in the Product that year, as before.
- The years are 2012–2024 like `exporter_count`, so the pipeline can serve the set if it is kept. The loop never opens 2023 or 2024: `loop.real_data` refuses the test year, and a test runs the real path on this set.
- The files take 7.0 GB on disk, against 5.9 GB for `exporter_count`.

**The top 100's rows are unchanged.** In 2021 and 2022, the rows of the 100 in-scope Exporters are the same rows as `exporter_count`'s, with the same features, and the UK grids are identical. The only difference is in Relatedness: float32 rounding, at most 6.6e−7, because the matrix product runs on a matrix with more Exporter rows. So the wider files equal the current ones plus the other 127 Exporters' rows, and #169's comparison isolates the scope.

### Row counts and sampling rates

Zero Cells of other Exporters are kept at 5%, as in every set (weight 20). Every UK row is kept at weight 1. The files are **not** sampled by Exporter: the loop samples other Exporters' rows at the non-UK rate when it trains (`training.sample_non_uk`, weight 1/rate). That keeps the unsampled files for the GPU external memory Arm, and lets the noise floor's seeds draw different samples.

| | `exporter_count` (top 100) | `all_exporters` | Change |
|---|---|---|---|
| Exporters | 100 | 227 | +127 |
| UK rows, 2012–2021 | 8,445,954 | 8,445,954 | 0 |
| Other Exporters' rows, 2012–2021 | 125,565,413 | 147,263,996 | +17.3% |
| of which Flows | 97,070,202 | 103,682,812 | +6.8% |
| of which Zero Cells (sampled at 5%) | 28,495,211 | 43,581,184 | +52.9% |
| Validation year 2022 (never sampled) | 14,246,975 | 16,765,385 | +17.7% |
| Non-UK rate that fits the VRAM budget | 0.6157 | **0.5079** | |
| Expected rows on the GPU at that rate | 100.0 M | 100.0 M | |

The non-UK rate is `search.final_non_uk_rate`, the rule the loop applies when an Arm leaves `non_uk_rate` unset. It is sized for the pipeline's VRAM budget (`training.row_budget`, 100,032,198 rows with 25 features: 7.5 GiB above an idle desktop). The same rule gives `exporter_count` 0.6157, the rate v2 trains at, which checks the method.

Most of the extra rows are Zero Cells, as ADR 0004 predicted: small Exporters add a Zero Cell for every Importer of each Product they ship. The extra Flows add 7% more Flow rows but carry little trade.

### For the training run (#169)

- Set `EXPORTSPREDICTOR_FEATURE_SET=all_exporters`. An Arm on these files needs `non_uk_rate` at 0.5079 or below (or `None`, which derives it). The best so far's 0.6157 would put about 115.9 M rows on the GPU, over the budget. The loop does not check an explicit rate against the budget (only the pipeline calls `training.check_row_budget`), so the Arm must change it.
- At 0.5079 the training matrix holds about **83.2 M rows** (8.4 M UK plus 74.8 M sampled others), and the whole 2022 year adds 16.8 M, for 100.0 M in all: the same size as v2's. One fold-0 model should take about as long as v2's, roughly 11 minutes on a quiet machine.
- That budget assumes most of the 10 GB is free. When the build ran, the user's API and desktop apps held about 6.8 GB, which leaves too little for 100 M rows. Check `nvidia-smi` before training.
- The comparison is then at a lower non-UK rate (0.51 against 0.62) as well as a wider scope. Keeping the rate fixed would need GPU external memory, which is the spec's idea 6.

These are build records only. Nothing here was trained or scored, and 2023 was not read.

## 3. Sibling Demand: a UK-free demand proxy as a feature (#164)

**The idea.** Idea 9 of the [open review](open-review-2022.md#9-a-uk-free-demand-proxy-hypothesis). Leave-one-out demand (`log_demand_loo`) removes the row's own Exporter, so where the UK supplies most of a Market's imports of a Product, the Market looks small to a UK row. UK milk and cream to Ireland is the open review's example. A demand proxy built without UK Flows can still see that the Market buys related goods. As a blend term, the Market's imports of the Product's HS4 siblings gained +0.0067 on folds 3–4. That is below the threshold, and it added nothing on top of Market Reach (+0.0005). As a model feature, the trees can let it interact with the Product and the other features. That form was untested.

**The feature, Sibling Demand** (`log_sibling_demand_loo`): ln(1 + the Market's imports that year of the other Products of the Product's HS4 heading, from every Exporter except the UK and the row's own).
- **No UK Flow** is read, for any row. So a UK row never sees its own pair or the UK's trade with the Market in other Products, and the definition is the same in training and serving.
- **The row's own Exporter is left out too**, so no row reads its pair's trade in other Products (the features' "no pair trade in other Products" rule).
- **The Product itself is left out.** Its other-Exporter demand is already `log_demand_loo`.
- GDP, the open review's other proxy, is already a feature (`log_gdp_d`), so only Sibling Demand is tried.
- Fixture tests show that a UK pair's Sibling Demand does not move when its own Flows change, or when every UK Flow is removed.

**The files.** The `sibling_demand` feature set is `exporter_count` (the served files) with Sibling Demand stored as a second candidate feature (`features.FEATURE_SET_CANDIDATES`). Every other column is built by the same code from the same Flows. So the base retrained on these files differs from the Arm only by the feature. A model cached on `exporter_count` is not reused: the feature set is part of the cache key.

**Pre-registered** (committed before any 2022 result was read):
- **Base:** the best so far as served (`served_base_arm`: Blend w = 0.3, Market Reach β = 4), retrained on fold 0 of the `sibling_demand` files.
- **Arm** (`+ log_sibling_demand_loo`): the base with Sibling Demand as a 26th feature, with no monotone constraint. It re-chooses its blend weight from 0.1 to 0.9 on the measure with Market Reach, as every Arm with a new model has since #71.
- **Gate:** keep it only if it beats the base by more than `current_win_threshold` (0.008869). If it is dropped, the served files stay `exporter_count`: the experiment records `files_kept_if_dropped: false`, so `kept_feature_set` ignores it unless an Arm is kept.
- **Command:** `$env:EXPORTSPREDICTOR_FEATURE_SET='sibling_demand'; scripts/wt.ps1 python -m exportspredictor.loop sibling-demand` (two models). It writes `experiments["sibling demand"]` to the loop's `results.json`.

**Decision: drop Sibling Demand.** The Arm scores **0.6213** against **0.6210** for the base retrained on the same files. That is a gain of **+0.0003**, against a Win Threshold of 0.008869. Without Market Reach it is −0.0003 (0.5754 against 0.5756). The best so far is unchanged (`results["best"]` is still the Blend without Market Reach). The served files stay `exporter_count`: `kept_feature_set` ignores this experiment because it was dropped. The round's next idea starts from `served_base_arm` on `exporter_count`, as this one did.

**Results.** Fold 0 of 2022, every figure under the tie-proof measure. Deviance is the Tweedie deviance on the validation rows. "Model alone" is the stopping measure: the model's own Selection Measure, without a blend or Market Reach.

| Arm | Features | w | **Selection Measure** | Change | Without Reach | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance | Model alone | Rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Best so far as served, retrained (base) | 25 | 0.3 | **0.6210** | | 0.5756 | 0.507 | 0.589 | 0.816 | 0.945 | 463.2 | 0.5300 | 790 |
| + Sibling Demand (the Arm) | 26 | 0.3 (chosen) | **0.6213** | **+0.0003** | 0.5754 | 0.507 | 0.589 | 0.816 | 0.943 | 454.5 | 0.5261 | 787 |

The Arm's blend grid with Market Reach, from w = 0.1 to 0.9, reads 0.6191, 0.6206, **0.6213**, 0.6211, 0.6203, 0.6189, 0.6171, 0.6151 and 0.6128. It chose the base's own weight.

**Reading it.**
- **Nothing the ranking can use.** Every ranking measure moves by 0.002 or less, in both directions. The model alone is slightly worse (0.5261 against 0.5300), and the Blend with Market Reach recovers the difference.
- **The deviance falls** (454.5 against 463.2). So the feature helps the model fit dollar levels across all rows. But the Selection Measure ranks Markets within a Product, and that ordering does not improve.
- **Market Reach already holds the signal.** The open review found the blend-term form added +0.0005 on top of Market Reach. As a feature it adds +0.0003. Market Reach reads the UK's presence in a Market across Products, which is the "this Market buys from the UK" information that leave-one-out demand misses. Sibling Demand reaches the same information another way.
- **The base reproduces #163's base**: 0.6210 on the `sibling_demand` files against 0.6210 scored from #127's model on `exporter_count`. Every other column of the two sets is identical, so this is a fresh GPU draw of the same model, inside the noise floor.
- As always, this is an association on one fold and one year. It does not say whether demand for related goods causes UK exports.

**Build.** `scripts/wt.ps1 python -m exportspredictor.features --set sibling_demand --parts 8`, for 2012–2024 like `exporter_count`, so that the 2022 UK grid is the same Gap Period union and the set could be served if kept. It took **34.0 min** on the CPU, with a peak of **15.7 GB** of RAM (the build process's peak working set). The files take 6.2 GB, against 5.5 GB for `exporter_count`. In 2021 and 2022, the training rows and UK grids have the same keys and row counts as `exporter_count`'s (14,246,975 training rows and 1,156,050 grid pairs in 2022), and every other column is exactly equal. Sibling Demand has no missing values. It is zero for 15.5% of the 2022 UK grid, where the Market imports no other Product of the heading from anyone but the UK.

**Run.** `$env:EXPORTSPREDICTOR_FEATURE_SET='sibling_demand'; scripts/wt.ps1 python -m exportspredictor.loop sibling-demand`, on 2026-11-24 from 10:58 to 12:15, one GPU run at a time. Both models trained on 84,062,589 rows, and nothing was taken from a cache: the base's key (`8ee3acf73e913a68`) is not the key of #127's model on `exporter_count` (`e5943f03616c0cf7`).

| Model | Matrices | Training (matrices included) | Rounds kept / trained |
|---|---|---|---|
| Base retrained (`8ee3acf73e913a68`) | 84 s | 4,011 s | 790 / 890 |
| + Sibling Demand (`a8a94c0aa05727a1`) | 94 s | 587 s | 787 / 887 |

The base took about seven times as long as the Arm for the same number of rounds. Before the run, other programs held 1.8 GB of the 10 GB of VRAM, and no other training run was active. Halfway through the base, 9.7 GB was in use, this run included, at 100% utilisation. The cause of the slowdown was not traced. Memory pressure near the VRAM limit is a guess. The Arm's 587 s matches #127's 452–680 s. 2023 was not read.

## 4. A fresh hyperparameter search (#166)

**The idea.** The served XGBoost parameters come from #54's search: Optuna TPE on the Sector subset, selected on 2022 Tweedie deviance, with no Market Reach and before the Selection Measure was tie-proof. #71 then compared #54's three confirmed configurations, v1's and depth 6 on the Selection Measure and found them within the noise. This search asks whether parameters chosen *on the Selection Measure*, for the model as served (the Blend with Market Reach), rank held-out UK Markets better.

**The design.** Five stages, run in the spec's order, each one experiment of the loop (`loop.HYPERPARAMETER_STAGES`, `python -m exportspredictor.loop hyperparameters <stage>`). Each stage runs against the best so far as served (`served_base_arm`), whose model is cached, so each Arm trains one model and every stage stays well within `MAX_TRAINED_MODELS` (6); a stage over the budget is refused before training. An Arm changes only the parameters it names; the others are the best so far's. Every Arm re-chooses its blend weight from 0.1 to 0.9, as every Arm with a new model has since #71, and keeps Market Reach at β = 4. Rounds still stop on the model alone's Selection Measure (patience 100). **Winners stack**: a kept stage becomes `results["best"]` (with Market Reach in its settings, so `served_base_arm` adds nothing after it), and the next stage changes its parameters. A stage is refused until the stages before it are decided.

The best so far at the start: `max_depth` 12, `eta` 0.1, `min_child_weight` 3.05, `subsample` 0.916, `colsample_bytree` 0.879, `reg_lambda` 4.95, `tweedie_variance_power` 1.266; Blend w = 0.3; Market Reach β = 4; `exporter_count` files; non-UK rate 0.6157.

**Pre-registered** (committed before any stage was run):

| Stage | Arms (the parameter each changes) | Why these values |
|---|---|---|
| Depth | `max_depth` 8, 10, 14 | Either side of 12. Depth 6 is left out: #71 found it stopped at round 12, decided by the stopping rule. |
| Learning rate | `eta` 0.05, 0.2 | Half and double. 0.05 roughly doubles the rounds and the GPU time. |
| Min child weight | `min_child_weight` 1, 10, 30 | From #54's lowest (1.4) up to well past its highest (6.8): larger leaves may suit a ranking within Products. |
| Subsampling | `subsample` 0.7; `colsample_bytree` 0.6; both | Each alone, then together. |
| Regularisation | `reg_lambda` 1, 20, 50 | #54's confirmed configurations had 0.96 and 23.8; 50 tests more. |

- **Gate:** within each stage, the Arm with the highest Selection Measure is kept only if it beats that stage's base by more than `current_win_threshold` (0.008869).
- **The stopping rule:** any Arm whose best round is below 50 (`STOPPING_RULE_ROUNDS`) is flagged here as decided by the stopping rule, not by its configuration.
- **Not searched:** `tweedie_variance_power` (not among the spec's parameters; it changes the loss, not the trees), `gamma` and `reg_alpha` (their scale depends on the Tweedie gradients, so no grid could be fixed without a pilot).
- **Budget:** 14 models in all. At #164's ~10 minutes a model that is about 2.5 hours of GPU time; at its slow base's 67 minutes it would be about 16 hours. If a stage's first model runs that slowly, the later stages' grids are cut before they run, and the cut is recorded here.
- **Commands:** `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop hyperparameters depth`, then `learning-rate`, `min-child-weight`, `subsampling` and `regularisation`. Each writes `experiments["hyperparameters: <stage>"]` to the loop's `results.json`. 2023 is not read.

**Decision: drop every stage; the served parameters stay.** No Arm of any stage comes near the Win Threshold. The best Arm of the whole search, `colsample_bytree` 0.6, scores **0.6215** against **0.6210** for the base, a gain of **+0.0005** against a threshold of 0.008869. Since no stage was kept, every stage ran against the same base, the best so far as served, scored from #127's cached model (`e5943f03616c0cf7`), and `results["best"]` is unchanged (the Blend w = 0.3 without Market Reach, 0.5771). The served files stay `exporter_count`. #167 starts from `served_base_arm` of that best so far, as this search did.

**Results.** Fold 0 of 2022, every figure under the tie-proof measure, with Market Reach at β = 4. "Without Reach" is the same Arm's Blend without Market Reach; "Model alone" is its stopping measure (no blend, no Reach); the base was scored from the cache, so its model-alone figure is not re-recorded (#164's retrained base: 0.5300). Deviance is the Tweedie deviance on the validation rows. Every Arm re-chose w = 0.3, the base's weight.

| Stage | Arm | **Selection Measure** | Change | Without Reach | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance | Model alone | Rounds kept / trained | Training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (every stage) | Best so far as served (base, cached) | **0.6210** | | 0.5762 | 0.507 | 0.589 | 0.816 | 0.943 | 464.1 | | 824 / 924 | cached |
| Depth | `max_depth` 8 | 0.6208 | −0.0002 | 0.5757 | 0.504 | 0.589 | 0.815 | 0.944 | 472.8 | 0.5338 | 1,106 / 1,206 | 329 s |
| | `max_depth` 10 | 0.6210 | −0.0000 | 0.5776 | 0.505 | 0.589 | 0.815 | 0.944 | 462.2 | 0.5378 | 1,441 / 1,541 | 451 s |
| | `max_depth` 14 | 0.6213 | +0.0004 | 0.5770 | 0.508 | 0.590 | 0.816 | 0.945 | 469.6 | 0.5315 | 857 / 957 | 1,340 s |
| Learning rate | `eta` 0.05 | 0.6209 | −0.0001 | 0.5750 | 0.507 | 0.589 | 0.815 | 0.942 | 462.1 | 0.5276 | 1,090 / 1,190 | 774 s |
| | `eta` 0.2 | 0.6213 | +0.0003 | 0.5763 | 0.506 | 0.589 | 0.816 | 0.944 | 471.9 | 0.5320 | 843 / 943 | 545 s |
| Min child weight | `min_child_weight` 1 | 0.6207 | −0.0003 | 0.5745 | 0.506 | 0.589 | 0.815 | 0.945 | 465.4 | 0.5268 | 527 / 627 | 430 s |
| | `min_child_weight` 10 | 0.6212 | +0.0002 | 0.5761 | 0.507 | 0.589 | 0.816 | 0.945 | 463.3 | 0.5305 | 986 / 1,086 | 596 s |
| | `min_child_weight` 30 | 0.6214 | +0.0004 | 0.5782 | 0.506 | 0.590 | 0.816 | 0.943 | 461.8 | 0.5360 | 1,052 / 1,152 | 637 s |
| Subsampling | `subsample` 0.7 | 0.6211 | +0.0001 | 0.5774 | 0.506 | 0.589 | 0.816 | 0.946 | 460.9 | 0.5365 | 867 / 967 | 624 s |
| | **`colsample_bytree` 0.6** | **0.6215** | **+0.0005** | 0.5772 | 0.507 | 0.590 | 0.816 | 0.943 | 464.9 | 0.5346 | 698 / 798 | 452 s |
| | `subsample` 0.7 + `colsample_bytree` 0.6 ⚑ | 0.6205 | −0.0005 | 0.5489 | 0.487 | 0.591 | 0.810 | 0.936 | 1,043.7 | 0.5107 | **6** / 106 | 66 s |
| Regularisation | `reg_lambda` 1 | 0.6211 | +0.0001 | 0.5765 | 0.506 | 0.589 | 0.815 | 0.944 | 463.6 | 0.5317 | 1,085 / 1,185 | 734 s |
| | `reg_lambda` 20 | 0.6210 | −0.0000 | 0.5757 | 0.507 | 0.589 | 0.816 | 0.944 | 461.8 | 0.5296 | 758 / 858 | 479 s |
| | `reg_lambda` 50 | 0.6211 | +0.0001 | 0.5764 | 0.507 | 0.589 | 0.815 | 0.943 | 463.0 | 0.5323 | 1,013 / 1,113 | 621 s |

Each stage's decision: depth (best `max_depth` 14, +0.0004), learning rate (`eta` 0.2, +0.0003), min child weight (30, +0.0004), subsampling (`colsample_bytree` 0.6, +0.0005) and regularisation (`reg_lambda` 50, +0.0001): all **dropped** against 0.008869.

**⚑ Decided by the stopping rule.** One Arm's best round is below 50: `subsample` 0.7 with `colsample_bytree` 0.6 peaked at round **6** and stopped at 106. Its model alone ranked best on the first few trees, dipped, and patience 100 ended training before it recovered, if it would have. Its deviance (1,043.7, more than double every other Arm's) shows a model that had barely started to fit. So its −0.0005 reflects the stopping rule, not the configuration. Each half of it, trained alone, ran its full course (867 and 698 rounds) and stayed inside the noise, which makes it unlikely, though not proven, that the pair would have cleared the threshold. This is the same caveat as #71's depth 6 and #70's starting point ([opportunity-ranking-experiments.md](opportunity-ranking-experiments.md#arms-decided-by-the-stopping-rule-not-by-their-configuration)).

**Reading it.**
- **The parameters do not matter for the ranking once the model is blended and Market Reach is applied.** The 13 fully trained Arms span 0.6207 to 0.6215, a range of 0.0008, under a tenth of the Win Threshold and well inside the noise floor's seed-to-seed spread (SD 0.0044). Every one re-chose the base's blend weight, and every blend grid has the same shape (peak at w = 0.3, 0.612–0.613 at w = 0.9).
- **The model alone moves more than the served form.** Model-alone measures span 0.5268 to 0.5378 (depth 10 highest), about ten times the served spread. The naive baseline in the Blend and Market Reach absorb most of what the parameters change. This repeats #71's finding ("configuration barely matters once blended") under the tie-proof measure and with Market Reach.
- **Deviance and ranking disagree again**: `subsample` 0.7 has the lowest deviance (460.9) and a gain of +0.0001; depth 8 the highest of the full runs (472.8) and −0.0002. #54 chose on deviance; this search confirms that choice costs the ranking nothing measurable.
- As always, these are associations on one fold and one year. A parameter that "wins" here ranks held-out UK Markets better in 2022 and says nothing about why.

**Run.** On 2026-11-24, one stage at a time, each started in the background after `nvidia-smi` showed the GPU idle (1.1–1.5 GB of the 10 GB held by other programs): `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop hyperparameters <stage>`.

| Stage | Wall clock | Models | Training (matrices included) |
|---|---|---|---|
| Depth | 12:38–13:15 (37.1 min) | 3 | 2,119 s |
| Learning rate | 13:15–13:39 (23.6 min) | 2 | 1,319 s |
| Min child weight | 13:39–14:09 (29.4 min) | 3 | 1,663 s |
| Subsampling | 14:09–14:30 (20.7 min) | 3 | 1,143 s |
| Regularisation | 14:30–15:02 (32.3 min) | 3 | 1,834 s |
| **Total** | **143.1 min** | **14** | **8,077 s = 134.6 min of GPU time** |

Each stage also re-scored the cached base (about 65 s on the CPU: XGBoost warns of mismatched devices for a loaded model, as #163 noted). The first model of each stage built the training matrices (83–85 s), and the stage's later models reused them. **#164's slowdown did not recur**: every depth-12 model trained at 0.55–0.70 s a round, against #164's 0.66 s for its Arm and 4.5 s for its slow base. `max_depth` 14 took 1.40 s a round, about twice depth 12, as expected for trees with up to four times the leaves. The pre-registered grids were run in full; no stage needed cutting. 2023 was not read.

## 5. Other model families: LightGBM and CatBoost (#167)

**The idea.** Every model so far is XGBoost ([ADR 0002](../adr/0002-single-xgboost-model-for-scenarios.md), superseded by ADR 0008 but still the served family). Gradient-boosted trees come in other implementations that grow trees differently: LightGBM grows leaf-wise (it always splits the leaf with the largest gain, so trees are unbalanced), and CatBoost grows symmetric ("oblivious") trees, where every node at one depth uses the same split. This idea asks whether either one ranks held-out UK Markets better, on exactly the same problem. In course terms, it is the model-selection step applied to the learner itself, holding the data, the loss and the validation design fixed.

**Fair comparison.** Each family is one Arm of its own experiment (`loop model-family <family>`, `loop.MODEL_FAMILY_ARMS`), against the best so far as served (`served_base_arm`, whose XGBoost model is cached). The Arm trains through the trainer seam (#165) on the same rows as the base: the same `exporter_count` files, fold 0 held out, the non-UK hash sample at 0.6157 weighted 1/rate, Zero Cells at 0.05, the same 25 features and FeatureTransform, and the same stopping rule (the model alone's Selection Measure on the held-out UK grid every round, patience 100, cap 5,000 rounds). The Arm re-chooses its blend weight from 0.1 to 0.9 and keeps Market Reach at β = 4, as every Arm with a new model has. The Selection Measure, the Win Threshold (0.008869) and the win rule are unchanged.

**The monotone constraints.** ADR 0008 keeps the +1 constraints on the agreement and bloc flags and −1 on the Applied Tariff. Both families enforce them, checked on this machine before any 2022 result was read (with lightgbm 4.7.0 and catboost 1.2.10, on synthetic data, 0 violations of 20,000 for each):

- **LightGBM**: `monotone_constraints` with `monotone_constraints_method="basic"`, its strict method, which bounds each split's children at their midpoint as XGBoost does. The pip wheel has **no GPU learner** (`device_type="gpu"` and `"cuda"` both fail: "GPU Tree Learner was not enabled in this build"), so it trains on the CPU (16 threads).
- **CatBoost**: `monotone_constraints` works **only on the CPU** (the GPU refuses it: "change of option monotone_constraints is unimplemented for task type GPU"; the [docs](https://catboost.ai/docs/en/references/training-parameters/common) list "Supported processing units: CPU") and **only for symmetric trees** (Depthwise and Lossguide raise "Monotone constraints are unsupported for non-symmetric trees yet"). So it trains on the CPU with `grow_policy="SymmetricTree"`; the trainer refuses any other grow policy with the constraints on.

So neither family is skipped for the constraints. Both run on the CPU, which the spec allows if they fit in RAM and time. They never hold the raw rows in memory: the trainer writes the prepared rows once to a float32 file (about 8.4 GB for 84 M rows × 25 features) and builds the binned dataset from it. On a 10 M-row test, LightGBM's binned Dataset took about 0.5 GB per 10 M rows. CatBoost copied the rows into its own Pool (about 1 GB per 10 M rows) before quantizing them to about 0.4 GB, so its peak is about 8.4 GB above the loop's own tables.

**Where each family's settings differ from XGBoost's** (each the closest the library allows):

| | XGBoost (base) | LightGBM | CatBoost |
|---|---|---|---|
| Loss | `reg:tweedie`, p = 1.2655, log link | `tweedie`, p = 1.2655, log link | `Tweedie:variance_power=1.2655`, log link |
| Starting point | intercept estimated from the label | `boost_from_average` (log of the mean label) | none for Tweedie (`boost_from_average` refuses the loss): the trainer sets a constant baseline, log of the weighted mean label, and writes it into the model as its bias |
| Trees | depth-wise, `max_depth` 12 | leaf-wise, `max_depth` 12, `num_leaves` 1,023 | symmetric, `depth` 8 (256 leaves) |
| Histogram | 256 bins | `max_bin` 255 | `border_count` 254 |
| Device | GPU | CPU (no GPU learner in the wheel) | CPU (constraints are CPU-only) |
| Stopping metric | Selection Measure (custom metric) | the same, `feval` | the same, a custom `eval_metric`; CatBoost also computes it on the training rows, where the trainer returns 0, and never uses it |

**Pre-registered** (committed before either Arm was run). Each family's parameters are translated from the best so far's XGBoost parameters where the family has an equivalent; nothing was tuned.

| Family | Parameters | Why |
|---|---|---|
| LightGBM | `learning_rate` 0.1; `max_depth` 12; `num_leaves` 1,023; `min_sum_hessian_in_leaf` 3.047 (XGBoost's `min_child_weight`); `min_data_in_leaf` 1 (XGBoost has no row floor); `bagging_fraction` 0.9158 with `bagging_freq` 1 (`subsample`); `feature_fraction` 0.8795 (`colsample_bytree`); `lambda_l2` 4.954 (`reg_lambda`); `tweedie_variance_power` 1.2655 | The same depth and regularisation. Leaf-wise growth needs a leaf cap: 1,023 is a quarter of what a full depth-12 tree could hold, so the trees stay about as large as XGBoost's while the leaf-wise order chooses where. |
| CatBoost | `learning_rate` 0.1; `depth` 8; `l2_leaf_reg` 4.954 (`reg_lambda`); `bootstrap_type` Bernoulli with `subsample` 0.9158; `rsm` 0.8795 (`colsample_bytree`, chosen per split level on the CPU); `tweedie_variance_power` 1.2655; other settings at CatBoost's defaults (`random_strength` 1) | A symmetric tree of depth 12 has 4,096 leaves that all share each level's split, and would take hours per hundred rounds on the CPU. Depth 8 is inside CatBoost's usual range of 6 to 10 and gives 256 leaves. |

- **Gate:** each family is kept only if its Arm beats the best so far as served by more than the Win Threshold (0.008869). Order: LightGBM first, then CatBoost. A kept LightGBM would become the best so far, and CatBoost would then run against it.
- **The stopping rule:** an Arm whose best round is below 50 is flagged as decided by the stopping rule.
- **Not done:** two-part models and a starting point (neither trainer supports them; neither is in the best so far), and the GPU for either family.
- **Commands:** `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop model-family lightgbm`, then `catboost`. Each writes `experiments["model family: <family>"]` to the loop's `results.json`. 2023 is not read.

**Decision: drop both families; XGBoost stays the model.** Neither family beats the best so far as served. LightGBM scores **0.6205** (−0.0005) and CatBoost **0.6191** (−0.0019), against **0.6210** for the base and a Win Threshold of **0.008869**. `results["best"]` is unchanged (the Blend w = 0.3 without Market Reach, 0.5771), and the served files stay `exporter_count`. [ADR 0002](../adr/0002-single-xgboost-model-for-scenarios.md) still describes the served family, so it is not amended. #169 starts from `served_base_arm` of that best so far, as this idea did.

**Results.** Fold 0 of 2022, under the tie-proof measure, with Market Reach at β = 4. "Without Reach" is the same Arm's Blend without Market Reach. "Model alone" is the family's own model, unblended and without Reach, which is what training stopped on. Deviance is the Tweedie deviance on the validation rows. Both Arms re-chose w = 0.3, the base's weight.

| Arm | **Selection Measure** | Change | Without Reach | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance | Model alone | Rounds kept / trained | Training (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Best so far as served (XGBoost, base, cached) | **0.6210** | | 0.5762 | 0.507 | 0.589 | 0.816 | 0.943 | 464.1 | (0.5300 retrained in #164) | 824 / 924 | cached |
| LightGBM ⚑ | 0.6205 | −0.0005 | 0.5456 | 0.483 | 0.591 | 0.808 | 0.935 | 1,242.0 | 0.5092 | **3** / 103 | 487 s |
| CatBoost | 0.6191 | −0.0019 | 0.5728 | 0.491 | 0.588 | 0.812 | 0.941 | 565.2 | **0.5515** | 208 / 308 | 1,546 s |

**⚑ LightGBM was decided by the stopping rule.** Its model alone ranked best after **3** rounds (0.5092). It then scored lower every round for 100 rounds, so patience ended training at round 103. Its deviance (1,242.0, almost three times the base's) shows a model that had barely started to fit. So its −0.0005 reflects the stopping rule, not the family. It is the same pattern as #166's `subsample` 0.7 with `colsample_bytree` 0.6 (best round 6) and #71's depth 6: a model whose earliest trees rank about as well as it will for a while, which patience 100 cannot see past. A plausible cause is leaf-wise growth at `min_data_in_leaf` 1. Each early tree spends its 1,023 leaves on the largest gains first, which can be a few tiny, heavy groups. That is not tested here. A LightGBM with a larger leaf floor or fewer leaves might train its full course, but that would be a new pre-registered Arm, not a rescue of this one. The served Selection Measure would also have to rise by about 0.009 to win, and no Arm of this round's search moved it by a tenth of that. (The results file records LightGBM's trained rounds as 3: the trainer then kept only the best rounds, a bug fixed before CatBoost ran (`keep_training_booster=True`). The 103 follows from the stopping rule: best round 3 plus patience 100. The score is unaffected, because prediction used 3 rounds either way.)

**Reading it.**
- **CatBoost's own model ranks best of any model in the loop, but the gain does not survive the Blend.** Its model alone scores **0.5515**, against 0.5268–0.5378 for every XGBoost model of #166's search and 0.5300 for #164's retrained base. That is about +0.02, the largest model-alone gap this round. Blended with the Naive Baseline and scaled by Market Reach, it ranks **0.0019 below** XGBoost. This repeats #166's finding: once blended, the model barely matters for the served ranking. The Naive Baseline and Market Reach carry what CatBoost's symmetric trees add, and a little more. One reading, an association rather than a tested cause: CatBoost's model already holds more of the market-share signal that the Naive Baseline adds, so the Blend gains less from combining them.
- **Deviance and ranking disagree again.** CatBoost's deviance (565.2) is higher than XGBoost's (464.1), yet its model alone ranks better.
- **The constraints hold on the real grid** for both families' trained models (see the check below), not only on the fixture, where no family's prediction moves with a Trade Agreement at all.
- As always, these are associations on one fold and one year.

**Run.** On 2026-11-24, one family at a time, from the main checkout in the background: `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop model-family <family>`. **No GPU time**: both trained on the CPU (Ryzen 7 5800X, 16 threads). The cached XGBoost base was re-scored each time (about 108 s, including the usual warning that a model loaded on the CPU scores data on the GPU). The GPU held 6.0–6.3 GB for other programs throughout, and no other loop, pipeline or feature build was running.

| Family | Wall clock | Rows on disk and binned | Training | Per round | Peak memory |
|---|---|---|---|---|---|
| LightGBM | 15:51–16:01 (10.5 min) | 138 s | 349 s for 103 rounds | 3.4 s | 12.6 GB working set (the mapped rows included); free RAM fell to 2.2 GB |
| CatBoost | 16:04–16:32 (28.1 min) | 170 s | 1,376 s for 308 rounds | 4.5 s | 20.0 GB working set, 25.2 GB committed; free RAM fell to 2.0 GB, but nothing failed |
| **Total** | **38.6 min** | | **28.7 min of CPU training** | | |

Each family trained on 84,062,589 rows, the same count as #164's XGBoost models, so the sample and weights matched. LightGBM's run used the trainer as first committed. Before CatBoost ran, its trainer was changed to hand CatBoost a column-major copy of the rows on disk, which CatBoost reads in place: a row-major matrix it copies whole, which would have added about 8.4 GB. Most of CatBoost's 25 GB is its own training state, not the rows. Even so, 32 GB of RAM fits a depth-8 symmetric model on 84 M rows with about 8 GB held by other programs. 2023 was not read.

**The constraints on the real grid.** After the runs, each family's cached model scored all 231,217 held-out UK pairs of 2022 twice: with `trade_agreement` set to 0 and to 1, and with the Applied Tariff as it is and raised by one unit. This trains nothing and reads no 2023 data. The served form with and without the Blend gave the same counts.

| Family | A Trade Agreement lowers | … raises | A higher tariff raises | … lowers |
|---|---|---|---|---|
| LightGBM (3 rounds) | **0** | 8,430 | **0** | 9,001 |
| CatBoost | **0** | 231,217 | **0** | 170,423 |

Neither family ever moves the wrong way. CatBoost's symmetric trees split on the Trade Agreement in some tree at every pair, so the flag raises every pair's Export Potential. LightGBM's three trees use it for a few thousand pairs only.

## 6. A wider Exporter scope (#169)

**The idea.** [ADR 0004](../adr/0004-training-scope-and-sampling.md) trains on the rows of the top 100 Exporters by mean GDP. v1 trained on the whole world, and v2 ranked held-out UK Markets 0.020 worse. #73 ruled out the sampling parts of that change. That leaves the scope (or the Excluded Products) as the likely cause. This idea trains on every Exporter's rows, using #168's `all_exporters` files (section 2). It asks whether the 127 smaller Exporters' Flows and Zero Cells, as training rows, help the model rank UK Markets. In course terms, it tests whether more, and more varied, training data beats a cleaner sample: a point on the learning curve along a different axis from the row count.

**A paired comparison on one set of files.** In `all_exporters`, the top 100's rows are the same rows as in `exporter_count`, with the same features (section 2). So the base is the best so far as served, retrained on these files with only the top 100 Exporters' rows. A new training field, `Settings.exporter_scope`, does this: it keeps only the rows of the top *n* Exporters by mean GDP (`scope.exporter_scope`, the UK always in), ranked from the same country-year table the files were built from. On the real files it gives exactly `exporter_count`'s 100 Exporters. The two models share the validation rows, the held-out UK grid, the Naive Baseline, Market Reach's table and the feature transform (fitted on every training row of these files). So the pair differs only in which Exporters' rows train the model.

**One non-UK rate for both.** At the served rate (0.6157), every Exporter's rows would put about 115.9 M rows on the GPU, over the 100.0 M budget. The largest rate that fits these files is **0.5079** (`search.final_non_uk_rate`, section 2). Both models train at 0.5079, so the comparison isolates the scope. The base therefore trains on fewer rows than the served model (about 72 M against 84 M). If the base had kept 0.6157, the pair would differ in both scope and rate. The base is then not quite the served model. Its score shows what the lower rate costs on its own, against the served 0.6210. A kept Arm would serve at 0.5079.

**A new check.** The loop now refuses, before anything trains, an experiment whose explicit non-UK rate might not fit the VRAM row budget (`training.check_row_budget`, counted as `search.final_non_uk_rate` counts: every training row and the whole validation year). Until now only the pipeline checked it, and an Arm could have asked for 0.6157 on these files.

**Pre-registered** (committed before any 2022 result was read):
- **Base** (`best so far as served, top 100 Exporters`): the best so far (Blend w = 0.3, the served XGBoost parameters, 25 features), with Market Reach at β = 4 (`served_base_arm`), `exporter_scope` 100 and non-UK rate 0.5079, retrained on fold 0 of the `all_exporters` files.
- **Arm** (`every Exporter`): the same, with every Exporter's rows (`exporter_scope` None) at 0.5079. It re-chooses its blend weight from 0.1 to 0.9 on the measure with Market Reach, as every Arm with a new model has since #71.
- **Gate:** keep it only if it beats the base by more than `current_win_threshold` (0.008869). If it is dropped, the served files stay `exporter_count` (`files_kept_if_dropped: false`).
- **If fold 0 wins:** the Arm becomes the best so far. It is then confirmed on all five folds through the loop's existing confirmation step (`python -m exportspredictor.loop confirm`) before it is adopted, and ADR 0004 is amended. If it does not win, no confirmation is run.
- **The stopping rule:** an Arm whose best round is below 50 is flagged as decided by the stopping rule.
- **Command:** `$env:EXPORTSPREDICTOR_FEATURE_SET='all_exporters'; scripts/wt.ps1 python -m exportspredictor.loop wider-scope` (two models, on the GPU, one at a time). It writes `experiments["wider exporter scope"]` to the loop's `results.json`. 2023 is not read.

**Decision: drop the wider scope; ADR 0004's top 100 stays.** Every Exporter's rows score **0.6212** (0.62119) against **0.6212** (0.62117) for the top 100 of the same files at the same rate. That is a gain of **+0.00002**, against a Win Threshold of 0.008869. Without Market Reach it is +0.00001 (0.5764 against 0.5764). The best so far is unchanged (`results["best"]` is still the Blend without Market Reach), and the served files stay `exporter_count`: `kept_feature_set` ignores this experiment because it was dropped. ADR 0004 is not amended, and no five-fold confirmation was run.

**Results.** Fold 0 of 2022, 231,217 held-out UK pairs, 4,858 ranked Products, under the tie-proof measure with Market Reach at β = 4. Deviance is the Tweedie deviance on the validation rows of these files (every Exporter's 2022 rows), so it is lower than on `exporter_count` and comparable only within this table. "Model alone" is the stopping measure.

| Arm | Exporters | w | **Selection Measure** | Change | Without Reach | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance | Model alone | Rounds kept / trained | Training rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Best so far as served, top 100 Exporters (base) | 100 | 0.3 | **0.6212** | | 0.5764 | 0.509 | 0.589 | 0.815 | 0.943 | 384.2 | 0.5324 | 810 / 910 | 70,521,571 |
| Every Exporter (the Arm) | 227 | 0.3 (chosen) | **0.6212** | **+0.0000** | 0.5764 | 0.505 | 0.589 | 0.816 | 0.944 | 351.1 | 0.5330 | 1,032 / 1,132 | 81,543,474 |

The Arm's blend grid with Market Reach, from w = 0.1 to 0.9, reads 0.6191, 0.6207, **0.6212**, 0.6210, 0.6201, 0.6187, 0.6170, 0.6151 and 0.6127. It chose the base's own weight.

**Reading it.**
- **The scope does not change how held-out UK Markets are ranked.** The two models agree to the fourth decimal on the Selection Measure, with and without Market Reach, and on the model alone (0.5330 against 0.5324). The secondary measures move by under 0.004, in both directions: "non-zero" falls slightly (0.505 against 0.509) and value-weighted rises slightly (0.816 against 0.815).
- **The extra rows help the model fit, not rank.** The deviance falls by 9% (351.1 against 384.2), mostly on the smaller Exporters' own 2022 rows, which the base never trained on. That is the same pattern as Sibling Demand's (section 3): a better fit to dollar levels across all rows, with no change in how UK Markets are ordered within a Product.
- **So the scope is not the cause of v2's −0.020 against v1.** #73 had ruled out the sampling parts of the v1-to-v2 change. This experiment rules out the scope, under today's features, Blend and Market Reach. Of the causes #73 left open, that leaves the Excluded Products, or the change being specific to v1's form. Neither is tested here. Section 8 (#171) then tests the Excluded Products: −0.0007, dropped.
- **The lower rate costs nothing measurable.** The base at 0.5079 scores 0.6212, against 0.6210 for the served model at 0.6157 on `exporter_count` (sections 1, 4 and 5). Fewer sampled rows of other Exporters, 70.5 M against 84.1 M, leave the ranking where it was.
- As always, this is an association on one fold and one year. Training on every Exporter does not rank held-out UK Markets better in 2022, and that says nothing about why.

**Run.** `$env:EXPORTSPREDICTOR_FEATURE_SET='all_exporters'; scripts/wt.ps1 python -m exportspredictor.loop wider-scope`, on 2026-11-24 from 16:58 to 17:19 (21.3 min wall), started in the background from the main checkout. Before the run, `nvidia-smi` showed 1.3 GB of the 10 GB in use by other programs, with the GPU idle and no other loop, pipeline or feature build running. While the Arm trained, 4.1 GB was in use, this run included. Neither model was taken from a cache.

| Model | Matrices | Training (matrices included) | Rounds kept / trained | Per round |
|---|---|---|---|---|
| Base, top 100 (`e4506bb6b0b778b9`) | 86 s | 545 s | 810 / 910 | 0.50 s |
| Every Exporter (`4144fbfc2a9b61de`) | 84 s | 694 s | 1,032 / 1,132 | 0.54 s |
| **Total** | | **1,239 s = 20.6 min of GPU time** | | |

The new row-budget check passed for both models at 0.5079. On these files it refuses the served 0.6157, which would put about 115.9 M rows on the GPU. 2023 was not read.

## 7. More training rows through GPU external memory (#170)

**The idea.** [ADR 0004](../adr/0004-training-scope-and-sampling.md) samples other Exporters' rows at whatever rate fits the GPU's memory, in-core: 0.6157 on the served `exporter_count` files, about 84 M training rows. It kept one "last resort": once the model is tuned and the features are final, train on every row through XGBoost's GPU external memory. #53's learning curve had not levelled off at the in-core budget, but its UK Spearman was already flat. In course terms, this is the far end of the learning curve: does the model rank held-out UK Markets better with about 57% more training rows? Only a small gain is expected. Section 6 also points that way: lowering the rate from 0.6157 to 0.5079 cost nothing measurable.

**How the rows fit.** A new training field, `Settings.external_memory` (XGBoost only; False by default and then left out of `to_dict`, so every cached model keeps its key), builds the training rows as an `ExtMemQuantileDMatrix` instead of a `QuantileDMatrix`. Each training year is one batch. XGBoost keeps the compressed pages in host memory and streams them to the GPU as each round reads them, so the rows need not fit in VRAM. The held-out UK grid, which the Selection Measure is read on every round, stays in-core and is binned by the same cuts. An arm with the field on is exempt from the loop's row-budget check (#169), and the check still refuses an in-core arm at the same rate. Another family refuses the field before anything trains. With XGBoost 3.4.1 on this Windows machine, neither RMM nor the CUDA async pool is available (the pool is not supported on Windows), so XGBoost warns that iterating through external memory may be slow. A smoke test on 3 M real 2021 rows gave predictions identical to the in-core matrix's, at the same speed per round (0.32 s against 0.31 s).

**Not run: the wider scope unsampled.** The ticket's second Arm, every Exporter's rows unsampled, was conditional on the wider scope being kept. #169 dropped it (section 6), so that Arm is not run.

**Pre-registered** (committed before any 2022 result of this Arm was read):
- **Base** (`best so far as served`): the best so far (Blend w = 0.3, the served XGBoost parameters, 25 features) with Market Reach at β = 4 (`served_base_arm`), at the served non-UK rate 0.6157 on the `exporter_count` files: the cached model (`e5943f03616c0cf7`), 0.6210.
- **Arm** (`non-UK rate 1.0, external memory`): the same, with every other Exporter's row (non-UK rate 1.0, about 132 M training rows on fold 0) and `external_memory` on. It re-chooses its blend weight from 0.1 to 0.9 on the measure with Market Reach, as every Arm with a new model has since #71.
- **Gate:** keep it only if it beats the base by more than `current_win_threshold` (0.008869).
- **If fold 0 wins:** the Arm becomes the best so far. It is then confirmed on all five folds (about 12 h) through the loop's confirmation step before it is adopted, and a new ADR superseding ADR 0004's sampling rule records it. If it does not win, no confirmation is run.
- **The stopping rule:** an Arm whose best round is below 50 is flagged as decided by the stopping rule.
- **Command:** `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop external-memory` (one model, on the GPU, alone). It writes `experiments["external memory"]` to the loop's `results.json`. 2023 is not read.

**Decision: drop it; ADR 0004's sampled rows stay.** Every other Exporter's row scores **0.6209** (0.62094) against **0.6210** (0.62099) for the best so far as served. That is a change of **−0.00004**, against a Win Threshold of 0.008869. Without Market Reach it is −0.0015 (0.5747 against 0.5762). The best so far is unchanged, and the served files stay `exporter_count`. No five-fold confirmation was run and no ADR is written. ADR 0004's last resort has now been tried, and it does not help.

**Results.** Fold 0 of 2022, 231,217 held-out UK pairs, 4,858 ranked Products, under the tie-proof measure with Market Reach at β = 4. "Model alone" is the stopping measure.

| Arm | Non-UK rate | w | **Selection Measure** | Change | Without Reach | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance | Model alone | Rounds kept / trained | Training rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Best so far as served (base, cached) | 0.6157 | 0.3 | **0.6210** | | 0.5762 | 0.507 | 0.589 | 0.816 | 0.943 | 464.1 | 0.5327 | 824 / 924 | 84,062,589 |
| Non-UK rate 1.0, external memory (the Arm) | 1.0 | 0.3 (chosen) | **0.6209** | **−0.0000** | 0.5747 | 0.508 | 0.589 | 0.815 | 0.943 | 453.2 | 0.5265 | 742 / 842 | 132,324,129 |

The Arm's blend grid with Market Reach, from w = 0.1 to 0.9, reads 0.6189, 0.6204, **0.6209**, 0.6208, 0.6200, 0.6187, 0.6170, 0.6150 and 0.6127. It chose the base's own weight.

**Reading it.**
- **57% more rows do not change how held-out UK Markets are ranked.** With the Blend and Market Reach the two agree to the fourth decimal. The model alone is 0.006 lower (0.5265 against 0.5327) and so is the measure without Reach (−0.0015). Both are well inside the spread of models retrained in this round (a retrain is a fresh draw within the noise floor). The secondary measures move by under 0.001.
- **The extra rows help the fit a little, not the ranking.** Deviance falls by 2% (453.2 against 464.1), the pattern of sections 3 and 6: more rows from other Exporters fit dollar levels better, and leave the order of UK Markets within a Product where it was.
- **For ranking, the learning curve is flat around the in-core budget.** Section 6 went down from 84 M rows to 72 M and lost nothing. This goes up to 132 M, every row the files hold, and gains nothing. #53 saw the same for UK Spearman. ADR 0004's sampling costs nothing measurable on the Selection Measure, so the in-core budget stays.
- As always, this is an association on one fold and one year. More training rows do not rank held-out UK Markets better in 2022, and that says nothing about why.

**Run.** `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop external-memory`, from 2026-11-24 17:45 to 2026-11-25 02:00 (8 h 15 min wall), started in the background from the main checkout. Before the run, `nvidia-smi` showed 1.4 GB of the 10 GB in use by other programs, with the GPU idle and no other loop, pipeline or feature build running, and 20.9 GB of RAM free. The base was taken from the cache, so only the Arm trained.

| Model | Matrices | Training (matrices included) | Rounds kept / trained | Per round |
|---|---|---|---|---|
| Base (`e5943f03616c0cf7`, cached; in-core when it was trained) | 91 s | 680 s | 824 / 924 | 0.74 s |
| Non-UK rate 1.0, external memory (`bbc1ffc801435eed`) | 95 s | 29,615 s | 742 / 842 | **35.2 s** |
| **Total this run** | | **29,615 s = 493.6 min (8.2 h) of GPU time** | | |

**External memory on this machine is about 48 times slower per round than in-core**, at about the same round count. ADR 0004 estimated 2.3 h per model; it took 8.2 h. XGBoost could use neither heterogeneous memory management nor an RMM or CUDA async pool on Windows (it warned of both), so every round moved the pages over PCIe. While it trained, the GPU stayed at 100% with 9.4–10.0 GB of VRAM in use (this run included), and 12.4–14.8 GB of RAM was free. The lowest free RAM, 6.9 GB, came during the loop's setup. Nothing failed. A five-fold confirmation at this speed would take about 41 h, not the 12 h the ticket expected. 2023 was not read.

## 8. The leftover research leads (#171)

**What this covers.** User story 22 of #155: every `docs/research/` idea marked "untested" or "hypothesis", and the leads #140's sanity review marked for this round (its "flags marked for #155's idea 7"), each run as an Arm on top of the best so far if it uses only data we already have, or skipped with a reason. Ideas #163–#170 already ran (sections 1–7) are not run again. The best so far is unchanged by sections 1–7: `served_base_arm(best)`, the cached model `e5943f03616c0cf7` on the `exporter_count` files, 0.6210. The Win Threshold is 0.008869.

**The inventory.**

| # | Idea | Source | Data | Run or skip |
|---|---|---|---|---|
| a | **Train on the Excluded Products' rows** (ADR 0005 drops them from training as well as from the rankings) | [opportunity-ranking-experiments.md](opportunity-ranking-experiments.md) (#73: "not tested (out of scope)"); section 6 above; #155's expected sizes | existing (BACI) | **Run**: one paired Arm |
| b | **A tariff of 0 where a Trade Agreement is in force but TRAINS records no preference** (#161's flag H3: Canada, Israel, Serbia, Vietnam, Ghana, Côte d'Ivoire, Cameroon, Kenya, Türkiye for the UK) | #140, flag H3 | existing (the tariff and agreement tables) | **Run**: a scoring-only probe first, then a retrained Arm only if the probe passes its gate |
| c | The tariff's step response near 0 (H1, M1: a Tariff part is a step, not a slope), its form or its monotone constraint | #140, flags H1 and M1 | existing | **Skip**: see below |
| d | Year-to-year stability of Export Potential across the Gap Period, e.g. features smoothed over years (M3, M7) | #140, flags M3 and M7 | existing | **Skip**: see below |
| e | Non-tariff market access (SPS certificates, trade remedies) as a policy feature | [open-review-2022.md](open-review-2022.md), idea 6 (**hypothesis**) | new | **Skip**: new data (#172's further work) |
| f | Agreement depth (World Bank Deep Trade Agreements, DESTA) | #138 | new | **Skip**: new data (#172's further work) |
| g | Gravity for the 20 Markets without it; group the UK Overseas Territories | open review, idea 8 (partly **hypothesis**, untested) | partly new | **Skip**: see below |
| h | The loop's early stopping may prefer trees that flatten the bottom of the ranking | open review, section 2 ("a hypothesis worth checking") | existing | **Skip**: already answered (see below) |
| i | A UK-free demand proxy as a feature | open review, idea 9 (**hypothesis**) | existing | Already run: section 3 (#164), dropped |

**Why the skips.**
- **c, the tariff's step.** Trees split on thresholds, so any monotone change of the tariff's form (it is already ln(1 + t)) gives the same splits and the same step. Removing the monotone constraint was #75's Arm (−0.0064, inside the noise), and ADR 0008 kept the constraints. The step is a property of the Tariff part of a Scenario Effect, which the Selection Measure does not measure, so no Arm on fold 0 could decide it. A smooth tariff response needs another Model Form (a parametric tariff term beside the trees): further work.
- **d, stability across years.** The Selection Measure ranks one year, so it cannot reward stability across the Gap Period. Smoothed features would be new features and a full feature rebuild. No note has evidence that they rank better, and every feature tried since #127 has moved the measure by at most 0.0011 (the Exporter count +0.0011, Sibling Demand +0.0003). Expected gain below the Win Threshold: skipped, and named as further work.
- **g, gravity for small Markets.** #126 filled GDP and population for these Markets from secondary sources, and #127 kept the fill as a data correction (its own effect −0.0001; [exporter-count-2022.md](exporter-count-2022.md)). Distance for them is not in the gravity tables we hold, so filling it needs new data. Grouping the UK Overseas Territories is an Advisor-facing filter, not a model change, so it cannot be an Arm.
- **h, early stopping and ties.** #115 made the Selection Measure tie-proof (tied predictions are ordered by a stable hash of Product and Market), so a flatter bottom can no longer raise the measure the loop stops on. #116 replayed every stopping decision under it ([loop-recheck-tie-proof-2022.md](loop-recheck-tie-proof-2022.md)) and corrected none.

**a. Train on the Excluded Products.** v2 ranks held-out UK Markets 0.020 worse than v1 on the gain ladder. #73 ruled out the sampling parts of the v1-to-v2 change, and section 6 ruled out the Exporter scope. Of #73's open causes, the Excluded Products are the one left that can be tested: v1 trained on their rows, v2 does not. ADR 0005 drops them because their Flows (oil, gas, bullion, antiques, radioactive elements) are large and erratic, and the Tweedie loss would let them shape splits for every other Product. They are still never scored or ranked, so the measure is over the same Products either way.
- **The files.** A new feature set, `excluded_products`: `exporter_count` (the served files) with the Excluded Products' training rows kept (their Flows and their sampled Zero Cells, by the same hash rule). The UK grid still leaves them out. Every feature is built from all Flows in both sets (ADR 0005: "the exclusion limits the rows, not the features"), so every row the two sets share is identical.
- **A training field**, `Settings.train_excluded_products` (False by default, left out of `to_dict` and so of every cached key). False drops the Excluded Products' rows before training, as the loop always has; True keeps them.
- **One non-UK rate for both**: the largest rate these files allow under the 100 M row budget (`search.final_non_uk_rate`, `non_uk_rate` None), resolved once and recorded. The extra rows mean it is a little below the served 0.6157. The FeatureTransform is fitted on the same files for both models.
- **Base** (`best so far as served, Excluded Products dropped`): the best so far (Blend w = 0.3, the served XGBoost parameters, 25 features), Market Reach at β = 4 (`served_base_arm`), at that rate, retrained on fold 0 of the `excluded_products` files with their Excluded Products' rows dropped.
- **Arm** (`train on the Excluded Products`): the same with `train_excluded_products` True. It re-chooses its blend weight from 0.1 to 0.9 on the measure with Market Reach, as every Arm with a new model has since #71.
- **Gate:** keep it only if it beats the base by more than `current_win_threshold` (0.008869). If it is dropped, the served files stay `exporter_count` (`files_kept_if_dropped: false`). If it wins on fold 0, it becomes the best so far, is confirmed on all five folds before adoption, and ADR 0005 is amended. An Arm whose best round is below 50 is flagged as decided by the stopping rule.
- **Command:** `$env:EXPORTSPREDICTOR_FEATURE_SET='excluded_products'; scripts/wt.ps1 python -m exportspredictor.loop excluded-products` (two models, on the GPU). It writes `experiments["excluded products"]` to the loop's `results.json`.

**b. A tariff of 0 where an agreement records no preference.** TRAINS has no preference schedule for some Trade Agreement partners, so their pairs carry the MFN rate while the agreement dummy is 1. #140 marked it for this round and noted that the fix needs a retrain. A retrained Arm needs a rule for what a missing preference means, a full feature rebuild and two models. Before that, a scoring-only probe measures how much the correction moves the 2022 ranking through the model already trained:
- **The rule** (`tariffs.unrecorded_preferences`): a held-out UK pair's Market has an *unrecorded preference* when the UK has a Trade Agreement with it that year, no Customs Union, and in none of the fold's Products with a known MFN rate is the Applied Tariff below the MFN rate. The probe sets the Applied Tariff of those Markets' rows to 0.
- **The probe:** the cached base (`e5943f03616c0cf7`) re-predicts fold 0's 2022 UK grid with the corrected tariffs, scored as served (Blend w = 0.3, Market Reach β = 4, the Market Reach table unchanged). The Naive Baseline has no policy feature, so it does not move. Nothing is trained.
- **Gate for the retrained Arm:** run it only if the probe gains more than half the Win Threshold (0.0044). A retrained model also learns from corrected training rows, so it could gain more than the probe, but a probe below half the threshold leaves little room for a win. Otherwise the Arm is skipped, and the probe's figure is its reason.
- **Command:** `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python scripts/research/final_round_2022/unrecorded_preferences.py` (CPU or GPU prediction only). It writes `data/derived/final_round_2022/unrecorded_preferences.json`.

2023 is not read by either.

**Decision on a: drop it; ADR 0005 stays.** Training on the Excluded Products' rows scores **0.6207** (0.62072) against **0.6214** (0.62142) for the best so far as served retrained on the same files with those rows dropped. That is a change of **−0.0007**, against a Win Threshold of 0.008869. Without Market Reach it is −0.0027 (0.5741 against 0.5768). The best so far is unchanged, and the served files stay `exporter_count` (`files_kept_if_dropped: false`). No five-fold confirmation was run, and ADR 0005 is not amended.

**Results of a.** Fold 0 of 2022, 231,217 held-out UK pairs, 4,858 ranked Products, under the tie-proof measure with Market Reach at β = 4. Both models train at the non-UK rate these files allow, **0.6109** (the served 0.6157, less the room the Excluded Products' rows take). Deviance is on the validation rows, from which the loop drops the Excluded Products, so the two are comparable.

| Arm | Excluded Products | w | **Selection Measure** | Change | Without Reach | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance | Rounds kept / trained | Training rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Best so far as served, Excluded Products dropped (base) | dropped | 0.3 | **0.6214** | | 0.5768 | 0.507 | 0.590 | 0.815 | 0.943 | 467.1 | 1,003 / 1,103 | 83,459,416 |
| Train on the Excluded Products (the Arm) | trained | 0.3 (chosen) | **0.6207** | **−0.0007** | 0.5741 | 0.504 | 0.589 | 0.816 | 0.943 | 466.6 | 501 / 601 | 83,958,601 |

The Arm's blend grid with Market Reach, from w = 0.1 to 0.9, reads 0.6187, 0.6201, **0.6207**, 0.6207, 0.6198, 0.6185, 0.6168, 0.6149 and 0.6126. Its model alone scores 0.5317 (the stopping measure).

**Reading a.**
- **The Excluded Products' rows neither help nor hurt the ranking of the other Products.** They are few: about 80,000 rows a year, 0.6% of the training rows (499,185 of 84.0 M on fold 0 at this rate). The Arm sits 0.0007 below its base, well inside the noise of one retrain, and every secondary measure moves by under 0.003.
- **So they are not the cause of v2's −0.020 against v1.** With the sampling (#73) and the scope (section 6) also ruled out, none of the parts of the v1-to-v2 change that can be tried on today's form explains it. What is left is that the gap belonged to v1's form as a whole, which no Arm on today's form can test.
- **ADR 0005's reason was the Tweedie loss**: large, erratic Flows of oil and bullion might shape splits for every other Product. On this evidence they do not, measurably, but they add nothing either, so the exclusion stays. It also keeps them out of the Advisor's rankings and leaves the rows to the sample.
- As always, this is an association on one fold and one year.

**Result of b: the probe moves nothing, so the retrained Arm is skipped.** The rule marks **28 Markets** whose UK pair had a Trade Agreement in 2022 but no preference recorded on any fold-0 line: Albania, Antigua and Barbuda, Canada, Switzerland, Côte d'Ivoire, Cameroon, Costa Rica, Fiji, Georgia, Ghana, Haiti, Iceland, Israel, Kenya, Mauritius, Norway, Panama, Papua New Guinea, Palestine, Singapore, Solomon Islands, Serbia, Trinidad and Tobago, Tunisia, Türkiye, Vietnam, Samoa and Zimbabwe. The UK's continuity agreements are mostly missing from TRAINS's 2021 schedules, a wider set than #161's list. The probe set the Applied Tariff to 0 on **14,181** of their 28,626 fold-0 grid rows. The rest were already duty-free under MFN or had no MFN rate, so nothing changed for Switzerland, Palestine, Singapore or Tunisia.

| Fold 0 of 2022, cached base `e5943f03616c0cf7` | **Selection Measure** | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share |
|---|---|---|---|---|---|
| As served (base) | **0.62099** | 0.5072 | 0.5892 | 0.8156 | 0.9426 |
| Unrecorded preferences set to 0 (the probe) | **0.62089** | 0.5071 | 0.5891 | 0.8155 | 0.9427 |

The change is **−0.0001**, against the pre-registered gate of +0.0044 (half the Win Threshold), so the retrained Arm is not run. Lowering these Markets' tariffs raises their Export Potential in every Product with a positive MFN rate, roughly evenly, and so barely reorders the Markets within a Product. The correction remains a data fix for Scenarios (an agreement partner shown with MFN tariffs), which #140 raised with the owner. It is not a ranking gain.

**Runs.**
- The `excluded_products` feature files: `scripts/wt.ps1 python -m exportspredictor.features --set excluded_products --parts 8` (2012–2024), 32.8 min on the CPU, 5.6 GB. Every row they share with `exporter_count` is identical (checked on 2021), and so is the 2022 grid.
- The probe: `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python scripts/research/final_round_2022/unrecorded_preferences.py`, 26 s, prediction only (XGBoost fell back to CPU prediction, as in section 1). It reproduces the base's 0.62099 and writes `data/derived/final_round_2022/unrecorded_preferences.json`.
- The experiment: `$env:EXPORTSPREDICTOR_FEATURE_SET='excluded_products'; scripts/wt.ps1 python -m exportspredictor.loop excluded-products`, on 2026-11-25 from 03:04 to 03:22 (18.7 min wall), started in the background from the main checkout. Before the run, `nvidia-smi` showed 0.5 GB of the 10 GB in use, with the GPU idle, no other loop, pipeline or feature process, and 18.4 GB of RAM free. It writes `experiments["excluded products"]` to the loop's `results.json`.

| Model | Matrices | Training (matrices included) | Rounds kept / trained |
|---|---|---|---|
| Base, Excluded Products dropped (`5c750451ea0cdc96`) | 86 s | 680 s | 1,003 / 1,103 |
| Train on the Excluded Products (`cb4813ee21e64bc1`) | 79 s | 412 s | 501 / 601 |
| **Total** | | **1,091 s = 18.2 min of GPU time** | |

Neither Arm is decided by the stopping rule. 2023 was not read.

**Summary of section 8.** Nothing is kept. The best so far stays `results["best"]` (the Blend at w = 0.3, with Market Reach at β = 4 added by `served_base_arm`), on the `exporter_count` files.

| # | Idea | Source | Run or skipped | Result and decision, or reason |
|---|---|---|---|---|
| a | Train on the Excluded Products' rows | #73 ([opportunity-ranking-experiments.md](opportunity-ranking-experiments.md)), section 6 | **Run** (paired Arm, 2 models) | 0.6207 against 0.6214: −0.0007. **Dropped**; ADR 0005 stays |
| b | Tariff 0 where an agreement's preference is unrecorded | #140, flag H3 | **Probe run** (scoring only); the retrained Arm skipped by its gate | −0.0001 against the +0.0044 gate. **Skipped**: no ranking gain; a data fix for Scenarios |
| c | The tariff's step near 0 | #140, flags H1 and M1 | Skipped | Trees are invariant to the tariff's form; constraints off tested by #75; not measured by the Selection Measure |
| d | Year-to-year stability of Export Potential | #140, flags M3 and M7 | Skipped | A one-year measure cannot reward it; new features with no evidence of a ranking gain |
| e | Non-tariff market access | open review, idea 6 | Skipped | New data (#172's further work) |
| f | Agreement depth | #138 | Skipped | New data (#172's further work) |
| g | Gravity for 20 small Markets; group the Overseas Territories | open review, idea 8 | Skipped | GDP and population already filled (#126, #127); distance needs new data; grouping is not a model change |
| h | Early stopping flattening the bottom | open review, section 2 | Skipped | Answered by #115 and #116 |
| i | UK-free demand proxy as a feature | open review, idea 9 | Already run | Section 3 (#164): +0.0003, dropped |
