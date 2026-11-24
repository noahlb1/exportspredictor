# The final round on the 2022 validation year

For spec #155, under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md), [ADR 0001](../adr/0001-out-of-fold-uk-training.md) and [ADR 0009](../adr/0009-test-year-read-once-per-frozen-version.md). The model is about to ship. This round tries the ideas left that use only data we already have. Each idea runs through the experiment loop (`exportspredictor.loop`) as one or more Arms on fold 0 of 2022. It is kept only if its Selection Measure beats the best so far by more than the Win Threshold. Winners stack: each idea starts from the ones kept before it.

**2023 is not read during the round.** The loop's data refuses the test year. If anything is kept, all winners become one new frozen version, and 2023 is read once for it through the Test-year Ledger.

## Summary

One row per idea, in the order run. Later tickets add their rows here and their own `## ` section below. "Base" is the best so far as served at the start of the idea. "Gain" is the Arm's Selection Measure minus the base's.

| # | Idea (ticket) | Arms | Base | Best Arm | Gain | Win Threshold | GPU time | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | [Cap on Market Reach's upside](#1-a-cap-on-market-reachs-upside-163) (#163) | 1 (4 caps) | 0.6210 | cap 16×, 0.6204 | −0.0006 | 0.008869 | none (scored from cache, 5.5 min) | **Dropped** |
| 2 | [Sibling Demand, a UK-free demand proxy, as a feature](#3-sibling-demand-a-uk-free-demand-proxy-as-a-feature-164) (#164) | 1 | 0.6210 (retrained on `sibling_demand`) | w = 0.3, 0.6213 | +0.0003 | 0.008869 | 76.6 min (2 models; see the section) | **Dropped** |

Not an idea, so not a row: [section 2](#2-wider-exporter-scope-feature-files-168) records the build of the `all_exporters` feature files (#168), which the wider Exporter scope experiment (#169) trains on.

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
