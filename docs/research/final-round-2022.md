# The final round on the 2022 validation year

For spec #155, under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md), [ADR 0001](../adr/0001-out-of-fold-uk-training.md) and [ADR 0009](../adr/0009-test-year-read-once-per-frozen-version.md). The model is about to ship. This round tries the ideas left that use only data we already have. Each idea runs through the experiment loop (`exportspredictor.loop`) as one or more Arms on fold 0 of 2022. It is kept only if its Selection Measure beats the best so far by more than the Win Threshold. Winners stack: each idea starts from the ones kept before it.

**2023 is not read during the round.** The loop's data refuses the test year. If anything is kept, all winners become one new frozen version, and 2023 is read once for it through the Test-year Ledger.

## Summary

One row per idea, in the order run. Later tickets add their rows here and their own `## ` section below. "Base" is the best so far as served at the start of the idea. "Gain" is the Arm's Selection Measure minus the base's.

| # | Idea (ticket) | Arms | Base | Best Arm | Gain | Win Threshold | GPU time | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | [Cap on Market Reach's upside](#1-a-cap-on-market-reachs-upside-163) (#163) | 1 (4 caps) | 0.6210 | cap 16×, 0.6204 | −0.0006 | 0.008869 | none (scored from cache, 5.5 min) | **Dropped** |

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
