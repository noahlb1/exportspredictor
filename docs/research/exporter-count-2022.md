# The Exporter count and the gravity fill on the 2022 validation year

For spec #113, Phase 2, ticket #127, under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md) and [ADR 0001](../adr/0001-out-of-fold-uk-training.md). The ranking-error review ([ranking-error-analysis-2022.md](ranking-error-analysis-2022.md), candidate F1) found that the number of other Exporters supplying a Product to a Market was the strongest candidate feature. As a re-ranking, it added +0.030 to the Blend. #126 rebuilt the feature files with it (`log_exporters_loo`, the spec's "supplier count") and with GDP and population filled for small Markets (the gravity fill). This note runs both through the experiment loop. It uses the tie-proof Selection Measure (#115), the recomputed Win Threshold (#116) and Market Reach at β = 4 (#124).

**Decision: drop the Exporter count, and keep the gravity fill as a data correction.** With Market Reach, the Exporter-count Arm scores 0.6179 against 0.6168 for the base retrained on the same files. That is a gain of **+0.0011** against a Win Threshold of **0.008869**, so the Arm is dropped. No five-fold confirmation was run. The gravity fill's own effect is −0.0001 with Market Reach and −0.0010 without, inside the noise of one retrain.

**2023 was not read.** Every figure is on fold 0 of the 2022 validation year (231,172 held-out UK pairs, 4,858 ranked Products). The Market Reach offsets come from the same feature set's 2021 UK grid.

Commands:
- `$env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python -m exportspredictor.loop exporter-count` decides the experiment and writes it to the loop's `results.json` under `experiments["exporter count"]`.
- `scripts/research/exporter_count_2022/report.py` re-scores the cached models for the dollar spread, the large-Market measure and the Arm at the base's weight. It writes `data/derived/exporter_count_2022/result.json`.

## How the Arms were set up

- **Base Arm** (`base on rebuilt files`) is the best so far, unchanged, retrained on fold 0 of the `exporter_count` files. That means v2's parameters, the 25 features, the monotone constraints, non-UK rate 0.6157, Zero Cells at 5% and the Blend at w = 0.3, scored with Market Reach at β = 4. #116 found that no decision needed correcting.
  - The loop's new `base_arm` option (`Loop.run_experiment`) runs it first on the rebuilt files and decides the other Arm against it, so the comparison is paired.
  - It keeps w = 0.3, the weight Market Reach was adopted at.
- **Exporter-count Arm** (`+ log_exporters_loo`) is the base with the Exporter count as a 26th feature. It has no monotone constraint. Like every Arm with a new model since #71, it re-chooses its blend weight from 0.1 to 0.9, on the measure with Market Reach.
- **Market Reach in the loop.** Reach is the scoring field `Settings.market_reach_beta`, so it trains nothing. Each held-out pair's Export Potential is multiplied by exp(4 · offset) of its Market, using fold 0's offsets from `market_reach.reach_table` on the 2021 UK grid, then rescaled to keep the grid's sum.
  - Check: the cached base on the original files scores **0.61693** with Reach in the loop, which reproduces #124's fold-0 figure of 0.6169.

## Results

All figures are on fold 0 of 2022 and use the tie-proof measure. "Top 30" and "Top 60" are the measure on the 30 and 60 largest importing Markets (#123), which are reported but never selected on. "Spread" is the 10th–90th spread of log(1 + actual) − log(1 + Export Potential) on the 937 fold-0 pairs above $10m (#124's dollar spread; #124's pooled figures cover five folds). Deviance is the Tweedie deviance on the validation rows.

| Arm | Files | w | Reach | **Selection Measure** | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance | Spread (median) | Top 30 | Top 60 | Rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cached base (#116) | original | 0.3 | – | 0.5771 | 0.519 | 0.537 | 0.797 | 0.941 | 448.7 | 3.01 (+0.97) | 0.4839 | 0.5363 | 951 |
| Cached base | original | 0.3 | β 4 | 0.6169 | 0.507 | 0.585 | 0.812 | 0.943 | 459.1 | 3.69 (+0.90) | 0.5370 | 0.5966 | 951 |
| Base retrained | exporter_count | 0.3 | – | 0.5762 | 0.517 | 0.536 | 0.797 | 0.938 | 453.2 | 2.96 (+0.89) | 0.4747 | 0.5335 | 824 |
| **Base retrained** (the experiment's base) | exporter_count | 0.3 | β 4 | **0.6168** | 0.507 | 0.585 | 0.812 | 0.941 | 463.8 | 3.70 (+0.82) | 0.5359 | 0.5956 | 824 |
| + Exporter count | exporter_count | 0.4 | – | 0.5765 | 0.511 | 0.537 | 0.795 | 0.944 | 518.3 | 2.98 (+0.85) | 0.4745 | 0.5259 | 514 |
| **+ Exporter count** (the Arm) | exporter_count | 0.4 | β 4 | **0.6179** | 0.506 | 0.586 | 0.813 | 0.942 | 531.6 | 3.62 (+0.77) | 0.5334 | 0.5942 | 514 |
| + Exporter count, base weight | exporter_count | 0.3 | β 4 | 0.6174 | 0.506 | 0.585 | 0.811 | 0.942 | 460.7 | 3.67 (+0.77) | 0.5405 | 0.5999 | 514 |

The Arm's blend grid with Reach, from w = 0.1 to 0.9, reads 0.6143, 0.6163, 0.6174, **0.6179**, 0.6179, 0.6173, 0.6162, 0.6146 and 0.6126. Its model alone, without Reach or a blend, scores 0.5300, against 0.5327 for the base model alone (its stopping measure).

**The Exporter count.**
- With Reach, it adds +0.0011 at its own weight and +0.0006 at the base's weight. Without Reach it adds +0.0003. All of these are far below the 0.008869 threshold.
- The review's +0.030 came from re-ranking the Blend after the fact. A trained model already gets most of that information from `log_demand_loo` and the other demand features, and Market Reach takes the rest of the presence effect: "zero against non-zero" is 0.585 with Reach whether or not the feature is in.
- Its model stops earlier (514 rounds against 824) and has a higher deviance (531.6 against 463.8).
- It does not help on the largest Markets either: top 30 is 0.5334 against 0.5359, and top 60 is 0.5942 against 0.5956.

**The gravity fill.** "Base retrained" minus "Cached base" measures the fill. It also includes the 45 fold-0 pairs that #126's grid no longer holds: the original 2022 grid was the Gap Period union of 2022–2024 (231,217 fold-0 pairs, against 231,172 now).
- The effect is −0.0010 without Reach and −0.0001 with Reach.
- That is well within one retrain's noise: the Noise Floor's seeds have an SD of 0.0044, and a GPU retrain is a fresh draw. The round count moving from 951 to 824 is noise of the same kind (the seeds stopped between 643 and 1,785).
- The fill gives GDP to 10 small Markets and population to 14. It moves no ranking measurably, which is expected, because those Markets are a tiny share of the grid's pairs.
- It is kept, as the spec says, because it is a data correction: those Markets stop falling down the missing-value branch. It is not an Arm that wins or loses.

**Market Reach on the retrained base** adds +0.0407 (0.5762 to 0.6168), the same as on the cached base (+0.0398). The dollar spread above $10m widens from 2.96 to 3.70. This is the trade-off #124 reported on five folds (3.09 to 3.74).

## GPU time

On the RTX 3080, one run at a time, each backgrounded with a logged PID and a bounded wait:

| Model | Setup | Matrices | Training (matrices included) | Rounds kept / trained | Run total |
|---|---|---|---|---|---|
| Base retrained (`e5943f03616c0cf7`) | 18 s | 91 s | 680 s | 824 / 924 | 699 s |
| + Exporter count | 18 s | 91 s | 452 s | 514 / 614 | 460 s for the Arm's row; the base row was scored from the cache in 88 s |

The spec's budget was about 11 minutes per fold-0 model. Both models stayed within it, and the five-fold confirmation (about 50 minutes) was not needed.

## The configuration the rebuild (#128) will use

Nothing in `best` changed, so the loop's served configuration is the current one, on the rebuilt files, with Market Reach:

- **Features**: the 25 of `features.FEATURES`. `log_exporters_loo` stays a candidate (`CANDIDATE_FEATURES`), stored in the files but not used.
- **Model Form**: the Blend, w = 0.3, with the monotone constraints. Market Reach is served by #125 at β = `market_reach.CHOSEN_BETA` (4).
- **Parameters**: v2's (search trial 15), non-UK rate 0.6157, Zero Cells at 5%.
- **Rounds**: 1,097, the median of the five-fold confirmation's rounds (`loop.served_configuration`). They were confirmed on the original files and not re-confirmed on the rebuilt ones. The fold-0 retrain's 824 is within the seeds' noise.
- **Feature set**: `exporter_count`, rebuilt for 2012–2024 with `--overwrite`.
  - The loop records it in the experiment's section, and `served_configuration` hands it to the pipeline.
  - The pipeline refuses to build from other files (`EXPORTSPREDICTOR_FEATURE_SET` must name it) and records `feature_set` in the artefact metadata.
  - `ledger.FINGERPRINT_KEYS` hashes `feature_set` whenever it is set, so the new version cannot share a fingerprint with v1, v2 or the Blend. Those three keep theirs.
- **Validation year**: 2022. The pipeline still refuses loop results from another validation year.

These are associations measured on one fold and one year. Neither result says that a Market's number of suppliers, or its GDP, causes UK exports.
