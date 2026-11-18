# Choosing the model on Opportunity ranking: the experiment loop and its five-fold confirmation

For spec #65 (tickets #67 and #70–#76), under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md). This note lists every experiment the loop ran, the arms that failed included, the rule that kept or dropped each one, and the five-fold confirmation of the result. Every figure is on the 2022 validation year. **The 2023 test year was not read** for any of it: the loop's data refuses it (`experiments.TestYearRead`). 2023 is read once, later, in the final run (#77).

Every result here is an association measured on held-out UK pairs in one year. An arm that "wins" ranked the UK's 2022 Markets better on that year's data. That does not say why, and it is not a forecast.

## The rule

- **Selection measure**: the mean over Products of the Spearman correlation between Export Potential and Actual Exports across Markets, on held-out UK pairs in 2022, on every non-Excluded Product. In the loop that is fold 0 of 5 (231,217 UK pairs, 4,858 ranked Products). It is the headline measure of the evaluation, moved to the validation year.
- **Reported, never selected on**: ranking on non-zero Flows only; ranking of "any Flow" (zero against non-zero); the Product ranking weighted by UK export value; the share of UK export value in the ten Markets with the highest Export Potential; Tweedie deviance (power 1.5) on the validation rows.
- **Rounds** are chosen by the selection measure of the model alone on the held-out grid each round. Training stops after 100 rounds without improvement (max 5,000). Deviance never stops training.
- **Noise floor**: the v2 configuration trained with seeds 0, 1 and 2. The seed changes both the non-UK row sample and XGBoost's seed. Selection measures were 0.5334, 0.5285 and 0.5374, with a sample SD of 0.00444.
- **Win threshold** = max(2 × SD, 0.005) = **0.00887**. In each experiment, the arm with the highest selection measure is kept only if it beats the best so far by more than the threshold. Otherwise every arm is dropped. Arms run on seed 0 so that comparisons are paired. The first best so far was seed 0 (0.5334).
- **Naive baseline** (the UK's leave-one-out supply × the Market's leave-one-out demand ÷ world trade in the Product) on the same pairs: **0.5472**. Non-zero only 0.450, zero against non-zero 0.509, value-weighted 0.765, top-10 share 0.915.

The rule, the order of the experiments and the budget were fixed in code (`exportspredictor.loop`) before any experiment was read. No arm was added or re-run after its result was seen.

## Every arm

Columns: selection measure; change against the experiment's base; naive weight w of the blend (chosen per arm from 0.1–0.9 on the same grid); the model alone (w = 0) where recorded; non-zero ranking; zero against non-zero; value-weighted; top-10 share; deviance; best round / trained rounds.

| Experiment (ticket) | Arm | Selection | Change | w | Model alone | Non-zero | Zero vs non-zero | Value-weighted | Top-10 | Deviance | Rounds | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Noise floor (#67) | seed 0 | 0.5334 | – | – | 0.5334 | 0.525 | 0.486 | 0.768 | 0.937 | 371.6 | 951 / 1,051 | base |
| | seed 1 | 0.5285 | – | – | 0.5285 | 0.525 | 0.481 | 0.765 | 0.935 | 374.6 | 643 / 743 | – |
| | seed 2 | 0.5374 | – | – | 0.5374 | 0.527 | 0.491 | 0.773 | 0.935 | 375.5 | 1,785 / 1,885 | – |
| 1–2. Naive baseline (#70) | blend | **0.5771** | +0.0438 | 0.3 | 0.5334 | 0.519 | 0.537 | 0.797 | 0.941 | 448.7 | 951 / 1,051 (cached) | **kept** |
| | starting point | 0.5621 | +0.0287 | – | 0.5621 | 0.483 | 0.524 | 0.777 | 0.939 | 795.3 | **13** / 113 | dropped |
| 3. Select on ranking (#71) | v1 | 0.5543 | −0.0228 | 0.1 | – | 0.464 | 0.515 | 0.770 | 0.929 | 2,421.8 | **1** / 101 | dropped |
| | search trial 15 (= v2, cached) | 0.5771 | 0 | 0.3 | – | 0.519 | 0.537 | 0.797 | 0.941 | 448.7 | 951 / 1,051 | dropped |
| | search trial 23 | 0.5793 | +0.0021 | 0.3 | – | 0.519 | 0.540 | 0.798 | 0.938 | 448.5 | 1,697 / 1,797 | dropped (best, inside the noise) |
| | search trial 19 | 0.5764 | −0.0007 | 0.3 | – | 0.523 | 0.536 | 0.798 | 0.941 | 449.1 | 1,858 / 1,958 | dropped |
| | depth 6 | 0.5645 | −0.0127 | 0.1 | – | 0.487 | 0.524 | 0.772 | 0.937 | 1,010.3 | **12** / 112 | dropped |
| 4. Two-part model (#72) | two-part | 0.5627 | −0.0144 | 0.4 | 0.4978 | 0.508 | 0.522 | 0.787 | 0.942 | 547.1 | binary **14** / 114; size 400 / 500 | dropped |
| 5. Scope (#73) | UK Zero Cells sampled (5%, weight 20) | 0.5763 | −0.0008 | 0.3 | 0.5346 | 0.517 | 0.536 | 0.794 | 0.942 | 443.7 | 756 / 856 | dropped |
| | zero rate 2% | 0.5777 | +0.0006 | 0.3 | 0.5355 | 0.520 | 0.538 | 0.796 | 0.940 | 439.0 | 1,009 / 1,109 | dropped |
| 6. Product detail (#74) | + trading_pair_share | 0.5750 | −0.0021 | 0.3 | 0.5264 | 0.521 | 0.535 | 0.798 | 0.945 | 436.0 | 759 / 859 | dropped |
| | + importing_markets | 0.5752 | −0.0020 | 0.3 | 0.5245 | 0.521 | 0.535 | 0.796 | 0.944 | 435.8 | 801 / 901 | dropped |
| | + hs4 | 0.5761 | −0.0011 | 0.3 | 0.5262 | 0.520 | 0.536 | 0.797 | 0.942 | 440.2 | 815 / 915 | dropped |
| 7. Monotone constraints (#75) | constraints on (cached) | 0.5771 | 0 | 0.3 | 0.5334 | 0.519 | 0.537 | 0.797 | 0.941 | 448.7 | 951 / 1,051 | dropped (= base) |
| | constraints off | 0.5708 | −0.0064 | 0.4 | 0.5207 | 0.512 | 0.531 | 0.792 | 0.944 | 510.7 | 1,115 / 1,215 | dropped |

The scope arms and the Product-detail arms are separate experiments in the code, run in order, so that each could be kept on its own. The blend's rank-average reference (the within-Product mean of the model's and the naive baseline's ranks, with no dollar value) scored 0.5739. It was never a candidate.

**One change was kept: the blend with the naive baseline (#70).** Export Potential = exp(0.7 · log model + 0.3 · log naive), both floored at $1. On fold 0 the weight's peak is flat: w = 0.4 scored 0.5769 against 0.5771 at 0.3. Every trained arm in #71–#75 landed within +0.003 of the blend or below it.

GPU budget: 13 trained models across the experiments (noise floor 3, starting point 1, select 4, two-part 2, scope 2, Product detail 3, constraints off 1). Each experiment stayed within the six-model limit. A 14th model timed one arm on a quiet machine (666 s).

## What the arms say

- **The naive baseline holds what the model lacks.** v2 alone ranks non-zero Flows better than the naive baseline (0.525 against 0.450) and "any Flow" worse (0.486 against 0.509). The blend keeps most of the first and gains the second (0.537), and it beats both of its parts.
- **Giving the model the naive baseline as its starting point is worse than blending.** At the v2 learning rate the trees improve on the starting point for 13 rounds and then make the ranking worse.
- **Configuration barely matters once blended.** The three confirmed v2 configurations sit within 0.003 of each other.
- **Neither part of the v1-to-v2 sampling change moves the ranking.** That leaves the gain ladder's −0.020 with the Exporter scope or the Excluded Products, which were not tested (out of scope).
- **Product-level constants lower the model-alone ranking a little** (about 0.525 against 0.533), though they lower deviance too.
- **The monotone constraints cost the ranking nothing measurable.** Without them the selection measure was 0.0064 lower (inside the noise, in the wrong direction) and deviance was worse. So they stay, and no further ADR is needed. This answers ADR 0008's open point.

## Arms decided by the stopping rule, not by their configuration

Four arms reached their best round within the first 50, then stopped 100 rounds later:

| Arm | Best round | Trained |
|---|---|---|
| v1 (#71) | 1 | 101 |
| depth 6 (#71) | 12 | 112 |
| starting point (#70) | 13 | 113 |
| two-part, binary part (#72) | 14 | 114 |

For these arms, the result reflects the stopping rule as much as the configuration. Their model-alone ranking peaks on the first trees, probably on a demand-driven split. It then dips, and patience 100 stops training before it can recover, if it would. **The stopping rule's caveat** (decision log, orchestrator entry "Stop rounds on the model's own measure or on the blended Export Potential?"): rounds stop on the model alone, while the arm is selected on the blend. Under a rule that stopped on the blend, or with more patience, these four arms might have scored higher. The rule was kept for the whole run for two reasons. Changing it after seeing results would have been adding arms during the run, which the spec forbids (story 18). And every arm stayed paired with the cached base model. The one fully trained alternative configuration (search trial 23, 1,697 rounds) was still inside the noise (+0.0021). That makes it unlikely, though not proven, that a short-stopping arm would have cleared +0.0089.

## The five-fold confirmation (#76)

The one-fold win was checked on every UK pair before the test year is read. The best arm ("winner": the blend, w = 0.3) and the current v2 configuration (the same model alone, i.e. the noise floor's seed 0) were trained on each of the five folds (2012–2021) and measured on 2022. Each fold's model scores only the UK pairs it held out (ADR 0001). The winner differs from v2 only in how the model is scored, so both use the same fold model. Fold 0 came from the loop's cache, and folds 1–4 were trained, four models in all, 3,068 s of training. Rounds were chosen per fold by the loop's rule: 951, 1,075, 1,097, 1,303 and 1,120. No round count was near the 5,000 cap or under 50. Command: `python -m exportspredictor.loop confirm` (`loop.confirm`).

**The five-fold measure** pools the five folds' held-out pairs, every UK pair of 2022: 1,156,050 pairs and 5,084 ranked Products. Each Product's Markets are then ranked once, as the headline measure ranks them. The blend weight stays at 0.3 and is not re-chosen here. Deviance is the mean of the five folds' deviances on their validation rows.

| Five folds, 2022 | Selection measure | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Deviance |
|---|---|---|---|---|---|---|
| **Winner: blend, w = 0.3** | **0.5635** | 0.501 | 0.524 | 0.793 | 0.617 | 449.3 |
| v2 (the model alone) | 0.5206 | 0.512 | 0.474 | 0.775 | 0.624 | 371.0 |
| Naive baseline | 0.5332 | 0.434 | 0.496 | 0.759 | 0.554 | – |

- **Winner − v2 = +0.0429, against a win threshold of 0.00887: the winner beats v2 by about five times the threshold.** It also beats the naive baseline by +0.030, which v2 alone does not do (−0.013). That repeats on 2022 what the audit found on 2023.
- **Fold by fold**, the winner beats v2 on every fold, by +0.040 to +0.045. The table below gives each fold's own measure on its own pairs. The mean over folds is 0.5751 for the winner against 0.5320 for v2 (+0.0431). The pooled measure is lower than any single fold's because each Product's Markets are ranked across the five fold models at once, and the five models' levels differ a little.

| Fold | UK pairs | v2 | Winner | Difference | Rounds | Trained here |
|---|---|---|---|---|---|---|
| 0 | 231,217 | 0.5334 | 0.5771 | +0.0438 | 951 | no (loop cache) |
| 1 | 231,663 | 0.5295 | 0.5745 | +0.0450 | 1,075 | yes |
| 2 | 232,241 | 0.5338 | 0.5754 | +0.0415 | 1,097 | yes |
| 3 | 230,602 | 0.5287 | 0.5739 | +0.0452 | 1,303 | yes |
| 4 | 230,327 | 0.5348 | 0.5748 | +0.0400 | 1,120 | yes |

- **What the blend trades away**: deviance (449 against 371) and a little of the non-zero ranking (0.501 against 0.512) and of the top-10 share. It gains on "any Flow" (0.524 against 0.474). ADR 0008 anticipated this: Export Potential's size is estimated less well, while the Markets are ranked better.
- The threshold comes from one fold's seed noise, so it is applied here as the spec's fixed rule, not as a five-fold confidence interval. The margin is wide enough that this caveat does not change the reading.

## The chosen form and configuration

**Form: a blend of the v2 model with the naive baseline in log dollars.** Export Potential = exp(0.7 · log model + 0.3 · log naive), with each floored at $1 (`loop.blend`, `naive_weight` 0.3). It stays in dollars, so the Export Gap is still defined. The naive baseline has no policy features, so a Scenario moves only the model's part, and every Scenario Effect shrinks by the factor 0.7 on the log scale. On fold 0, an added Trade Agreement was associated with +36.9% for the model alone and +23.9% for the blend, with no pair lower (#75).

**Configuration** (unchanged from v2, `results["best"]["settings"]`):

- One Tweedie XGBoost model with the saved search winner's parameters (trial 15): `max_depth` 12, `min_child_weight` 3.047, `subsample` 0.916, `colsample_bytree` 0.879, `reg_lambda` 4.954, `tweedie_variance_power` 1.266, `eta` 0.1.
- The 25 features of `features.FEATURES`, with the monotone constraints on policy features kept (#75).
- Non-UK row rate 0.6157, other Exporters' Zero Cells at 5%, and every UK Zero Cell kept at weight 1. Seed 0.
- Rounds chosen per fold by the selection measure of the model alone (patience 100, max 5,000): 951 / 1,075 / 1,097 / 1,303 / 1,120.
- No starting point, no two-part model, no Product-detail features.

The five confirmed fold models are in `data/derived/loop_V202601/models/`: `a9314febb8f2e546` (fold 0), `47a1d779af67057c` (1), `f5531a1394b61d75` (2), `6a511c331b5547e0` (3) and `2b7d1510b9dca2b0` (4), each `.ubj` with a `.json` holding its rounds. The final run (#77) builds the served artefact with this form and reads 2023 once.

## Reproducing

`python -m exportspredictor.loop noise | naive | select | two-part | scope | product-detail | constraints | confirm` (`src/exportspredictor/loop.py`). The results are in `data/derived/loop_V202601/results.json` (sections `noise_floor`, `experiments`, `best` and `confirmation`). The fold models are in `data/derived/loop_V202601/models/`, and each fold's scored held-out grid is in `data/derived/loop_V202601/confirmation/fold_<k>.parquet`. A re-run reads the cache and trains nothing. GPU training is not bit-reproducible: retraining the seed-0 settings gave 0.5299 against 0.5334, which is a fresh draw within the noise floor.
