# The experiment loop re-checked under the tie-proof Selection Measure

For spec #113, ticket #116, under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md) (amended by #115). Spec #65's loop (`docs/research/opportunity-ranking-experiments.md`) chose every setting with the old Selection Measure, which used average ranks for ties on both sides, so tied predictions could raise it. This note re-scores each of its decisions with the tie-proof measure (`exportspredictor.selection_measure.selection_measure`) and says whether each one stands.

**Nothing was trained and 2023 was not read.** Every figure is on the 2022 validation year. Every model came from the loop's cache (`data/derived/loop_V202601/models`) and predicted on the CPU. The cached models and feature files were not changed.

Scripts: `scripts/research/loop_recheck_2022/rescore.py` (CPU, about 23 minutes) and `apply.py`. Outputs are in `data/derived/loop_recheck_2022/`: `rescore.json`, `curves.parquet` (the measure after every saved round of every model), `decisions.json`, and `results.before_recheck.json` (the loop's results before the re-check).

## How the round counts were re-checked

Early stopping cuts the trees but does not change them. So the held-out grid was scored one tree at a time, which gives the tie-proof measure of the model alone after every saved round. The loop's rule was then replayed: stop 100 rounds after the last improvement. Each cached model holds its best round plus 100. If the replay stops inside the saved trees, the round it picks is exactly what a re-run would pick. If it does not stop, the round is **censored**: the measure was still rising where training ended, and only a GPU retrain could say how far it would go. At the re-picked round, every Arm with a blend grid re-chose its naive weight from 0.1 to 0.9, as the loop does.

Check: the old measure, recomputed here at the recorded rounds, matches `results.json` to within 2e-6 on every Arm and fold. The CPU and GPU predictions differ only at float precision.

## Noise Floor and Win Threshold

| Seed | Old measure | Tie-proof | Round (old → tie-proof) |
|---|---|---|---|
| 0 | 0.53339 | 0.53339 | 951 → 951 |
| 1 | 0.52854 | 0.52854 | 643 → 643 |
| 2 | 0.53739 | 0.53739 | 1,785 → 1,785 |
| SD | 0.0044352 | 0.0044346 | |
| **Win Threshold** = max(2 × SD, 0.005) | 0.008870 | **0.008869** | |

v2's fully trained models hardly tie, so the noise is the same. **The recomputed Win Threshold is 0.008869.** It is saved in the loop's results (`results.json` → `recheck.noise_floor.win_threshold`). The loop reads it through `loop.current_win_threshold`, which every later experiment (`Loop.run_experiment`) and the five-fold confirmation (`loop confirm`) now use. #124 and #127 read it the same way.

## Every decision, old against tie-proof

Fold 0, 231,217 held-out UK pairs. "Change" is against the experiment's base under the tie-proof measure: seed 0 (0.53339) for the first experiment, the Blend (0.57715) for the rest.

| Experiment | Arm | Old | Tie-proof at old round | Tie-proof, round re-chosen | w (old → new) | Round (old → new) | Change | Decision |
|---|---|---|---|---|---|---|---|---|
| 1–2. Naive Baseline | blend | 0.5771 | 0.5771 | 0.5771 | 0.3 → 0.3 | 951 → 951 | +0.0438 | kept, **stands** |
| | starting point | 0.5621 | 0.5620 | 0.5620 | – | 13 → 13 | +0.0286 | dropped, stands |
| 3. Select on ranking | v1 | 0.5543 | 0.5434 | 0.5630 | 0.1 → 0.4 | 1 → ≥100 (censored) | −0.0142 | dropped, stands (see below) |
| | search trial 15 (= v2) | 0.5771 | 0.5771 | 0.5771 | 0.3 → 0.3 | 951 → 951 | 0 | dropped, stands |
| | search trial 23 | 0.5793 | 0.5793 | 0.5793 | 0.3 → 0.3 | 1,697 → 1,697 | +0.0021 | dropped, stands |
| | search trial 19 | 0.5764 | 0.5764 | 0.5764 | 0.3 → 0.3 | 1,858 → 1,858 | −0.0007 | dropped, stands |
| | depth 6 | 0.5645 | 0.5550 | 0.5577 | 0.1 → 0.1 | 12 → 15 | −0.0194 | dropped, stands |
| 4. Two-part model | two-part | 0.5627 | 0.5627 | 0.5627 | 0.4 → 0.4 | binary 14 → 14, size 400 → 400 | −0.0144 | dropped, stands |
| 5. Scope | UK Zero Cells sampled | 0.5763 | 0.5763 | 0.5763 | 0.3 → 0.3 | 756 → 756 | −0.0008 | dropped, stands |
| | zero rate 2% | 0.5777 | 0.5777 | 0.5777 | 0.3 → 0.3 | 1,009 → 1,009 | +0.0006 | dropped, stands |
| 6. Product detail | + trading_pair_share | 0.5750 | 0.5750 | 0.5750 | 0.3 → 0.3 | 759 → 759 | −0.0021 | dropped, stands |
| | + importing_markets | 0.5752 | 0.5752 | 0.5752 | 0.3 → 0.3 | 801 → 801 | −0.0020 | dropped, stands |
| | + hs4 | 0.5761 | 0.5761 | 0.5761 | 0.3 → 0.3 | 815 → 815 | −0.0011 | dropped, stands |
| 7. Monotone constraints | constraints on | 0.5771 | 0.5771 | 0.5771 | 0.3 → 0.3 | 951 → 951 | 0 | dropped (= base), stands |
| | constraints off | 0.5708 | 0.5708 | 0.5708 | 0.4 → 0.4 | 1,115 → 1,115 | −0.0064 | dropped, stands |

The Naive Baseline alone on fold 0 falls from 0.5472 to 0.5377, because its zeros tie. It was a reference in the loop, never an Arm, so no decision rests on it.

**Ties decided two Arms.** Both were among the four early stoppers that spec #65's note flagged.

- **v1** stopped at round 1 under the old measure. A single tree gives few distinct predictions, so the tied zeros scored well: 0.5543 as a blend. Under the tie-proof measure, round 1 scores 0.4333 for the model alone and 0.5434 as a blend. The measure then falls and climbs back to 0.4901 at round 100, where it is still rising, but only 101 trees were saved. Re-chosen there (w = 0.4), v1 scores 0.5630, which is 0.0142 below the Blend. Its decision stands with a caveat: where v1 would stop is censored. Only a GPU retrain could show whether more rounds would close a 0.0142 gap. The one fully trained alternative configuration (search trial 23, 1,697 rounds) beat the Blend by only +0.0021, so v1 is unlikely to clear the 0.0089 threshold. It was not retrained (out of scope: no GPU, and "Retuning hyperparameters beyond what the loop's existing Arms already cover" is out of scope of spec #113).
- **depth 6** peaks at round 15 instead of 12 (0.5168 for the model alone, 0.5577 as a blend, against 0.5645 before). After round 15 its measure falls to about 0.47 and is still near there at round 112. Rounds 113–115 cannot plausibly regain 0.05, so it would stop at 115. Strictly, 15 is censored by three rounds.

The starting point (round 13) and the two-part model's binary part (round 14) peaked at the same rounds under both measures. Their early peaks were real, not made by ties.

## The five-fold confirmation

Pooled over every 2022 UK pair (1,156,050 pairs). Rounds are the per-fold rounds of the winner's fold models, re-chosen by replay.

| Five folds, 2022 | Old | Tie-proof |
|---|---|---|
| **Blend (w = 0.3)** | 0.5635 | **0.5635** |
| Model alone (v2) | 0.5206 | 0.5206 |
| Naive Baseline | 0.5332 | **0.5240** |
| Blend − model alone | +0.0429 | +0.0429 (beats the threshold) |
| Blend − Naive Baseline | +0.0302 | **+0.0395** |
| Per-fold rounds | 951, 1,075, 1,097, 1,303, 1,120 | the same (none censored) |
| Served rounds (median) | 1,097 | **1,097** |

The Blend's margin over the Naive Baseline grows under the fixed measure (#115 predicted this), because the Naive Baseline's zeros no longer win from ties. The naive weight re-chosen on the pooled grid is still 0.3 (0.5635, against 0.5629 at 0.4 and 0.5586 at 0.2). On fold 0, which is where the loop chooses it, it is also 0.3 (0.5771, against 0.5769 at 0.4).

## What changes

**No decision flips.** No dropped Arm now wins, the naive weight stays 0.3, and every round count that matters stays the same: the served median is 1,097, and every per-fold round is unchanged. Nothing is re-applied to the served artefact, and nothing needs to be.

**Phase 2's base setting** is the current configuration unchanged: v2's parameters (search trial 15), the 25 features, the monotone constraints on, non-UK rate 0.6157, Zero Cells at 5%, the Blend with w = 0.3, 1,097 rounds served. Its tie-proof measure on fold 0 is **0.57715**. The loop's results hold it as `best` (its old value is kept as `recheck.legacy_best`), and later experiments start from it. The sections of spec #65 stay as they were measured, old measure included, so the old record is still there to read.
