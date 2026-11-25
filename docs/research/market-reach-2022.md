# Market Reach on the 2022 validation year

For spec #113, Phase 1, ticket #124, under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md) and [ADR 0001](../adr/0001-out-of-fold-uk-training.md). The open review of 2022 ([open-review-2022.md](open-review-2022.md), idea 1) found that the ranking does not know where the UK already sells, and proposed a per-Market offset learned from the UK's other Products. This note tests the lagged, out-of-fold version the spec asks for, under the tie-proof Selection Measure (#115) and the recomputed Win Threshold (#116).

**Decision: adopt, with β = 4.** On the pooled five-fold 2022 grid, Market Reach lifts the served Blend from 0.5635 to 0.6041, a gain of **+0.0406** against a Win Threshold of **0.008869**. The gain is +0.040 on every fold alone. It costs dollar accuracy: the 10th–90th spread of the log residual above $10m widens from 3.09 to 3.74.

**Nothing was trained and 2023 was not read.** The offsets come from the 2021 UK grid. The scores are the loop's cached five-fold confirmation of 2022 (`data/derived/loop_V202601/confirmation`). The large-Market measure also reads the 2022 BACI Flows. All computation ran on the CPU, in about a minute.

Script: `scripts/research/market_reach_2022/run.py`. Outputs: `data/derived/market_reach_2022/result.json` and `reach_2022.parquet` (the table per fold and Market).

## What was pre-registered

The form, the grid, the pseudo-count, the lag, the rule for choosing β and the adoption gate were all committed in **ed982b0** (`src/exportspredictor/market_reach.py`, its tests, and the script's docstring), before any 2022 result was read.

- **Form: the open review's share form**, lagged one year. For fold *k* and scored year *t*, Market *m*'s offset is learned from the UK pairs outside fold *k* in year *t* − 1:

  offset = log((sent + 5) ÷ (imported + 10)) − log((Σ sent + 5) ÷ (Σ imported + 10)).

  Here "sent" counts the Market's source pairs with a UK Flow. "Imported" counts the source pairs whose Product the Market imports from any other Exporter (`log_demand_loo` > 0). The sums run over every Market's source pairs, and the pseudo-count is *a* = 5.
  - The first term is the Market's UK share, shrunk towards a half.
  - The second term is the UK's pooled share, so an offset of 0 means "as expected".
  - "Products expected" is the pooled share times "imported".
  - A Market with no source pairs gets an offset of 0.
- **Why the share form, not the rank form.** The review's other form compares the observed share with a "share the ranking expects", which is the summed within-Product rank of Export Potential. Lagged to *t* − 1, that form needs every pair's out-of-fold Export Potential in *t* − 1, so serving it would mean another scoring pass. The share form needs only the previous year's Flows. The review measured the two within 0.002 of each other (+0.046 against +0.044).
- **Export Potential × exp(β · offset) × c**, where c keeps the UK's summed Export Potential over the grid unchanged. The summed Export Potential is $235.39bn at every β.
- **β** comes from {0, 0.5, 1, 2, 4}. It is the β with the highest tie-proof measure on the pooled rows of folds 0–2, each pair taking its own fold's offsets, with ties going to the smaller β.
- **Gate**: adopt only if the chosen β beats the Blend on the pooled five-fold grid by more than `loop.current_win_threshold(loop.load())`.

**Leakage.** A pair is held out with its fold, so none of its own Flows, in any year, enters its offset. Only year *t* − 1 is read, so no UK Flow of 2022 enters either. `tests/test_market_reach.py` checks both: it changes every fold-0 Flow and every Flow of the scored year, and fold 0's offsets stay unchanged. The script also checks that every pair's confirmation fold equals its pair-hash fold.

## Results

Every figure is tie-proof unless marked "old". "Fit" is folds 0–2 pooled, where β is chosen. "Check" is folds 3–4 pooled.

| β | Fit (0–2) | Check (3–4) | **Pooled** | Pooled (old) | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Dollar spread (median) | Top 30 | Top 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (served Blend) | 0.5661 | 0.5676 | 0.5635 | 0.5635 | 0.501 | 0.524 | 0.793 | 0.617 | 3.09 (+0.89) | 0.5048 | 0.5437 |
| 0.5 | 0.5864 | 0.5881 | 0.5838 | 0.5838 | 0.502 | 0.547 | 0.807 | 0.629 | 3.11 (+0.85) | 0.5208 | 0.5626 |
| 1 | 0.5961 | 0.5978 | 0.5935 | 0.5935 | 0.503 | 0.559 | 0.813 | 0.636 | 3.14 (+0.84) | 0.5329 | 0.5756 |
| 2 | 0.6040 | 0.6057 | 0.6015 | 0.6015 | 0.499 | 0.568 | 0.816 | 0.638 | 3.27 (+0.81) | 0.5493 | 0.5920 |
| **4 (chosen)** | **0.6067** | **0.6082** | **0.6041** | 0.6041 | 0.489 | 0.572 | 0.811 | 0.630 | **3.74** (+0.82) | 0.5635 | 0.6059 |

**Gain at β = 4, against the Blend:**

| | Gain |
|---|---|
| Pooled five folds | **+0.0406** (Win Threshold 0.008869) |
| Folds 3–4 pooled | +0.0406 |
| Fold 3 alone | +0.0410 |
| Fold 4 alone | +0.0398 |
| Folds 0, 1, 2 alone | +0.0398, +0.0403, +0.0402 |

- **The gain is information, not ties.** The tie-proof and old measures agree to four decimals at every β. A per-Market multiplier ties nothing.
- **It is mostly the extensive margin.**
  - "Zero against non-zero" rises from 0.524 to 0.572.
  - The ranking among non-zero Flows falls a little at β = 4 (0.501 to 0.489) and is flat at β ≤ 1.
  - This matches the review's findings.
- **The dollar spread is what it trades away.** Above $10m of Actual Exports (4,818 pairs), the 10th–90th spread of log(1 + actual) − log(1 + Export Potential) is unchanged up to β = 1 (3.09 to 3.14), then widens to 3.27 at β = 2 and 3.74 at β = 4. The median residual moves from +0.89 to +0.82. The pre-registered rule picks β = 4 on the Selection Measure, as ADR 0008 makes ranking the one objective. **β = 1 was the cautious alternative**: +0.030 for almost no change in the spread. It was not chosen, because the rule was fixed beforehand.
- **Large Markets (reported, never selected on).** #123 found that on the 30 largest importing Markets the model alone (0.5329) ranks above the Blend (0.5048). With Market Reach at β = 4, the Blend scores 0.5635 on the top 30 (+0.059) and 0.6059 on the top 60 (+0.062), so on those Markets it is also above the model alone without Reach. This note did not measure the model alone with Reach.

## Example Markets

Lagged (2021) table, averaged over the five folds. Each fold sees about 80% of the pairs, so the counts are about four-fifths of a whole Market's. The offset ranges over the folds by less than ±0.03 everywhere.

| Market | Products sent | Products imported (from others) | Expected | Offset |
|---|---|---|---|---|
| Iran | 526 | 3,455 | 1,356 | −0.94 |
| Colombia | 1,314 | 3,759 | 1,476 | −0.11 |
| Bulgaria | 1,565 | 3,962 | 1,555 | +0.01 |
| Mexico | 1,587 | 3,937 | 1,545 | +0.03 |
| Viet Nam | 1,786 | 3,934 | 1,544 | +0.15 |
| Brazil | 2,031 | 3,814 | 1,497 | +0.30 |
| Russian Federation | 2,471 | 3,967 | 1,557 | +0.46 |
| Japan | 2,509 | 4,029 | 1,581 | +0.46 |
| Saudi Arabia | 2,693 | 3,914 | 1,536 | +0.56 |
| India | 2,805 | 3,925 | 1,541 | +0.60 |
| China | 2,973 | 4,026 | 1,581 | +0.63 |
| USA | 3,398 | 4,078 | 1,601 | +0.75 |
| Germany | 3,582 | 4,087 | 1,604 | +0.80 |
| France | 3,761 | 4,079 | 1,601 | +0.85 |
| Ireland | 3,885 | 3,987 | 1,565 | +0.91 |
| Falkland Islands | 839 | 264 | 104 | +2.06 |

Across all fold-Market rows, the offset's 5th, 25th, 50th, 75th and 95th percentiles are −3.50, −1.33, −0.32, +0.29 and +0.71. The many small Markets with few UK Products sit in the long negative tail.

**Reading the table.**
- **The centring constant.** The review printed the share term alone: Mexico −0.9, Iran −1.9. Here the pooled share (about 39%) is subtracted, so Mexico reads +0.03 and Iran −0.94. Their difference, which is all the ranking uses, is the same.
- **Many-Product Markets sit above 0.** The pooled share counts each source pair once. The UK sends more than 39% of the Products that Markets importing many Products take, and much less to small Markets.
- **Mexico is about average among Markets.** It sits far below the large Markets the UK serves broadly: 1,587 of 3,937 Products, against 3,761 of 4,079 for France. The spec's example sentence ("2,017 of the Products Mexico imports, against about 3,040 expected") came from the review's rank form on 2022 and is not this figure. #125's display should use this table's "expected".
- **Russia is +0.46**, because the offset is lagged: it is learned from 2021, before the 2022 sanctions. A served offset for 2024 from 2023 would reflect the sanctions. Russia is a Sanctioned Market (#119) and carries that caveat whatever its offset.

## What it changes

Market Reach associates broad UK reach into a Market with higher Export Potential there, product by product. It does not say that reach causes exports.

- **An Export Gap now means a Product-specific shortfall within a Market.** A Market-wide shortfall (the UK sends few Products to Iran) no longer shows as many pair-level gaps, so #125 shows each Market's reach beside it.
- **Scenarios.** The multiplier is the same for every Product of a Market, and c is a fixed constant, so a Scenario's relative effect is unchanged (tested). Dollar Scenario Effects scale with the Market's multiplier.
- **Not a feature.** It is post-processing in the Model Form, so `features.py`'s "no pair trade in other Products" rule for features still holds. The leave-one-out guardrail (now in CODING_STANDARDS.md, "Modelling guardrails") needs rewording for this exception, and #125 writes the ADR.
- **Not measured here.** Whether the Potential Range's coverage on UK-like pairs suffers (calibration rows are other Exporters' and get no offset) is left to #125, which rebuilds the bands through the served path.

## For the tickets that use it

- **#125 (serve).**
  - Call `market_reach.reach_table(uk_rows, baseline_year)` on the UK grid of the year before the Baseline year, with each pair's fold.
  - Store the table and β = `market_reach.CHOSEN_BETA` in the Model Form, with `PSEUDO_COUNT`, `LAG_YEARS` and `FORM`.
  - Compute c once with `market_reach.rescale_constant` on the Baseline grid, store it, and apply it with `market_reach.apply` inside `prediction.export_potential`. Never recompute c per Scenario, or the Scenario's level would be renormalised away.
  - Add Market Reach's settings to `ledger.FINGERPRINT_KEYS`.
- **#127 (Phase 2 base).** Apply the same `reach_table` (2021 grid, scored year 2022) and `apply` with `CHOSEN_BETA` to the fold-0 Blend of each Arm. The cached base's fold-0 measure with Reach at β = 4 is 0.6169 (0.5771 without).

## Re-check: sent nested in imported (`nested_share`)

The first form counted every UK Flow as sent, so a Market could have more Products "sent" than "imported". Only the Falkland Islands did (839 against 264 above, offset +2.06), and its served multiplier of about 230× made it the fourth Market by Blocked Opportunity value. The fix counts as sent only the pairs that imported also counts (ADR 0010, *Amendment*). The same script was run again unchanged apart from the form; it now writes to `<data>/derived/market_reach_2022/<FORM>/`, and #124's result stays in the folder itself.

| Form | Chosen β | Pooled measure | Gain over the Blend (0.5635) | Folds 3–4 gain | Falkland Islands offset |
|---|---|---|---|---|---|
| `share` (#124) | 4 | 0.6041 | +0.0406 | +0.0406 | +2.06 |
| `nested_share` | 4 | 0.6084 | +0.0449 | +0.0449 | +0.35 |

The Win Threshold is 0.008869, so the nested form is adopted under the same gate. Every fold gains (+0.044 to +0.045, against +0.040 to +0.041). The other example Markets move by about 0.01 (Ireland +0.907, the United States +0.761, Iran −0.939). The offset's percentiles are −3.63, −1.39, −0.34, +0.30 and +0.71. The 30 largest Markets' measure is unchanged (0.5635 against 0.5635). The test year was not read.
