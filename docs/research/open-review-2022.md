# An open review of the 2022 held-out ranking: where Export Potential goes wrong, and what might fix it

An open-ended review of the served Model Form (the Blend, naive weight 0.3) on the 2022 validation year, under [ADR 0008](../adr/0008-opportunity-ranking-is-the-one-objective.md). It slices the ranking errors, looks at the biggest misses one by one, reads the served artefact as an Advisor would, and ends with improvement ideas ranked by their evidence. It does not repeat what the [experiment loop](opportunity-ranking-experiments.md) already tested.

**Rules kept.** Every figure is on 2022. Nothing was computed against 2023 Actual Exports, and no 2023 row was opened. No model was trained and the GPU was not used. One test re-predicts the 2022 UK grid with the five saved confirmation fold models on the CPU, each pair still scored by the fold model that held it out ([ADR 0001](../adr/0001-out-of-fold-uk-training.md)). Every adjustment that learns from Actual Exports is **cross-fitted by fold**: a pair's adjustment comes only from UK pairs in other folds, so no pair ever sees its own Actual Exports. Tuning is done on folds 0–2 and checked on folds 3–4.

Every result is an association measured on held-out UK pairs in one year. "Gains X" means it ranked the UK's 2022 Markets closer to its 2022 Actual Exports. It does not say why, and it is not a forecast.

## Summary

- **The biggest missing signal is the UK's reach into each Market.** The ranking does not know how many of a Market's imported Products the UK already sends there. A simple adjustment learned from the UK's *other* Products in the same Market, cross-fitted by fold, gains **+0.046** on the check folds (five times the Win Threshold of 0.00887). It gains +0.041 when it is built only from 2021 Flows, and +0.042 even after the tie effect below is controlled for. It moves the meaning of the Export Gap, so it needs an ADR before it is served.
- **The Selection Measure rewards ties at the bottom of the ranking.** Flooring the served Export Potential at $1,000, which ties 52% of pairs and adds no information, raises the measure from 0.5635 to 0.6062 (+0.043). That is as much as the Blend's whole win over v2. The Blend's win and the reach gain both survive this control, but future Arms should be checked against it.
- **The measure is mostly a measure of how widely the UK exports a Product.** Across Products, the Spearman correlation is 0.96-correlated with the share of Markets the UK exports the Product to. For Products the UK sends to fewer than 10% of Markets, the Blend ranks worse than the Naive Baseline.
- **Advisors are shown Opportunities in Markets that are closed.** The served artefact lists 9,915 Opportunities in Russia and Belarus ($1.2bn of Export Gap, 138 in the top 10,000), and whisky to Saudi Arabia is Opportunity #552. One in three of the top 100 Blocked Opportunities shows an "Export Potential with a Trade Agreement" below what the UK already exports.
- **Tested and found nothing:** the leave-one-out penalty on UK supply, a naive weight that depends on the Product's breadth, aligning the five fold models' levels, and UK-free demand proxies (all below the threshold).

## The starting point, reproduced

The five confirmation files (`data/derived/loop_V202601/confirmation/fold_<k>.parquet`) hold every UK Product–Market pair of 2022: 1,156,050 pairs, 5,084 ranked Products. Recomputing the Blend from their `v2` and `naive_usd` columns matches their `winner` column to 1e-7.

| 2022, held-out UK pairs | Pooled (the measure) | Mean of the five folds | Fit folds 0–2 | Check folds 3–4 |
|---|---|---|---|---|
| Served Blend (w = 0.3) | **0.5635** | 0.5751 | 0.5661 | 0.5676 |
| v2, the model alone | 0.5206 | 0.5320 | 0.5228 | 0.5254 |
| Naive Baseline | 0.5332 | 0.5436 | 0.5368 | 0.5358 |

- **The pooled measure sits 0.0116 below the fold mean, and the Naive Baseline shows almost the same gap (0.0104),** though it has no fold models. So the gap comes from ranking each Product over all its Markets at once rather than over a fifth of them, not from the five fold models' levels differing. Shifting each fold model so its mean log(model ÷ naive) matches the others changes the pooled measure by +0.00005.
- **Dollar accuracy, as asked** (pairs with Actual Exports above $10m, 4,818 pairs): log(1 + Actual) − log(1 + Export Potential) runs from −0.29 (10th percentile) through +0.89 (median) to +2.80 (90th), a spread of **3.09**. v2 alone: 3.07; Naive Baseline: 3.88. This selects pairs by their outcome, so a positive median is expected of any honest estimate. Selected by the prediction instead (Export Potential above $10m, 3,501 pairs), the median is −0.14.

## The failure review

### 1. The measure is mostly breadth, and the Blend loses to the naive rule on narrow Products

Each Product's correlation, grouped by the share of the 225 Markets the UK exported it to in 2022:

| UK exports the Product to | Products | Blend | Naive Baseline | v2 alone | UK exports, $bn |
|---|---|---|---|---|---|
| under 2% of Markets | 206 | 0.081 | **0.105** | 0.064 | 0.03 |
| 2–5% | 324 | 0.194 | **0.244** | 0.141 | 0.6 |
| 5–10% | 500 | 0.319 | **0.333** | 0.257 | 3.1 |
| 10–20% | 969 | **0.464** | 0.446 | 0.407 | 16.4 |
| 20–40% | 1,656 | **0.631** | 0.585 | 0.585 | 66.6 |
| 40–60% | 952 | **0.763** | 0.700 | 0.732 | 72.6 |
| 60% or more | 477 | **0.847** | 0.790 | 0.839 | 176.7 |

- Across Products, the correlation and the breadth share are themselves correlated at **0.96** (Spearman). Across Sectors, 0.96. The weakest Sectors (26 ores, 03 fish, 47 pulp, 01 live animals, 02 meat) are simply the ones the UK sends to few Markets.
- **For narrow Products most Markets are tied at zero**, and the correlation is decided by which Markets the UK reaches at all. That is where the model is weakest: it beats the Naive Baseline only once the UK reaches at least 10% of Markets. Choosing the naive weight per breadth band on folds 0–2 (0.6 for the narrowest band down to 0.3) gained only +0.0015 on folds 3–4, inside the noise.

### 2. Markets: the ranking does not know where the UK already sells

Within each breadth band and each within-Product rank of Export Potential, the share of pairs with a UK Flow gives the number of Products the UK "should" send to each Market if the ranking were right on average. Observed against expected:

- **The UK reaches more Products than expected** in small Commonwealth and former-colony Markets and close neighbours: Montserrat 923 against 247, Seychelles 1,986 against 799, Grenada, Dominica, Bermuda, Barbados, Malta (3,445 against 1,838), Gambia, Malawi, Iceland, the Falklands, Cyprus, and Ireland (4,824 against 3,627).
- **Fewer than expected** in microstates the UK never ships to (Tokelau, Pitcairn, Christmas Island, Palau: almost nothing against 200–400 expected), in sanctioned Markets (North Korea 4 against 239, Syria 52 against 657, Iran 613 against 1,873, Russia 1,738 against 3,055), and in large Markets where the UK is weak: Mexico (2,017 against 3,037), Uzbekistan, the DR Congo, Bulgaria, Colombia, Viet Nam.
- **In dollars**, Actual Exports are 4.1 times Export Potential in Ireland, 2.0 in the Netherlands, 2.2 in the UAE and 15 in the Falklands, and 0.4 in Russia and Croatia.

The fold models train on other UK pairs in the same Market, but no feature names the Importer, so this UK-specific pattern can be learned only through the UK's own gravity values, which deep trees on 84 M mostly non-UK rows barely do. The Market-level pattern is learnable from the UK's other Products, which is what idea 1 uses.

**Twenty Markets have no gravity data at all** (GDP or distance missing): Gibraltar, the Falklands, Montserrat, Saint Helena, Anguilla, the British Virgin Islands, the British Indian Ocean Territory, Pitcairn, Bonaire, North Korea, and ten small Pacific and French territories. Half are UK Overseas Territories, where the UK's trade has nothing to do with gravity.

### 3. Where the UK dominates a Market's imports, the Market looks small

Leave-one-out demand (the Market's imports from everyone but the UK) is the main driver of both the model and the Naive Baseline. Where the UK supplies most of a Market's imports of a Product, what is left looks small. Among pairs with a UK Flow:

| UK share of the Market's imports | Pairs | UK exports, $bn | Mean rank error (Export Potential rank − Actual rank) | Median log(Actual ÷ Export Potential) |
|---|---|---|---|---|
| under 5% | 264,255 | 121.6 | −0.02 | −1.16 |
| 5–20% | 40,523 | 120.2 | −0.17 | +1.04 |
| 20–50% | 14,126 | 59.2 | −0.25 | +2.02 |
| 50–80% | 5,088 | 23.7 | −0.35 | +2.81 |
| 80% or more | 6,377 | 11.3 | −0.45 | +2.98 |

Part of this is selection on the outcome, but the mechanism is real. UK milk and cream (040120) to Ireland, 96% of the UK's exports of it, gets a naive value of $12k against $353m of Actual Exports, because Ireland buys almost none from anyone else. Two UK-free demand proxies were tested as an extra term in log Export Potential: the Market's imports of the Product's HS4 siblings from other Exporters (+0.0067 on folds 3–4) and the Market's GDP (+0.0056). Both are below the threshold, and neither adds anything on top of idea 1 (+0.0005 each).

### 4. Closed Markets: sanctions and alcohol prohibition

- **Russia.** UK exports to Russia fell from $4.17bn in 2021 to $1.23bn in 2022, and the UK sent 1,738 Products against 3,089. Export Potential for 2022 is $3.12bn, and Russia sits among the top 10 Markets by Export Potential in 354 Products. Leaving Russia and Belarus out of the 2022 measure raises it by 0.003 (this changes the measure, not the model).
- **In the served artefact** (`20261120T133841Z`, Baseline year) Russia and Belarus hold 9,915 Opportunities with $1.2bn of Export Gap: 10 in the top 1,000 and 138 in the top 10,000. They include whisky ($46m gap), cosmetics, diagnostic reagents, cars, and aircraft parts to Russia, and cars to Belarus as a New Market.
- **Alcohol.** Whisky to Saudi Arabia is Opportunity #552 ($17.4m gap) and whisky to Kuwait #1,307 ($9.3m, New Market). In 2022 whisky to Saudi Arabia ranked 46th of 225 Markets by Export Potential with no Flow. Saudi Arabia, Kuwait, Libya and Iran prohibit alcohol imports. Nothing in the features can tell the model so.

### 5. Agri-food and steel: confident leads with no Flow

Among the ten Markets with the highest Export Potential for Products the UK sends to at least 20 Markets, the share where the UK sends nothing:

| Group | Top-10 pairs | No UK Flow | In ten large non-EU Markets (USA, China, Japan, Korea, India, Australia, Canada, Brazil, Mexico, Saudi Arabia) |
|---|---|---|---|
| Agri-food (HS 01–24) | 5,350 | **16.7%** | **26.5%** of 1,141 |
| Iron and steel (HS 72–73) | 2,470 | 8.7% | 10.8% of 664 |
| Everything else | 33,900 | 7.3% | 7.6% of 10,011 |

The leads at the top of this list are typical market-access cases: scallops and lobsters to the USA, sheep meat to China, pork to Japan and Italy, rapeseed oil to China, semi-finished steel to Türkiye, Italy and Indonesia, slag to the USA. Agri-food Products also rank worst overall (mean correlation 0.41 against 0.60 for the rest). This pattern fits sanitary and phytosanitary barriers and trade remedies, which no feature describes. That reading is a hypothesis: the data here cannot separate a closed Market from a Market the UK has simply not tried.

### 6. Lumpy and one-off Products

Products whose UK exports swung most from year to year over 2012–2021 (the standard deviation of the yearly log change, training years only) rank far worse in 2022:

| Volatility quintile | Products | Mean correlation | Mean breadth |
|---|---|---|---|
| calmest | 1,017 | 0.760 | 0.49 |
| 2 | 1,017 | 0.677 | 0.38 |
| 3 | 1,016 | 0.588 | 0.28 |
| 4 | 1,017 | 0.493 | 0.20 |
| lumpiest | 1,017 | 0.300 | 0.09 |

Volatile Products are also narrow, but the association holds within each breadth decile (correlation −0.27 on average).

### 7. The biggest misses, case by case

Of the UK's 2022 Flows above $20m, many are ranked outside their Product's top 20 Markets. Ireland has 25 of them, Saudi Arabia 16, Israel and Norway 13 each. The largest:

| Product | Market | Actual | Rank by Export Potential | What it looks like |
|---|---|---|---|---|
| 300490 medicaments | Ireland | $1,457m | 25th (3rd by Actual) | UK–Ireland distribution and supply chains: the Market-reach gap of section 2 |
| 880230 aircraft, 2–15 t | Qatar | $887m (86% of UK exports of it) | 77th | one-off aircraft deliveries |
| 841112 turbojets | Saudi Arabia, Italy, Luxembourg, Netherlands, Ethiopia | $255–700m each | 23rd–55th | engine sales follow airline fleets and leasing hubs |
| 284390 precious-metal compounds | North Macedonia | $511m (33%) | 94th | probably one producer's catalyst plant: Flows of one firm, not of a Market |
| 040120 milk and cream | Ireland | $353m (96%) | 31st | the UK dominates Ireland's imports, so its demand looks small (section 3) |
| 854239 integrated circuits | Thailand | $290m (35%) | 30th | supply-chain trade |
| 490700 stamps, banknotes | Angola | $186m (51%) | 45th | printed banknotes: a sovereign contract |
| 120190 soya beans | Iran | $185m (99.7%) | 48th | the UK grows little soya: likely a transit or humanitarian Flow |
| 890520 drilling platforms | Netherlands | $122m (95%) | 33rd | one platform |

The confident leads with no Flow (Export Potential rank 1–3, in Products the UK sends to at least 20 Markets): 670 pairs, mostly in the USA (98), Italy (66), China (64), India (32) and Spain (29), and dominated by the agri-food and steel cases of section 5, plus 284390 to Hong Kong and aircraft to China.

### 8. Dollar level and the Blocked Opportunities

- **Export Potential sums to 70% of Actual Exports**: $235bn against $336bn in 2022 (the model alone $248bn, the Naive Baseline $280bn). Grouped by decile of Export Potential, Actual Exports sum to between 1.0 and 4.7 times Export Potential, and 1.4 times in the top decile. A geometric blend of two skewed estimates is a median-type figure, so totals built from it understate. Summed Export Gaps (for example the Leaderboard summary of a filter) are therefore not on the same scale as summed Actual Exports.
- **Blocked Opportunities.** In the served artefact, the "Export Potential with a Trade Agreement" (Baseline Export Potential plus the Blocked Opportunity value) is below the UK's Actual Exports in 33 of the top 100 Blocked Opportunities, 108 of the top 500 and 379 of the top 2,000. The second largest, passenger cars (870324) to the USA, reads $1.47bn of Export Potential, +$636m with an agreement, against $3.36bn already exported. An Advisor reading "with an agreement the UK could export $2.1bn" would be misled, though the figure is labelled Illustrative.

## Tested and found nothing

| Test | Result on 2022 | Reading |
|---|---|---|
| **The leave-one-out penalty on UK supply.** Within a Product, leave-one-out supply is total UK supply minus the pair's own Flow, so it falls exactly as Actual Exports rise: their within-Product Spearman is −0.997. In 1,247 Products one Market takes over half of UK exports. The test makes supply constant within a Product inside each fold (UK exports to Markets outside the row's fold), for the naive side and, re-predicting on the CPU, for the model. | Naive Baseline 0.5307 against 0.5332; Blend 0.5619 (naive side) and 0.5629 (model side) against 0.5635 | No gain: the penalty is real but too small to matter. **A warning follows from it:** any UK supply feature normalised within a Product (for example divided by the Product's largest value) would hand the model the target almost exactly. |
| A naive weight per Product breadth band, from 2021 breadth | +0.0015 on folds 3–4 | inside the noise |
| Aligning the five fold models' levels (label-free) | +0.00005 pooled | the fold models agree in level |
| Market offsets from within-Product rank errors | +0.0005 (Market), +0.0037 (Market × Sector) | far weaker than the reach offset of idea 1 |
| A dollar Market offset (log of summed Actual ÷ summed Export Potential in other folds) | +0.0109 on folds 3–4, +0.0113 pooled, spread 3.05 | Alone it is just over the threshold. On top of idea 1, folds 0–2 choose weight 0 for it, so it adds nothing. |
| UK-free demand proxies (HS4 sibling demand, GDP) | +0.0067 and +0.0056 alone; +0.0005 on top of idea 1 | below the threshold |

## Ideas, ranked

Gains are on folds 3–4, with the adjustment's weight chosen on folds 0–2 by leave-one-fold-out, and the cross-fitted five-fold pooled figure beside them. Every gain is against the served Blend's 0.5676 on folds 3–4 (0.5635 pooled). "Dollar" is the 10th–90th spread on pairs with Actual Exports above $10m (3.09 for the served Blend).

| # | Idea | Kind | Gain (folds 3–4 / pooled) vs 0.00887 | Dollar spread | Cost | Guardrails and ADRs |
|---|---|---|---|---|---|---|
| 1 | **UK Market reach offset**, cross-fitted | post-processing (or a feature) | **+0.046 / +0.046** (β 4); **+0.031 / +0.031** (β 1); lagged 2021 version +0.041 | 3.40 (β 4); 3.09, unchanged (β 1) | CPU, seconds; one small table per fold in the artefact; no retraining | Amends the leave-one-out guardrail's wording; keeps ADR 0001's pair hold-out; changes what an Export Gap means (needs a new ADR beside 0008) |
| 2 | **Make the Selection Measure tie-proof** | measure governance | 0 (stops spurious wins worth up to +0.05) | – | none | ADR 0008's measure; the loop's per-round stopping |
| 3 | **Closed-Market flags**: sanctions, alcohol prohibition | Advisor protection | not a model gain (leaving Russia and Belarus out moves the 2022 measure by +0.003) | – | a short hand-kept table, a filter and a caveat | extends ADR 0005's Market flags |
| 4 | **Honest Blocked Opportunity and dollar figures** | Advisor protection | 0 | 0 | app and API text | GLOSSARY wording; ADR 0008's "Illustrative" |
| 5 | **Flag volatile and one-off Products** | Advisor protection | 0 | – | a Product table from training years | extends ADR 0005's Flagged tier |
| 6 | Non-tariff market access (SPS, trade remedies) as a policy feature | feature, **hypothesis** | not testable here; **skipped** by #171 (new data; further work) | unknown | new data; a full fold set retrained on the GPU (about 1 hour) | a new policy feature: monotone constraint, Scenarios (ADR 0006, 0008) |
| 7 | Report the measure on the Markets Advisors use | measurement | 0 | – | none | ADR 0008 (reported, never selected on) |
| 8 | Gravity for the 20 Markets without it; group the UK Overseas Territories | data, partly **hypothesis** | GDP and population filled by #126 and kept by #127 (−0.0001); distance needs new data: **skipped** by #171 | – | one lookup table; features rebuilt (CPU) and fold models retrained | none |
| 9 | UK-free demand proxy | feature, **hypothesis** | +0.0067 as a blend term (below); as a feature (#164), +0.0003 on fold 0 with Market Reach: **dropped** | worse as a blend term | features rebuilt and fold models retrained | none |

### 1. A UK Market reach offset

*Tested by #124 ([market-reach-2022.md](market-reach-2022.md)): the lagged share form under the tie-proof measure, +0.0406 pooled at β 4; adopted.*

**What it is.** For each Market, the share of Products the UK sends there, observed against what the ranking expects, learned from UK pairs in the *other* folds only:

offset(Market) = log((UK Flows + 5) ÷ (pairs + 10)) − log((summed within-Product rank of Export Potential + 5) ÷ (pairs + 10)),

and Export Potential × exp(β · offset). The second term is label-free; only the first uses Actual Exports, of other Products in the same Market. A simpler form gives almost the same: the log of the UK's share of the Products the Market imports from anyone else (+0.044). So does a lagged form built only from 2021 Flows (+0.041), so the gain is not a 2022 sanctions effect.

**Evidence.**

| β | Folds 3–4 | Pooled | Non-zero | Zero vs non-zero | Value-weighted | Top-10 share | Dollar spread (median) |
|---|---|---|---|---|---|---|---|
| 0 (served) | 0.5676 | 0.5635 | 0.501 | 0.524 | 0.793 | 0.617 | 3.09 (+0.89) |
| 1 | 0.5987 | 0.5943 | 0.499 | 0.561 | 0.815 | 0.636 | 3.09 (+1.06) |
| 2 | 0.6084 | 0.6041 | 0.492 | 0.573 | 0.818 | 0.638 | 3.15 (+1.23) |
| 4 (chosen on folds 0–2) | **0.6137** | **0.6093** | 0.476 | 0.581 | 0.813 | 0.638 | 3.40 (+1.54) |

- **It holds on each check fold alone** (fold 3 +0.047, fold 4 +0.045 at β 4), and it is larger on the Markets that matter most: +0.060 on the 30 largest importers and +0.063 on the 60 largest.
- **It is information, not ties.** With each form floored at its own best floor on folds 0–2 (idea 2), β 4 still gains +0.042 and β 1 +0.024.
- **It is mostly the extensive margin.** It raises "zero against non-zero" from 0.524 to 0.581, and lowers the ranking among non-zero Flows a little (0.501 to 0.476 at β 4, 0.499 at β 1). Offsets by Market × Sector do no better (+0.047), and a dollar offset on top adds nothing.
- Example offsets (2022, the share form): Iran −1.9, Russia −1.0, Colombia −1.0, Bulgaria −1.0, Mexico −0.9, Viet Nam −0.7; the USA −0.17, Germany −0.13, Ireland −0.04; the Falklands +1.0.

**Cost.** A group-by per fold on the scored grid. Nothing is retrained. In serving, fold *k*'s pairs take the offset learned from the UK pairs outside fold *k* in the Baseline year, or in the year before.

**Guardrails.**
- **ADR 0001 holds**: a pair never sees its own Actual Exports, because the offset is cross-fitted on the same pair folds. But CLAUDE.md's wording, "any feature built from UK data excludes the target pair's own Market", is broken by design: the offset is built from UK Flows into the target Market in other Products. That needs a new ADR, stating that the hold-out unit is the pair.
- **It changes what an Export Gap means.** Fundamentals would then include the UK's demonstrated reach into the Market. A market-wide shortfall (the UK is weak in all of Mexico) would no longer show as many Opportunities in Mexico, and the remaining gaps would be Product-specific under-performance within a Market. This is the trade-off ADR 0001 guards against, at Market rather than pair level. ITC's Export Potential Indicator has, as far as this review recalls its methodology (Decreux and Spies, 2016; not re-read here), a comparable bilateral "ease of trade" term built from the exporter's actual trade with the Market. To keep the market-wide shortfall visible, the app could show each Market's reach beside the Leaderboard ("the UK sends 2,017 of the Products Mexico imports, against about 3,040 expected").
- **Scenarios are not affected in proportion.** The offset multiplies Export Potential by a constant per Market, so a Scenario's relative effect is unchanged. Dollar Scenario Effects and Blocked Opportunity values scale with the Market's offset. As post-processing it also avoids features.py's "no pair trade in other Products" rule, which exists because such a feature *inside* the model would absorb the policy effect. As a model feature it would hit that rule. That is a reason to keep it outside the model.
- **β 1 is the cautious choice**: +0.031, no change to the dollar spread, and the non-zero ranking almost unchanged. β 4 is what the Selection Measure picks. The offsets are mostly negative, so the dollar median moves up. A constant rescale would restore it without changing the ranking.

### 2. Make the Selection Measure tie-proof

**Evidence.** Flooring the served Blend at a dollar value ties the pairs below it. Nothing is learned, yet:

| Floor | Pairs tied at the floor | Pooled | Folds 3–4 |
|---|---|---|---|
| $1 (served) | 0.03% | 0.5635 | 0.5676 |
| $100 | 13% | 0.5643 | 0.5686 |
| $1,000 | 52% | **0.6062** | 0.6133 |
| $3,000 | 64% | **0.6103** | 0.6180 |
| $10,000 | 75% | 0.5875 | 0.6002 |

The Spearman correlation is a correlation of ranks. Most pairs at the bottom of a Product's ranking are tied zeros in Actual Exports, so ordering them adds variance without covariance, and tying them removes it. On the same check, **v2 alone floored at $11k scores 0.5717, above the unfloored Blend (0.5676).** So an Arm that compresses the bottom of the ranking, by design or by accident, can win on the measure without ranking anything better. Each form compared at its own best floor (chosen on folds 0–2): the Blend still beats v2 by +0.046 and the Naive Baseline by +0.075. The loop's choice therefore stands.

**Proposal.** Leave the measure unchanged and add a pre-registered control: compare every Arm with the best so far at each one's own best floor (chosen on the fit folds), and report it beside the measure. Alternatively, fix one common floor in the Model Form. Do not serve a floor as an improvement: it changes no ranking an Advisor uses. A hypothesis worth checking: the loop chooses rounds by the measure every round, so its early stopping may prefer trees that flatten the bottom of the ranking. *(Answered, so not run by #171: #115 made the measure tie-proof, so a flatter bottom can no longer raise the measure the loop stops on, and #116 replayed every stopping decision under it, [loop-recheck-tie-proof-2022.md](loop-recheck-tie-proof-2022.md).)*

### 3. Closed-Market flags (Advisor protection)

Add a Market flag, like the Re-export Hub flag, for Markets under broad UK export restrictions (Russia and Belarus since 2022), and for those under the UK's other wide sanctions regimes (North Korea, Syria, Iran). Add a Product–Market flag for alcohol (HS 2203–2208) in Markets that prohibit it (Saudi Arabia, Kuwait, Libya, Iran). Show both with a caveat and a filter, hidden by default. The evidence is section 4: 9,915 served Opportunities in Russia and Belarus, and whisky to Saudi Arabia at #552. The model need not change. Idea 1 already pushes Russia down (−1.0) in a year after the ban, but no model can know a prohibition. Cost: a hand-kept table with dates and sources, to be checked against current UK sanctions guidance. It extends ADR 0005.

### 4. Honest Blocked Opportunity and dollar figures (Advisor protection)

- Where Actual Exports exceed Export Potential, show the Blocked Opportunity value as a percentage uplift, or apply it to the larger of the two, rather than a "with an agreement" dollar figure below what the UK already sells (33 of the top 100).
- State that Export Potential is a median-type figure that sums to about 70% of Actual Exports (2022). Leaderboard totals should be read as relative, not as dollars the UK could add.

Neither changes the measure.

### 5. Flag volatile and one-off Products (Advisor protection)

Extend ADR 0005's Flagged Product tier with a volatility rule computed on training years (for example, the top quintile of the year-to-year volatility of UK exports). Name the one-off Products the case review found: 4907 (stamps and banknotes, sovereign contracts), 8802 (aircraft), 8901 and 8905 (vessels and platforms), and 284390 (precious-metal compounds, probably one producer's Flows, close to the Sector 71 Products already flagged). Correlation falls from 0.76 to 0.30 across volatility quintiles. Gaps on these Products are mostly timing.

### 6. Non-tariff market access (hypothesis)

*Skipped by #171 ([final-round-2022.md](final-round-2022.md#8-the-leftover-research-leads-171)): it needs new data (export health certificate lists or I-TIP notifications), so it is further work.*

The model knows tariffs and agreements, not whether a UK Product may enter a Market at all. The agri-food and steel false leads of section 5 (26.5% with no Flow among the top-10 Markets in large non-EU Markets, against 7.6% elsewhere) fit sanitary and phytosanitary barriers and trade remedies. A candidate source is the UK's own export health certificate lists (which agri-food Products have an agreed certificate for each Market), or WTO I-TIP's notifications. A binary "access agreed" feature would be a policy feature (monotone, Scenario-able) and needs a retrained fold set. Untested here.

### 7. Report the measure on the Markets Advisors use

The measure weighs each of the 225 Markets equally, and about 100 of them are microstates whose zeros dominate narrow Products. On the 30 largest importers the Blend scores 0.507, and on the 60 largest 0.544 (0.5635 on all). Report it beside the headline, never select on it, so the write-up shows how well the ranking does where Advisors look.

*Added by #123, which reports it.* `scripts/wt.ps1 python scripts/research/open_review_2022/s22_large_markets.py` reproduces both figures with the package's function (`large_markets.restricted_measures`) and gives the tie-proof measure (#115) beside them. Ties do not move the Blend or the model alone; they lower only the Naive Baseline. The review ranked Markets by their imports of the UK grid's Products from Exporters other than the UK (`s05_anyflow`); the evaluation ranks them by total imports from every Exporter, which swaps Saudi Arabia out of the top 30 for Sweden.

| 2022, held-out UK pairs | Top 30, old | Top 30, tie-proof | Top 60, old | Top 60, tie-proof | Every Market, tie-proof |
|---|---|---|---|---|---|
| Blend, Markets as the review ranked them | **0.5065** | 0.5065 | **0.5437** | 0.5437 | 0.5635 |
| Blend, Markets as the evaluation ranks them | 0.5048 | 0.5048 | 0.5437 | 0.5437 | 0.5635 |
| v2, the model alone (evaluation's ranking) | 0.5329 | 0.5329 | 0.5621 | 0.5621 | 0.5206 |
| Naive Baseline (evaluation's ranking) | 0.3577 | 0.3573 | 0.4030 | 0.4025 | 0.5240 |

On the largest Markets the model alone ranks better than the Blend, and the Naive Baseline much worse: the Naive Baseline's weight was chosen on every Market. That is an association on one validation year, reported, not chosen on.

### 8. Gravity for the 20 Markets without it; group the UK Overseas Territories

*Partly done, the rest skipped by #171 ([final-round-2022.md](final-round-2022.md#8-the-leftover-research-leads-171)): #126 filled GDP and population for these Markets and #127 kept the fill as a data correction (its own effect −0.0001, [exporter-count-2022.md](exporter-count-2022.md)). Distance for them is not in the gravity tables we hold, so it needs new data. Grouping the UK Overseas Territories is an Advisor-facing filter, not a model change.*

Twenty Markets have no GDP or distance (section 2), so every split on those features sends them down the default branch. Gibraltar and the Falklands receive $5–15 for every $1 of Export Potential. Filling GDP and distance from another source is cheap but needs features rebuilt and models retrained, for small Markets: a hypothesis with a small expected gain. A cheaper Advisor-facing step is to group the UK Overseas Territories (Gibraltar, the Falklands, Montserrat, Saint Helena, Anguilla, the British Virgin Islands, the British Indian Ocean Territory, Pitcairn, Bermuda, the Cayman Islands, Turks and Caicos) so they can be filtered like the Re-export Hubs.

### 9. A UK-free demand proxy (hypothesis)

*Tested by #164 ([final-round-2022.md](final-round-2022.md#3-sibling-demand-a-uk-free-demand-proxy-as-a-feature-164)): HS4 sibling demand as a model feature ("Sibling Demand"), +0.0003 against the base retrained on the same files, with Market Reach; dropped.*

Section 3's UK-dominated pairs look small because leave-one-out demand removes the UK. As blend terms, HS4 sibling demand and GDP gain +0.0067 and +0.0056, below the threshold and nothing on top of idea 1. As model features, interacting with the Product, they might do better. Untested.

## What not to do

- **Do not add a lagged UK pair Flow, or any within-Product normalised UK supply.** Either would raise the measure a lot by copying Actual Exports. Leave-one-out supply already carries a within-Product Spearman of −0.997 with Actual Exports; only its tiny within-Product range keeps it harmless.
- **Do not serve a floor** (idea 2): it wins the measure and ranks nothing better.

## Reproducing

Scripts are in `scripts/research/open_review_2022/`, run in order by `run_all.ps1` (CPU only, about 30 minutes, from a worktree: `powershell -File scripts/research/open_review_2022/run_all.ps1`). They read `data/derived/loop_V202601/confirmation/fold_<k>.parquet`, the 2012–2022 UK grids and training files in `data/derived/features_V202601/` (2012–2021 only to refit the clip ranges and to measure volatility), the 2022 BACI Flows, and the newest artefact's tables (qualitative review only). They write only to their own `out/` folder, which git ignores.

| Script | What it gives |
|---|---|
| `common.py` | the 2022 frame (scores, fold, features, names), the measure, the dollar spread, the fit/check harness and cross-fitting |
| `cpu_models.py` | CPU re-prediction with the five saved fold models; reproduces `v2` to 7e-5 in 12 s |
| `s01_reproduce` | 0.5635 and the starting table |
| `s02_slices`, `s03_markets` | sections 1 and 2 |
| `s04_market_affinity`, `s05_anyflow`, `s09_affinity_variants`, `s14_combined`, `s18_affinity_robustness`, `s19_reach_plus_dollar` | idea 1 and the Market-offset variants |
| `s06_loo_supply`, `s07_model_supply` | the leave-one-out supply test |
| `s10_more` | diagnostics, the breadth weight, fold alignment, the naive-zero count, calibration |
| `s11_cases`, `s17_products` | section 7 |
| `s12_structure`, `s13_demand` | sections 3, 5 and 6, and the demand proxies |
| `s08_artefact`, `s15_access`, `s16_blocked` | the served artefact and section 4 (no metric, no 2023 rows) |
| `s20_floor`, `s21_floor_fair` | idea 2 |
| `s22_large_markets` | idea 7, with the package's `large_markets` (added by #123, after the review) |
