# Where the blend's 2022 Opportunity ranking fails, and which features might help

A read-only error analysis of the served form (the blend, naive weight 0.3, ADR 0008) on the **2022 validation year**. It trains nothing. It reads the five-fold confirmation's held-out UK scores (`data/derived/loop_V202601/confirmation/fold_<k>.parquet`, #76) and 2022 Flows. **No 2023 file or test-year output was read.** Its purpose is to rank candidate changes before any GPU time is spent on them in the experiment loop.

Every result here is an association on held-out UK pairs in one year. A candidate that "helps" re-ranks the UK's 2022 Markets closer to what the UK exported that year. That does not say why, and it is not a forecast.

## Method

- **The measure** is the selection measure of `docs/research/opportunity-ranking-experiments.md`: the mean over Products of the Spearman correlation between Export Potential and Actual Exports across Markets (Products with at least 3 Markets), pooled over the five folds, 1,156,050 UK pairs and 5,084 ranked Products. Reproduced here: **blend 0.5635**, model alone (v2) 0.5206, naive baseline 0.5332, as in #76.
- **Candidate features** are built from 2022 Flows only, as every demand feature of the model is built from its own year's Flows. Every one except idea 6 leaves the UK out: F1–F3 read other Exporters' Flows only, and F4 compares Exporters with a UK export basket that leaves out the target Market, as the leave-one-out rule asks (CLAUDE.md).
- **The test is a re-ranking of the held-out scores**: score = log(blend) + β · z(feature), with z the feature standardised over the grid. β is chosen on folds 0–2 from a grid of −8 to 8 and checked on folds 3–4. Every fitted β is inside the grid. This measures what a feature adds *beyond* the blend, used linearly. A model trained with the feature could gain more (interactions) or less (it may learn part of it from features it has), so treat each gain as an estimate of size, not a result. It is compared with the loop's win threshold, **0.00887**, which came from one fold's seed noise; the fit and check folds agree to within 0.001 for every candidate, so the gains below are not a fitting artefact.

## 1. Where the ranking fails

**The loss is mostly in "which Markets trade at all", not in the order of the Markets that do.**

| 2022, pooled five folds | Blend | Naive baseline |
|---|---|---|
| Selection measure | 0.5635 | 0.5332 |
| Order among Markets with UK trade (non-zero only) | 0.501 | 0.434 |
| Any trade against none (zero vs non-zero) | 0.524 | 0.496 |

Two oracles bound the headroom. If each Product's Markets with UK trade were known and put first, in the blend's order, the measure would be **0.696** (+0.133). If which Markets trade were left as the blend has it, but the traded Markets were put in their true order inside the slots the blend gave them, it would be **0.595** (+0.032). So about four fifths of the reachable gain is in presence.

Taking, for each Product, as many top-ranked Markets as the UK actually sold to (k): 77% of them had UK trade. The 77,394 that did not ("said trade, got none") carry $6.5bn of Export Potential. The same number of Markets with UK trade were ranked below the top k ("said none, got trade"), but they hold only **1.8%** of UK export value: the misses in that direction are small Flows.

**By breadth.** The fewer Markets the UK sells a Product to, the worse the ranking, and for the narrowest Products the blend loses to the naive baseline.

| Markets the UK sells the Product to | Products | Blend | Naive | UK exports |
|---|---|---|---|---|
| 1–2 | 105 | 0.060 | 0.051 | $0.0bn |
| 3–10 | 382 | 0.166 | **0.220** | $0.5bn |
| 11–30 | 904 | 0.353 | **0.360** | $7.6bn |
| 31–80 | 1,988 | 0.579 | 0.539 | $71bn |
| 81+ | 1,705 | 0.777 | 0.717 | $257bn |

By UK export value the pattern is the same: the smallest fifth of Products scores 0.275 (naive 0.295), the largest 0.753 (naive 0.709).

**By Sector** (HS2 chapters with at least 15 Products). Weakest: 26 ores, slag and ash (0.25), 03 fish (0.27), 47 pulp (0.30), 01 live animals (0.30), 02 meat (0.34). Strongest: 34 soaps (0.80), 33 essential oils and cosmetics (0.80), 42 leather articles (0.78), 94 furniture (0.76), 90 instruments (0.76). The blend ranks below the naive baseline in ten chapters, mostly raw materials and commodities: 51 wool (−0.055), 26 ores (−0.049), 81 other base metals (−0.046), 05 animal products (−0.046), 75 nickel (−0.043), 41 hides (−0.033), 47 pulp (−0.030), 37 photographic goods (−0.028), **29 organic chemicals** (−0.023, 329 Products, $11bn of UK exports) and 71 precious metals and stones (−0.021).

**By Market region** (rank correlation within each Product over the region's Markets): North America 0.83, Europe & Central Asia 0.67, East Asia & Pacific 0.62, South Asia 0.62, Middle East & North Africa 0.51, Sub-Saharan Africa 0.39, Latin America & Caribbean 0.35, unclassified territories 0.21. The blend beats the naive baseline in every region.

**Over-rated Markets.** Measured as the mean, over Products, of the Market's predicted rank minus its actual rank (as a share of Markets), the most over-rated are Russia (+0.28), Mexico, Iran, Egypt, Viet Nam, Bulgaria, Colombia, Ukraine, the Dominican Republic and Morocco. Russia, Iran and Ukraine are where sanctions and war cut UK trade in 2022, which no feature records. The measure is distorted by ties at $0 for small Markets, so the under-rated end (small Pacific islands) is not read.

## 2. Worked examples

**The worst-ranked Products** with at least 15 UK Markets and $50m of UK exports are commodities and goods sold through a few large contracts: heavy helicopters (880212, 0.27), oil-cake (230400, 0.34), pig carcasses (020311, 0.34), propylene (290122, 0.38), soya beans (120190, 0.39), semi-finished steel (720712, 0.40; naive 0.45), slag and ash (262099, 0.40; naive 0.45), hot-rolled steel coils (720838, 0.40), cigarettes (240220, 0.41) and ammonium nitrate (310230, 0.41).

**The largest misses in both directions are aerospace, pharmaceuticals and cars,** where a handful of plants and corporate supply chains decide the Flow:

| Over-predicted (said trade, got little) | Actual | Blend | | Under-predicted (said little, got trade) | Actual | Blend |
|---|---|---|---|---|---|---|
| 841112 turbojets → Hong Kong | $76m | $750m | | 293379 heterocyclic compounds → Ireland | $3.44bn | $11m |
| 841112 turbojets → China | $332m | $760m | | 300490 medicaments → Germany | $3.34bn | $421m |
| 880330 aircraft parts → China | $129m | $464m | | 870324 large cars → USA | $3.57bn | $1.00bn |
| 841112 turbojets → Ireland | $201m | $516m | | 841191 turbine parts → Hong Kong | $2.59bn | $180m |
| 880230 aeroplanes → USA | $16m | $326m | | 880330 aircraft parts → Germany | $2.43bn | $278m |
| 880330 aircraft parts → Singapore | $37m | $335m | | 841112 turbojets → Singapore | $2.45bn | $442m |
| 300490 medicaments → Italy | $81m | $376m | | 870323 cars → China | $1.60bn | $214m |
| 841191 turbine parts → France | $107m | $391m | | 300490 medicaments → Ireland | $1.46bn | $83m |

**Turbine parts (841191), France against Germany**, the case that started this note. The blend gives France $391m (actual $107m) and Germany $202m (actual $1.18bn). On the candidate features the two look alike: other suppliers 82 and 70, supplier concentration (HHI) 0.48 and 0.42, Market imports of the HS4's other codes $3.9bn and $5.8bn, and UK-like share 0.98 and 0.99. Idea 6b, the UK's exports of the same HS4's other codes to the Market, points the wrong way: $1.42bn to France against $0.60bn to Germany. **No candidate here separates them.** The difference is which engine programmes each country's plants build, which country-level trade data cannot see. Hong Kong and Singapore, aircraft maintenance and re-export hubs, appear on both sides of the table for the same reason.

## 3. Candidate features

Re-ranking gain in the selection measure, β fitted on folds 0–2 and checked on folds 3–4; "all" is the five folds pooled at that β. "Residual association" is the mean within-Product Spearman correlation of the feature with the blend's log residual, log(1 + Actual) − log(1 + blend), over every pair (and over traded pairs only).

| Candidate (2022 Flows) | Residual assoc. (all / traded) | β | Gain, folds 0–2 | **Gain, folds 3–4** | Gain, all | vs 0.00887 |
|---|---|---|---|---|---|---|
| **F1** other Exporters supplying the Product to the Market (log count) | +0.18 / −0.14 | 5 | +0.0305 | **+0.0296** | +0.0304 | 3.3× |
| F2 concentration of those suppliers (HHI) | −0.08 / +0.03 | −1 | +0.0114 | **+0.0114** | +0.0115 | 1.3× |
| F3 the Market's imports of the HS4's other codes (log) | +0.12 / −0.15 | 1.4 | +0.0134 | **+0.0129** | +0.0135 | 1.5× |
| F4 imports from UK-like Exporters (log value) | +0.08 / −0.21 | 1 | +0.0064 | **+0.0062** | +0.0065 | 0.7× |
| F4b imports from UK-like Exporters (share) | +0.11 / −0.00 | 1.4 | +0.0177 | **+0.0175** | +0.0179 | 2.0× |
| **F1–F4 together** (F1, F2, F3, F4b) | | | +0.0311 | **+0.0303** | +0.0310 | 3.4× |
| 6a UK exports to the Market, every other Product (log) | +0.07 / −0.09 | 5 | +0.0280 | **+0.0278** | +0.0281 | 3.1× |
| 6b UK exports to the Market, the HS4's other codes (log) | +0.20 / +0.02 | 3 | +0.0346 | **+0.0352** | +0.0349 | 4.0× |
| F1–F4 + 6b | | | +0.0481 | **+0.0484** | +0.0483 | 5.5× |

- **F1 carries almost all of F1–F4.** Together the four add +0.030, the same as F1 alone. Fitted jointly, F2's weight goes to about 0 and F3's turns negative: they say what F1 says. "Many other countries already sell this Product to this Market" is the strongest single sign that the UK does too.
- **The gain is in presence, not order.** F1–F4 together raise the "any trade" ranking by +0.037 and lower the order among traded Markets by −0.019. A trained model could use F1 for presence without paying that price, but the re-ranking cannot.
- **F4's "UK-like" Exporters** are, by cosine similarity of log export baskets with the UK's basket leaving out the Market, Germany, France, the USA, the Netherlands, Italy, Belgium, Spain and Poland. With raw values instead of logs, oil exporters (the USA, Algeria, Russia) came first, because a few huge Products decided the similarity.
- **Idea 6 helps most, and is the one the rules forbid as written.** CLAUDE.md requires every feature built from UK data to leave out the target pair's own Market, and 6a and 6b are UK data *about* that Market. 6b adds +0.017 over F1–F4, and it is the only candidate that also improves the order among traded Markets (+0.011 combined). Two risks argue against it without an ADR. Neighbouring HS6 codes often carry the same company's shipments under another code, so 6b can leak the target Flow back in. And a model fitted to the UK's existing Market pattern pulls Export Potential towards Actual Exports, which shrinks the Export Gaps ADR 0001 exists to protect. Its gain partly measures how much of the answer it already contains.

## 4. The pooled measure against the mean of folds (idea 5)

The pooled measure is 0.5635 and the mean of the five folds' own measures is 0.5751, a gap of 0.0116. That gap was read as the cost of mixing five fold models with slightly different levels within a Product. **It is not.** The naive baseline, which has no fold models, shows nearly the same gap: 0.5332 pooled against 0.5436 for the mean of folds (0.0104). The gap comes from measuring each Product over a fifth of its Markets, not from fold levels.

Aligning the levels confirms it. Shifting each fold's log blend per Product so its median gap to the naive baseline matches the Product's (predictions only, no outcome) gives 0.5578 (−0.006). Shifting each fold by one global offset gives 0.5635 (0.000). The five folds' global offsets differ by at most 0.07 in log. **Idea 5 is dropped.**

## 5. A new idea: the blend weight by Product breadth

Section 1 shows the model alone adding little for narrowly traded Products. Re-blending the model alone and the naive baseline with a weight chosen per band of Product breadth, where breadth is the number of Markets importing the Product from other Exporters (non-UK information only), chooses on folds 0–2:

| Markets importing the Product (non-UK) | Products | Naive weight chosen |
|---|---|---|
| up to 30 | 54 | 1.0 |
| 31–80 | 289 | 1.0 |
| 81–140 | 939 | 1.0 |
| 141 or more | 3,856 | 0.3 (as served) |

That is, for the 1,282 Products imported by 140 Markets or fewer, the naive baseline alone ranked the 2022 Markets better than any blend. The gain is +0.0099 on folds 0–2 and **+0.0095 on folds 3–4**, 1.07× the threshold: at the edge of the noise. It needs no training. A naive weight of 1.0 would also leave those Products with no Scenario Effect, so this would need a floor on the model's weight if Scenarios are to keep working there (ADR 0008).

## 6. Would the candidates narrow the Potential Range?

The Potential Range is set by the spread of log residuals, log(1 + Actual) − log(1 + Export Potential), within bands of Export Potential (ADR 0007, amended). Here the spread is the 10th–90th percentile width of the 2022 log residual, in natural-log units (a width of 3.0 means the 90th percentile is e³ ≈ 20 times the 10th), with the lower tail (median − 10th) and upper tail (90th − median) apart. Each candidate adjusts the level: log(1 + blend) + β · z(feature), with β chosen **separately within each size group** to minimise the width on fit rows and checked on held-out rows. A single β for every size narrowed the overall width by trading away large pairs' width, so it was not used. Two populations: the UK's held-out pairs (fit on folds 0–2, check on folds 3–4), and 1.5 million of the 2022 non-UK training rows that calibrate the range, scored by the loop's cached fold-0 model (trained on 2012–2021; nothing trained here), split in hash halves and weighted by their sampling weights. Every non-UK feature leaves the row's own Exporter out. F4b needs a similarity for every Exporter and was built for the UK only. Width before → after, on the check rows:

| Candidate | UK, all pairs | UK > $10m (1,432) | UK > $100m (80) | Non-UK, all | Non-UK > $10m (10,152) | Non-UK > $100m (776) |
|---|---|---|---|---|---|---|
| Blend as served (lower / upper tail) | 8.46 (3.14 / 5.32) | 3.09 (1.90 / 1.19) | 2.99 (1.70 / 1.29) | 7.76 (4.76 / 2.99) | 3.74 (2.52 / 1.22) | 2.39 (1.44 / 0.95) |
| F1 suppliers | 7.69 | 3.03 | 2.84 | 7.56 | 3.74 | 2.47 |
| F2 concentration | 8.09 | 3.18 | 2.84 | 7.69 | 3.74 | 2.45 |
| F3 HS4 sibling imports | 8.32 | 3.09 | 2.93 | 7.64 | 3.74 | 2.38 |
| F4b UK-like share | 8.41 | 3.05 | 2.92 | – | – | – |
| F1–F4 (non-UK: F1–F3) | 7.68 | 3.14 | 2.88 | 7.52 | 3.72 | 2.47 |
| 6a own exports to the Market, other Products | 8.20 | 3.04 | 2.96 | 7.38 | 3.73 | 2.41 |
| 6b own exports to the Market, same HS4 | 7.77 | 3.08 | 2.94 | 7.38 | 3.74 | 2.37 |
| F1–F4 + 6b (non-UK: F1–F3 + 6b) | 7.48 | 3.04 | 2.87 | 7.48 | 3.66 | 2.48 |

- **No candidate narrows the range for large pairs by much.** Above $100m, the best UK change is 0.15 log units (F1 and F2: 2.99 → 2.84, a 20-fold spread becoming 17-fold), on only 80 check pairs. On the 776 non-UK rows above $100m, which is what the range is calibrated on, no candidate narrows it by more than 0.02, and F1 widens it (2.39 → 2.47). Above $10m every change is within 0.1 either way. For F4b and the combinations above $10m, β reached the grid's edge (±4), so their figures there may understate a little.
- **The overall narrowing is the presence effect again.** Over all pairs, F1 narrows the UK width by 0.77 by raising the level of pairs with many other suppliers and lowering it where there are few, which mostly separates traded pairs from Zero Cells and reorders the tails (lower 3.14 → 4.80, upper 5.32 → 2.90). The range of a Product–Market pair with real trade barely moves.
- **"Compare only with pairs that already trade" does not narrow it either.** UK pairs with a Flow in 2021 are 28.9% of pairs but hold 98.9% of the UK's 2022 export value. Among them the width is 9.41 over all pairs (wider than 8.46, because some stopped trading), 2.99 above $10m (against 3.09) and 2.99 above $100m (the same: every UK check pair above $100m already traded in 2021). F1–F4 refitted on them changes the widths by −0.11 to +0.01 above $10m.

So, on 2022, the width of the Potential Range for large pairs, about a 20-fold spread between its 10th and 90th percentiles, is not something these features explain. They are candidates for ranking, not for a narrower range.

## Shortlist

"Range width" is the change in the 10th–90th width of the log residual above $100m, UK check pairs / non-UK check rows (section 6); negative is narrower.

| Rank | Change | Evidence (folds 3–4) | Against 0.00887 | Range width > $100m | GPU cost | Guardrails |
|---|---|---|---|---|---|---|
| 1 | **F1**: other Exporters supplying the Product to the Market | +0.030, re-ranking; works through presence | 3.3× | −0.15 / +0.08 | Feature rebuild (CPU) and one loop arm on fold 0 (about 11 min), then the five-fold confirmation (four models, about 50 min) | None: non-UK Flows of the row's year, as the demand features |
| 2 | Blend weight by Product breadth | +0.0095 | 1.07×, marginal | not measured | None: scoring only, one loop arm with `grid` | Must keep a model weight above 0 where Scenarios matter |
| 3 | F4b: share of imports from UK-like Exporters | +0.0175 alone; adds nothing beside F1 | 2.0× alone | −0.07 / not built | As F1, as a second arm | UK basket leaves out the Market: complies |
| 4 | F3 HS4 sibling imports; F2 HHI | +0.013; +0.011; redundant with F1 | 1.5×; 1.3× | F3 −0.06 / −0.01; F2 −0.15 / +0.06 | As F1 | None |
| — | Idea 6b: UK exports of the HS4's other codes to the Market | +0.035 alone, +0.017 beyond F1–F4 | 4.0× | −0.05 / −0.02 | As F1 | **Breaks the leave-one-out rule; leakage through neighbouring codes; shrinks Export Gaps.** Needs an ADR and a leakage test before any arm |
| — | Idea 5: align fold levels | 0.000 / −0.006 | — | — | — | Dropped: the gap is a measurement effect |
| — | Compare only with pairs that traded in 2021 | not a ranking change | — | 0.00 / not measured | None | — |

The largest single misses (aerospace, pharmaceuticals, cars, re-export hubs) are not separated by any candidate. They follow company supply chains that country-level data do not record.

## Reproducing

From a checkout, with the shared data directory (`scripts/wt.ps1` in a worktree):

```
uv run python scripts/research/ranking_error_2022/build.py      # the table: held-out 2022 scores, lookups, candidate features (CPU, about 2 min)
uv run python scripts/research/ranking_error_2022/analyse.py    # sections 1-4 (CPU, a few minutes)
uv run python scripts/research/ranking_error_2022/analyse2.py   # the wider beta grid, combinations and section 5
uv run python scripts/research/ranking_error_2022/range_width.py  # section 6: scores non-UK rows with the cached fold-0 model (GPU, prediction only); stops if the model is missing
```

They write `grid.parquet`, `results.json`, `results2.json` and `range_width.json` to `data/derived/ranking_error_2022/`. They read `data/derived/loop_V202601/confirmation/`, the loop's cached fold-0 model and its 2022 validation rows, `data/derived/features_V202601/uk_grid_2022.parquet`, the 2021 and 2022 BACI Flows and the lookups, and nothing from 2023.
