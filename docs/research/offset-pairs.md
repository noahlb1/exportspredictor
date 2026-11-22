# Offset Pairs: an Export Gap offset by the same Heading in the same Market

For spec #129, ticket #131, under [ADR 0005](../adr/0005-product-exclusions-and-re-export-hubs.md) (Caveat Tiers), [ADR 0001](../adr/0001-out-of-fold-uk-training.md) (out-of-fold UK scores) and [ADR 0010](../adr/0010-market-reach-in-the-model-form.md) (Market Reach). The scripts used when the idea was first explored were not kept, so every figure here is recomputed by one committed script.

**Finding.** The model can be right about a Heading and wrong about the split between its Products. The two largest Opportunities on the Leaderboard each sit beside a surplus of about the same size one Product away, in the same Heading and Market. The spec's rule marks 27% of Opportunities, and about a fifth of the top of the Leaderboard at every depth. Across Headings, a pair's surplus and its neighbours' surplus move together (+0.24 on the 2022 held-out scores), so a feature built from the rest of the Heading would point the wrong way for these cases. The Offset Pair tier is therefore a caveat, not a feature or an exclusion.

**Every recomputed figure matches the spec's to rounding.** The one the spec rounds is the Heading's fit: it says "within about 1%" of Actual Exports, and the recomputed figure is within 1.3%. The rest match as stated: 27% of Opportunities (27.5%), 21 / 159 / 1,709 by rank, $19.0bn of $97.7bn of positive Export Gap, +0.24 (0.236) and 15 of 107 negative Headings.

**Nothing was trained and 2023 was not read.** The script reads the served artefact **`20261120T221726Z`** (Baseline year 2024, the lexically newest version folder when this was written), the loop's cached five-fold confirmation of 2022 (`data/derived/loop_V202601/confirmation/fold_<k>.parquet`) and, for the Market Reach check, the 2021 UK grid. It runs on the CPU in about 4 seconds.

Script: `scripts/research/offset_pairs/run.py`. Output: `data/derived/offset_pairs/result.json`.

    scripts/wt.ps1 python scripts/research/offset_pairs/run.py

## Terms

- **Heading**: the first four digits of a Product's HS6 code. Here it only groups a Product with its neighbours; the model is not trained or measured at this level.
- **The rule (spec #129).** For each Market and Heading, the Heading gap is the summed Export Gap of every scored Product of the Heading in that Market. A pair is an **Offset Pair** when its Export Gap is positive (it is an Opportunity), its Heading has at least two scored Products in that Market, and the Heading gap is at most half of its own Export Gap. Equivalently, the rest of the Heading sells above its Export Potential by at least half of the pair's Export Gap. The offset is the pair's Export Gap minus the Heading gap.
- The script applies the rule on its own, without the package's Offset Pair code (#130), so the two can be checked against each other (#132). They agree: the metadata `recaveat` recorded on version `20261122T010816Z` (made from `20261120T221726Z`, no score changed) gives the same 246,790 Offset Pairs, 21 / 159 / 1,709 by rank and $18.97bn of $97.74bn. Excluded Products are not on the Baseline, so they are left out of every Heading sum.

## 1. Heading 8411 in France and Germany

UK exports of Heading 8411 (turbo-jets, gas turbines and their parts) on the served Baseline. "Whole Heading" sums all 8 scored Products of 8411 in the Market.

| | Turbo-jets (841112) | Turbine parts (841191) | Whole Heading 8411 |
|---|---|---|---|
| France, Actual Exports | $1,761m | $97m | $1,940m |
| France, Export Potential | $688m | $1,092m | $1,915m |
| France, Export Gap | −$1,074m | **$995m** | −$25m |
| Germany, Actual Exports | $460m | $1,424m | $2,037m |
| Germany, Export Potential | $1,205m | $604m | $2,013m |
| Germany, Export Gap | **$745m** | −$820m | −$24m |

- **The Heading is right and the split is not.** Export Potential for the whole Heading is within 1.3% of Actual Exports in both Markets. France buys finished engines and few parts; Germany buys the reverse. The model puts most of France's Export Potential in parts and most of Germany's in turbo-jets, the reverse of each Market's Actual Exports.
- **So Opportunity #1 (turbine parts to France, $995m) and #2 (turbo-jets to Germany, $745m) each sit beside a surplus one Product away.** The rest of 8411 sells France $1,019m above its Export Potential, and Germany $769m. Both are Offset Pairs, fully offset.
- **No feature can see the split.** What separates France from Germany here is the UK's own trade with each Market, and a feature may not read a pair's own Market from UK data (the leave-one-out guardrail). The spec reports that the 2022 ranking-error review ([ranking-error-analysis-2022.md](ranking-error-analysis-2022.md)) found no candidate feature that separates the two.

## 2. How often the rule applies

On the served Baseline: 1,156,050 scored pairs, 898,686 of them Opportunities.

| | Offset Pairs | Share |
|---|---|---|
| All Opportunities | 246,790 of 898,686 | **27.5%** |
| Top 100 by Export Gap | **21** | 21% |
| Top 1,000 | **159** | 15.9% |
| Top 10,000 | **1,709** | 17.1% |
| Summed positive Export Gap | $19.0bn of $97.7bn | **19.4%** |

Ties in Export Gap are broken by Product, then Market. With a stricter "fully offset" rule (the Heading gap at or below zero) the top-100 count is 19, the top 1,000 131 and the top 10,000 1,474, so the "at least half" threshold adds partial offsets, not a different set at the top.

The ten largest Offset Pairs:

| Rank among Offset Pairs | Product | Market | Export Gap | Offset in the rest of the Heading |
|---|---|---|---|---|
| 1 | 841191 turbine parts | France | $995m | $1,019m |
| 2 | 841112 turbo-jets | Germany | $745m | $769m |
| 3 | 870323 spark-ignition cars, 1,500–3,000 cc | USA | $581m | $2,035m |
| 4 | 841191 turbine parts | USA | $434m | $367m (partial) |
| 5 | 870324 spark-ignition cars, over 3,000 cc | Germany | $395m | $999m |
| 6 | 880320 aircraft under-carriages and parts | USA | $313m | $726m |
| 7 | 300439 medicaments containing hormones or antibiotics, retail | USA | $306m | $595m |
| 8 | 870322 spark-ignition cars, 1,000–1,500 cc | USA | $206m | $1,660m |
| 9 | 841112 turbo-jets | China, Hong Kong SAR | $175m | $3,668m |
| 10 | 870323 spark-ignition cars, 1,500–3,000 cc | Belgium | $124m | $297m |

**Reading the reach.** About a quarter of Opportunities and a fifth of the positive Export Gap are Offset Pairs. That is broader than Restricted Pairs and close to Flagged Products (a fifth of Products). Uncertain Opportunity was kept off the rows because it covers most of them (ADR 0007); this tier covers about a quarter, so the badge still separates rows.

The rule covers supply-chain stages (engines and their parts) and near-substitutes (cars of two engine sizes) alike. It shows only that the Products share a Heading and that the surplus exists; it does not say the two are the same business.

## 3. Across Headings, a pair and its neighbours move together (2022)

If a pair's surplus were usually mirrored by a gap among its neighbours, a feature such as "the rest of the Heading over-performs" could correct cases like 8411. The held-out 2022 scores say the relation runs the other way.

**Method (spec #129, Further Notes).** On the loop's five-fold confirmation of 2022, each UK pair is scored by the fold model that held it out (ADR 0001). For pairs in a Heading with at least two Products in the Market, the residual is log(1 + Actual Exports) − log(1 + Export Potential), and the rest-of-Heading residual is the same log ratio of the other Products' summed Actual Exports and summed Export Potential in that Market. The correlation is Pearson, among pairs whose Actual Exports or Export Potential exceeds $10m. Per Heading, the same correlation is taken over Headings with at least 15 such pairs.

| 2022, pooled five folds | Pairs above $10m | Correlation | Headings with ≥ 15 pairs | Negative | Median Heading correlation |
|---|---|---|---|---|---|
| **Blend** (the spec's figure) | 5,609 | **+0.236** | **107** | **15** | +0.30 |
| Blend with Market Reach (β = 4) | 5,763 | +0.340 | 115 | 11 | +0.49 |

- **The general relation is positive.** Where the UK sells above its Export Potential in one Product, it tends to sell above it in the rest of the Heading in that Market too. Only 15 of 107 Headings run the other way. A feature learning from this would associate an over-performing Heading with a higher Export Potential for the pair, which moves turbine parts to France further from its Actual Exports, not closer.
- **With Market Reach the relation is stronger, not weaker.** The served Model Form multiplies Export Potential by a per-Market factor (ADR 0010), which the confirmation's `winner` column does not include. Applying it at the served β, with offsets from the 2021 UK grid and one rescaling constant over the five folds, raises the correlation to +0.34. So the conclusion does not depend on which of the two Export Potentials is used.
- **The 8411 cases are the exception the caveat is for**: Headings where the model has the total about right and the split wrong. They cannot be separated from the general case without reading the pair's own Market, which a feature may not do.

These are associations on held-out UK pairs in one year. They describe how the model's errors co-vary within a Heading; they say nothing about why the UK sells what it does.

## What it means for the tier

- **A caveat, not a feature.** No feature can see the split without breaking the leave-one-out guardrail, and a permitted feature built from neighbours would point the wrong way on average (section 3).
- **A caveat, not an exclusion.** The Export Gap may be real: the Products in a Heading can be unrelated lines of business. The Advisor decides, so the caveat says the Export Gap "may be trade recorded under a neighbouring Product".
- **It reads Actual Exports only after scoring.** The rule uses the Baseline's out-of-fold Export Potential and its Actual Exports, after every score is final. No model reads a pair's own Actual Exports (ADR 0001), and no feature changes (the leave-one-out guardrail).
- **It changes no score, ranking or metric.** Offset Pairs are scored, ranked and measured as normal.
- **Not tried here:** a log-target loss (the spec reports it tied on fold 0 of 2022 and widened the dollar spread), offsets across Headings or within a Sector, and ranking by a netted Export Gap.
