# Exclude Products whose Flows the model cannot explain; flag the rest of Sector 71 and Re-export Hubs

In v1, the largest Export Gaps were dominated by crude oil, pipeline gas and gold. These Flows follow bullion-market and pipeline geography that the model cannot see, and gold's recorded values are the least reliable in the data: BACI and HMRC disagree badly on UK gold. ITC's Export Potential Map, the closest published analogue, excludes almost the same Products ([research note](https://github.com/noahlb1/exportspredictor/blob/research/misleading-products/docs/research/misleading-products.md)).

v2 therefore has two tiers of Product treatment and one Market flag:

- **Excluded Products** are dropped as rows from training, scoring and every ranking. The app and the governance text list them with their reasons:
  - **Sector 27 except 2712**: crude and refined oil, gas, coal and electricity (2716). Their trade follows pipelines, grids and tankers.
  - **7106 silver, 7108 gold, 7110 platinum-group metals, 7112 precious-metal scrap and 7118 coins**: bullion trade, which is volatile and poorly recorded.
  - **9706 antiques**: one-off objects, not production capacity.
  - **2844 radioactive elements**: strategic, licensed trade.
- **Flagged Products** are scored and ranked as normal but shown with a caveat. This tier covers the rest of Sector 71: 7101–7105, 7107, 7109, 7111 and 7113–7117. Their values are volatile and entrepôt-driven, but they are made goods the UK genuinely sells.
- **Re-export Hubs** are flagged as Markets, with a filter to hide them. The list is NLD, BEL, IRL, HKG, SGP, ARE, CHE and LUX. They stay in training and rankings because several are top genuine UK Markets.

Excluded Products are removed from training as well as from the rankings, not just hidden. Gold moves by a factor of three to four a year, and the Tweedie loss weights large Flows heavily, so keeping these rows would shape splits for every other Product. Dropping them also frees VRAM for the non-UK sampling rate (ADR 0004).

## Considered Options

- **Exclude from the rankings only.** It keeps the distorting Flows in the loss and spends VRAM on rows that are never reported.
- **Exclude or flag medicaments (3004, 3002).** They are the largest single group (5% of the positive gap), and transfer pricing is documented. But HMRC and BACI agree on them, no agency or potential model treats them specially, and they are a genuine UK strength. They stay normal.
- **Exclude arms (93, 8710, 890610), as ITC does.** Export licensing is a lever policy makers control directly, and the group barely touches the rankings (0.2% of the gap, none of the top-1000). Arms stay normal.
- **Exclude Re-export Hubs.** That would remove the Netherlands and Ireland, two of the UK's largest Markets.
- **ITC's supply-side re-export filter** (chapters 73 and 84–90). BACI does not separate re-exports, and the UK is scored only as an Exporter on fundamentals, not through a supply index.
- **Exclude waste and scrap, as ITC does.** The UK is a large genuine scrap exporter, and the group holds 1.7% of the gap. Only precious-metal scrap (7112) is excluded.

## Consequences

- **Rows are dropped, but totals are not.** Every country and world total still counts Excluded Products. This covers the Exporter total and the world totals inside RCA, so Saudi Arabia's or Norway's RCA in other Products is not inflated. The exclusion limits the rows, not the features, as in ADR 0004.
- **Every metric is computed over non-Excluded Products.** This covers within-Product Spearman, Lift, the Event Backtest and the Sector breakdowns. v1 is re-scored on the same Product set when the two are compared, so v2 is not flattered by dropping the hardest Products. The notebook also reports v1 on all Products, to show what the exclusion alone changes.
- **A hub feature is left to Feature additions** as a candidate (for example the Importer's re-export intensity), so it is not decided here.

## Amendment: Sanctioned Markets and Restricted Pairs (spec #113, ticket #119)

Two more Caveat Tiers mark what no feature records, and that an Advisor must know before presenting a pair as a lead:

- **Sanctioned Markets**: Markets under broad UK trade sanctions in the Baseline year. On 20 November 2026 these were Russia and Belarus (broad since 2022), North Korea (since 2007) and Syria (2011 to 2024, before SI 2025/507 removed its trade, oil, investment and financial restrictions). Iran is not one: gov.uk calls both UK Iran regimes "limited in scope", as they restrict named goods, not trade at large. A Market is flagged only in Baseline years its regime covers, so a later rebuild neither flags a regime that has ended nor labels years before one began. Each row's caveat names the regime and the year it became broad.
- **Restricted Pairs**: Products a Market prohibits outright. On the same date these were alcoholic beverages (HS 2203–2206 and 2208) into Saudi Arabia, Kuwait, Libya and Iran. Ethyl alcohol (HS 2207) is left out: it is restricted (imported under approval for industrial and medical use), not prohibited outright.

**Caveats, not exclusions.** Both tiers are scored, ranked and measured as normal, like Re-export Hubs, and each has a hide filter that is off by default. Dropping them would barely move the evaluation: leaving Russia and Belarus out of the 2022 Selection Measure raises it by only +0.003 ([open review](https://github.com/noahlb1/exportspredictor/blob/research/open-review-2022/docs/research/open-review-2022.md), section 4). But the harm to an Advisor in the few Markets affected is large: the served artefact held about 20,000 Opportunities in Sanctioned Markets ($1.2bn of Export Gap, 138 in the top 10,000), and whisky to Saudi Arabia ranked #552. Removing them would also hide the Markets from an Advisor asked why the UK does not sell there. Sanctions as a model feature were left out: worth +0.004 at most, and no model can know a prohibition.

**Hand-kept, dated tables.** Both lists live in `exportspredictor.restrictions`, each entry with its source (gov.uk sanctions guidance and the legislation that sets a regime's years; official trade guidance or law for each prohibition) and the date they were last checked (`VERIFIED`). They change only when someone checks them again; `python -m exportspredictor.pipeline recaveat` then marks the saved Baseline without retraining.

## Amendment: volatile and one-off Flagged Products (spec #113, ticket #120)

The open review found that the Products whose UK exports swing most from year to year rank far worse: the 2022 Selection Measure falls from 0.76 for the calmest fifth of Products by volatility to 0.30 for the lumpiest ([open review](https://github.com/noahlb1/exportspredictor/blob/research/open-review-2022/docs/research/open-review-2022.md), section 6). Their Export Gaps are mostly timing: a contract delivered one year and not the next. Flagged Products therefore gain a rule beside Sector 71, and each Flagged Product's caveat now names its reason.

- **The rule.** A Product's volatility is the standard deviation of the year-to-year log change of the UK's total exports of it (a year with none counts as $1). It is read over the training years only: the years before the validation year, 2012–2021 for the 2024 Baseline, so neither the validation year, the test year nor the Baseline year decides which Products are flagged. Products above the 80th percentile of volatility among the scored Products are flagged with the reason "volatile: UK exports swing widely from year to year". On the served artefact the threshold is 0.734 and the rule flags 1,028 Products, 1,018 of them outside Sector 71 (`exportspredictor.volatility`).
- **Named one-offs.** The Products the review's case study named are flagged with the reason "one-off: few, large contracts" when the rule misses them: 4907 (banknotes and stamps, sovereign contracts), 8802 (aircraft), 8901 and 8905 (vessels and floating platforms) and 284390 (precious-metal compounds, probably one producer's Flows). On the served artefact the rule already catches every vessel and platform and every aircraft but light helicopters, so three Products carry the one-off reason: 490700, 880211 and 284390.
- **Sector 71** keeps its own reason, which wins where a Product is both.

The threshold, the years, the share flagged and every flagged Product with its reason are recorded in the artefact metadata (`volatile_products`). `python -m exportspredictor.pipeline recaveat` recomputes the rule from the training years' BACI Flows and marks a saved Baseline without retraining; a training run computes it from the same Flows. Like the other Caveat Tiers, the rule changes no score, ranking or metric. Volatility as a model feature was not tried here: it would change the model, and the caveat's purpose is to tell the Advisor, not to rank differently.

## Amendment: Offset Pairs (spec #129, ticket #131)

The two largest Opportunities on the served artefact (`20261120T221726Z`, Baseline year 2024) are turbine parts to France (Export Gap $995m) and turbo-jets to Germany ($745m). Each sits beside a surplus of about the same size one Product away: the rest of Heading 8411 sells France $1,019m above its Export Potential, and Germany $769m. Export Potential for the whole Heading is within 1.3% of Actual Exports in both Markets. The model can be right about a Heading and wrong about the split between its Products ([research note](../research/offset-pairs.md)).

A fifth Caveat Tier, **Offset Pairs**, marks these pairs.

- **The rule.** A Heading is the first four digits of a Product's HS6 code. For each Market and Heading, the Heading gap is the summed Export Gap of the Heading's scored Products in that Market; Excluded Products are not on the Baseline, so they are left out. A pair is an Offset Pair when its Export Gap is positive, its Heading has at least two scored Products in that Market, and the Heading gap is at most half of its own Export Gap. Equivalently, the rest of the Heading sells above its Export Potential by at least half of the pair's Export Gap. The threshold (one half) is one named constant, recorded in the artefact metadata. Each row's caveat gives the offset, the Heading's code and the other Product with the largest surplus, and says the Export Gap "may be trade recorded under a neighbouring Product".
- **A caveat, not a feature.** What separates France from Germany in 8411 is the UK's own trade with each Market, which a feature may not read (the leave-one-out guardrail). And across Headings the relation runs the other way: on the 2022 held-out scores of the five-fold confirmation, a pair's log residual and the rest of its Heading's move together (correlation +0.24 for pairs above $10m; only 15 of 107 Headings with at least 15 such pairs are negative; +0.34 with Market Reach applied). A permitted feature built from a pair's neighbours would associate an over-performing Heading with a higher Export Potential for the pair, which makes the 8411 cases worse.
- **A caveat, not an exclusion.** Products that share a Heading may be stages of one supply chain (engines and their parts), near-substitutes (cars of two engine sizes) or unrelated lines of business. The rule cannot tell them apart, so the Advisor decides. Like Restricted Pairs, Offset Pairs are shown by default with their caveat, and a hide filter is off by default.
- **The guardrails are untouched.** The rule reads Actual Exports only after scoring, beside the Baseline's out-of-fold Export Potential. No model reads a pair's own Actual Exports ([ADR 0001](0001-out-of-fold-uk-training.md)), and no feature changes. It changes no score, ranking or metric, trains nothing and adds nothing to the Configuration Fingerprint.
- **The tier's reach.** On the served artefact the rule marks 246,790 of 898,686 Opportunities (**27.5%**), 21 of the top 100 by Export Gap, 159 of the top 1,000 and 1,709 of the top 10,000, holding $19.0bn of the $97.7bn of positive Export Gap (19.4%). That is broader than Restricted Pairs and close to Flagged Products (a fifth of Products). Uncertain Opportunity was kept off the rows because it covers most of them ([ADR 0007](0007-potential-range-calibrated-on-other-exporters.md)); this tier covers about a quarter, so its badge still separates rows. "At least half" was chosen over "fully offset" (19 of the top 100), so that partial offsets such as turbine parts to the USA are marked too.
