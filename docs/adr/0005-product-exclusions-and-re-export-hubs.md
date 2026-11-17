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
