# Train on the top-100 Exporters and every Market, keeping all UK rows and sampling the rest

The full HS6 world, with all its Zero Cells, does not fit in the 10 GB of VRAM. v1 fitted it by keeping a 30% sample of non-UK rows and 2% of Zero Cells.

In v2 the model trains only on Flows and Zero Cells whose Exporter is one of the **top 100 economies by mean GDP over 2012–2023**, the UK always included. **Every Importer is kept.** All UK Flows and all UK Zero Cells are kept at weight 1. Other Exporters' Zero Cells are sampled at a starting rate of 5%. Their rows are then sampled at whatever rate fits a VRAM budget, and a learning curve in the notebook settles the final rates. Training runs in-core on the GPU, with the `QuantileDMatrix` built from device chunks; that fits about 110 M rows. Sampled rows carry the inverse of their sampling rate as their weight, so weighted totals stay unbiased.

The cut is on Exporters only, because the UK is only ever scored as an Exporter. Keeping every Market means every UK Product–Market pair is scored by a model that trained on Flows into that Market. Dropping the 127 smallest Exporters loses 1.7% of world trade but 35% of the Zero Cells. A small Exporter adds a Zero Cell for every Importer of each Product it ships, and the space it frees pays for a higher zero rate. A higher zero rate sharpens the Export Potential of New Market and New Product Opportunities, which are themselves Zero Cells.

## Considered Options

- **Full world, sampled.** At a given row budget it needs a lower zero rate or a lower non-UK rate. It gains only the 1.7% of world trade that small Exporters carry.
- **Top-100 on both sides.** At 5% zeros it is almost unsampled, but it drops 126 of the UK's 225 Markets, Estonia, Latvia, Cyprus and Malta among them. Those Markets would then be either unscored or scored out of distribution.
- **Top-50.** It keeps under half of UK Flows by count.
- **CPU `hist`.** It fits no more rows (about 100 M in 32 GB) and is about 15× slower.
- **GPU external memory**, spilling past VRAM to host memory. It trains all rows of the full world, but at about 2.3 h per model, so about 12 h per set of fold models, which rules out tuning and experiments.

## Consequences

- **Small Exporters are never training rows**, but their Flows still feed every Importer-side and demand feature. The scope limits the rows, not the features.
- **The rates are recorded, not fixed.** The non-UK rate is derived from the VRAM budget for the actual rows and features, so it shifts when Product exclusions or Feature additions change the data. Each artefact records the rates it was trained with, and the governance text reports them.
- **A last resort remains.** Once the model is tuned and the features are final, if the learning curve is still rising at the in-core budget, the final fold models could be trained once on every Exporter in the world, unsampled, using GPU external memory. It would be an overnight run with the tuned parameters, not part of v2's plan. If it is done and kept, a superseding ADR records it.
