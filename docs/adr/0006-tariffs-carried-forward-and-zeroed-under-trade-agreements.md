# Applied Tariffs carried forward from 2021 and zeroed under Trade Agreements

v2 adds the Applied Tariff as a policy feature, as ADR 0002 anticipated. It is `ln(1+t)`, monotone-constrained at −1 and clipped to the training-year range. TRAINS, the only free bulk bilateral HS6 source, ends in 2021. The only routes to 2022–2024 data are a bulk-download permission from WTO or a MAcMap v2022 request to CEPII ([research note](https://github.com/noahlb1/exportspredictor/blob/research/tariff-data-v2/docs/research/tariff-data-v2.md)), and neither will be pursued. So **2021 tariffs are carried forward to 2022–2024**, which are the validation year, the test year and the whole Gap Period. Every carried row is marked as imputed. That mark is metadata, not a model input, because it is 1 only in years that are never trained on.

Two couplings keep the tariff consistent with the Trade Agreement flags, so the model is never asked about a combination it did not see in training:

- **Agreements in force after 2021** (for example RCEP, UK–Australia, UK–New Zealand and UK CPTPP accession): the carried-forward tariff is set to 0 for those pairs. This overstates liberalisation during phase-in, but it is closer to reality than an agreement with full MFN tariffs.
- **Scenarios**: adding a Trade Agreement also sets the Applied Tariff to 0. Removing one restores the Importer's 2021 MFN rate. A new tariff Scenario sets the rate a Market applies to UK goods, optionally for chosen Sectors only. Rates beyond the training range are clipped, so the Scenario Effect of a very large hike flattens, and the app says so.

## Considered Options

- **Ask WTO for ADB bulk access, or CEPII for MAcMap v2022.** These would give real 2022+ rates, but depend on replies the project will not seek.
- **Leave 2022–2024 tariffs missing.** The evaluation and Baseline years would then carry no tariff information, and Scenarios would have no Baseline to change.
- **Carry 2021 forward unchanged even under new agreements.** This produces an "agreement with full tariffs" pattern that is absent from training, just where the model is judged.
- **Use the Importer's best 2021 preferential rate for new agreements.** It is a proxy for the new schedule, with no better grounding than 0.
- **Trade Agreement Scenarios that flip only the flag.** These would understate the Scenario Effect, because training pairs with an agreement almost always also have low tariffs.

## Consequences

- The notebook reports test metrics split by imputed and observed tariff rows.
- The API's pair detail carries a "tariff carried forward from 2021" caveat.
- If real 2022+ data ever arrives, it replaces the carried values without changing the feature's definition.
