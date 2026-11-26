# Applied Tariffs carried forward from 2021 and zeroed under Trade Agreements

v2 adds the Applied Tariff as a policy feature, as ADR 0002 anticipated. It is `ln(1+t)`, monotone-constrained at −1 and clipped to the training-year range. TRAINS, the only free bulk bilateral HS6 source, ends in 2021. The only routes to 2022–2024 data are a bulk-download permission from WTO or a MAcMap v2022 request to CEPII ([research note](../research/tariff-data-v2.md)), and neither will be pursued. So **2021 tariffs are carried forward to 2022–2024**, which are the validation year, the test year and the whole Gap Period. Every carried row is marked as imputed. That mark is metadata, not a model input, because it is 1 only in years that are never trained on.

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

## Addendum: an unknown tariff stays unknown under an added agreement

*Added in the code review of spec #39.* "Adding a Trade Agreement sets the Applied Tariff to 0" applies only where the tariff is known, observed or carried forward. Where a pair has no tariff at all (TRAINS reports none, so the feature is missing), an added agreement leaves it missing, in Scenarios, in the Trading Bloc presets, in the Blocked Opportunity values and in the Event Backtest's "agreement on" state. One expression in `exportspredictor.scenarios` (`AGREEMENT_TARIFF`) states the rule for all of them.

A missing tariff is unknown, so "removing" it is not a defined policy change. And XGBoost's monotone constraints bound the model only along observed values: a missing value follows each split's learned default branch, which the constraint does not cover, so a missing tariff replaced by 0 can lower a prediction. In the first real v2 artefact this gave 22,135 UK pairs, every one with a missing tariff, a negative Trade Agreement Scenario Effect. With the rule, the Trade Agreement flag is the only change on those pairs, and its +1 constraint keeps their Scenario Effect non-negative.

Considered and rejected: imputing missing tariffs and retraining (it re-tunes the model after the test year was read), and clipping negative effects at 0 (it hides the cause).
