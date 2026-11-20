# Potential Range is conformal, calibrated on other Exporters' Baseline-year residuals, never the UK's

Each UK Product–Market pair gets a Potential Range: an 80% prediction interval for Actual Exports, built by split conformal prediction. The obvious calibration set is the UK's out-of-fold residuals (ADR 0001), and we deliberately do not use it. A UK pair's residual, Actual Exports minus Export Potential, is its Export Gap with the sign flipped. That gap is the signal the model exists to find, so an interval calibrated on it would only measure "a typical UK gap", and every gap would look normal. Instead we calibrate on **other Exporters' Flows and Zero Cells in the Baseline year**. The fold models train only on earlier years, so those residuals are honestly out of sample. Each calibration row is scored by the fold model its hash assigns it to, as UK pairs are.

Calibration is done separately within 20 equal-count bands of Export Potential (Mondrian conformal), because Tweedie errors grow with the prediction. Each band needs at least 1,000 rows, or neighbouring bands are merged. Within each band, the 10th and 90th percentiles of signed residuals become the range's offsets, and the range is clipped at zero. Residuals are on the log scale, log(1 + Actual Exports) − log(1 + Export Potential), so each offset is a ratio of Export Potential (*Amendment* below). An Opportunity whose Gap Period mean Actual Exports lies inside the range is an Uncertain Opportunity.

## Considered Options

- **UK out-of-fold residuals**: circular, as above.
- **XGBoost quantile objectives**: these need two extra models per fold, can cross, carry no coverage guarantee, sit outside the single monotone Tweedie model (ADR 0002), and give a lower quantile that is almost always 0 on zero-heavy HS6 data.
- **Spread across the five fold models**: four of the five trained on any given UK pair, so their spread mixes one held-out prediction with four in-sample ones and understates uncertainty.
- **Calibrating on other Exporters' 2022–2023 rows**: these are training years for the fold models, so their residuals are optimistic.

## Consequences

- **Exchangeability between the UK and other top-100 Exporters is assumed, not guaranteed.** The notebook reports the coverage achieved on a held-out half of the calibration rows, split by Exporter, overall and per band.
- **The comparison runs over the Gap Period, but calibration uses only 2024.** The 2024 band offsets are applied around the Gap Period mean Export Potential and compared with the Gap Period mean Actual Exports. A three-year mean is less noisy than one year, so the range is wider than strictly needed. This errs towards flagging more Opportunities as uncertain, never fewer.
- **Scenario Effects get no interval.** Conformal prediction covers outcomes, not differences between two predictions, and Event Backtests remain the check on Scenario Effects.

## Outcome (spec #66, artefact `20261118T185910Z`)

- **The Uncertain Opportunity is reported as a share, not shown per row.** On the served version 904,649 of 1,032,453 Opportunities (88%) are Uncertain Opportunities, so a badge on each one no longer separated one Opportunity from another. The badge, its caveat on the labels the API serves and the "Hide Uncertain Opportunities" filter were removed from the ranking pages and the ranking query.
- **The Potential Range itself is unchanged.** Its calibration, the artefact's tables (each UK pair keeps its range and its `uncertain_opportunity` flag) and the artefact format stay as decided above. The range is still shown on the Why? page for a pair, and the Model Evaluation page states its held-out coverage and the share of Opportunities whose Actual Exports lie inside it.

## Amendment: residuals on the log scale (artefact `20261120T133841Z`)

The offsets were first dollar residuals. The top band runs from about $2m of Export Potential to over $1bn, and its rows are mostly near $2m, so it gave every pair in it one width: about −$8m to +$16m. A $929m pair got a range of $921m to $945m. The band's 80% coverage was real but came from its small pairs. Split by Export Potential on the held-out half of the calibration rows, the dollar range covered **6.8%** of pairs above $100m (6,804 rows) and 42% of pairs from $10m to $100m.

Residuals are now log(1 + Actual Exports) − log(1 + Export Potential), and the range is exp(log(1 + Export Potential) + offset) − 1. Conformal coverage holds for any score, so the bands, quantiles, calibration rows and held-out check are unchanged. On the same held-out half, the log range covers 85% below $2m, 78% from $2m to $10m, 84% from $10m to $100m and 86% above $100m, and 85.1% overall (dollar offsets: 85.5%).

- **Ranges for large pairs are wide, mostly downwards.** In the top band, comparable pairs' Actual Exports run from about 1% to 2.8 times their Export Potential, because many of them trade far less than predicted or nothing. The same $929m pair now has a range of $8m to $2.6bn.
- **The largest Export Gaps are now Uncertain Opportunities too.** 87% of Opportunities are (88% before), but so are 94% of the top 1,000 by Export Gap (0% before) and every Opportunity with Export Potential above $100m (5% before). The dollar range had made large gaps look certain only by being too narrow. An Export Gap ranks a lead; it does not prove one.
- **The artefact format stays 3.** The bands table holds `low_offset_log` and `high_offset_log` instead of `*_offset_usd`, and `metadata.potential_range.scale` is `"log"`. An older version with dollar offsets still loads and still reports its fixed width as a known limitation. `python -m exportspredictor.pipeline rerange` rebuilds a version's range from its saved fold models, without retraining.

### Considered Options

- **More bands at the top.** Each band would still give one dollar width across a wide span of Export Potential, only a narrower span.
- **Residuals divided by Export Potential** (normalised conformal). Unstable near zero, where most rows lie, and a ratio of Actual Exports to Export Potential is what the log residual already measures.
