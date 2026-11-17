# Potential Range is conformal, calibrated on other Exporters' Baseline-year residuals, never the UK's

Each UK Product–Market pair gets a Potential Range: an 80% prediction interval for Actual Exports, built by split conformal prediction. The obvious calibration set is the UK's out-of-fold residuals (ADR 0001), and we deliberately do not use it. A UK pair's residual, Actual Exports minus Export Potential, is its Export Gap with the sign flipped. That gap is the signal the model exists to find, so an interval calibrated on it would only measure "a typical UK gap", and every gap would look normal. Instead we calibrate on **other Exporters' Flows and Zero Cells in the Baseline year**. The fold models train only on earlier years, so those residuals are honestly out of sample. Each calibration row is scored by the fold model its hash assigns it to, as UK pairs are.

Calibration is done separately within 20 equal-count bands of Export Potential (Mondrian conformal), because Tweedie errors grow with the prediction. Each band needs at least 1,000 rows, or neighbouring bands are merged. Within each band, the 10th and 90th percentiles of signed residuals become the range's offsets, and the range is clipped at zero. An Opportunity whose Gap Period mean Actual Exports lies inside the range is an Uncertain Opportunity.

## Considered Options

- **UK out-of-fold residuals**: circular, as above.
- **XGBoost quantile objectives**: these need two extra models per fold, can cross, carry no coverage guarantee, sit outside the single monotone Tweedie model (ADR 0002), and give a lower quantile that is almost always 0 on zero-heavy HS6 data.
- **Spread across the five fold models**: four of the five trained on any given UK pair, so their spread mixes one held-out prediction with four in-sample ones and understates uncertainty.
- **Calibrating on other Exporters' 2022–2023 rows**: these are training years for the fold models, so their residuals are optimistic.

## Consequences

- **Exchangeability between the UK and other top-100 Exporters is assumed, not guaranteed.** The notebook reports the coverage achieved on a held-out half of the calibration rows, split by Exporter, overall and per band.
- **The comparison runs over the Gap Period, but calibration uses only 2024.** The 2024 band offsets are applied around the Gap Period mean Export Potential and compared with the Gap Period mean Actual Exports. A three-year mean is less noisy than one year, so the range is wider than strictly needed. This errs towards flagging more Opportunities as uncertain, never fewer.
- **Scenario Effects get no interval.** Conformal prediction covers outcomes, not differences between two predictions, and Event Backtests remain the check on Scenario Effects.
