# Opportunity ranking is the one objective; Scenarios are illustrative

Supersedes [ADR 0002](0002-single-xgboost-model-for-scenarios.md).

ADR 0002 gave one monotone-constrained Tweedie XGBoost model two jobs: Export Potential for ranking Opportunities, and Scenario Effects. v2 was built and tuned under that rule, and its results show the cost. From now on the model is chosen for one thing only: **how well Export Potential ranks Markets within a Product for the UK, out-of-fold.** Scenario Effects and Blocked Opportunity values are still shown, as **illustrative** figures, and they no longer shape any modelling choice.

What v2 showed (artefact `20261118T100524Z`):

- **The model was selected on a measure that moved against the ranking.** Early stopping, the feature ablation and the hyperparameter search all chose on Tweedie deviance, with the UK ranking only as a guardrail. On the 2022 validation year the search winner had the lowest UK ranking of the three confirmed configurations (0.528 against 0.531 and 0.536). On 2023, v2 ranks UK Markets at 0.511, below the naive baseline (0.523) and v1 (0.540).
- **The monotone constraint inflates the Scenario Effect it was meant to protect.** In the constraint comparison (three Sectors, one fold, 3,327 UK pairs), the model associates adding a Trade Agreement with +26.6% Export Potential with the constraint and +4.0% without it. The PPML gravity benchmark gives +5.1%, and the EU–Japan Event Backtest observed +2% against the model's +35%. Without the constraint, 39% of pairs move the wrong way, so the constraint turns a noisy association into a one-sided one.

## The decision

- **One selection measure.** Features, sampling, hyperparameters and the form of the model are chosen on the UK out-of-fold within-Product ranking on the validation year. Tweedie deviance and every other metric are reported, not selected on.
- **The model need not be a single XGBoost model.** A blend, a two-part model or another form is allowed if it ranks better. Whatever the form, ADR 0001 still holds: every UK Export Potential comes from a model that held that pair out.
- **Scenarios are illustrative.** A Scenario still changes only policy features and re-predicts. The app and the notebook show each Scenario Effect beside the PPML benchmark and the Event Backtest result, and describe it as an upper-end association. No model is preferred because its Scenario Effects look better.
- **The monotone constraints stay until they are tested against the ranking.** If removing them improves the ranking beyond noise, the ranking wins, and a further ADR records what then serves Scenarios.

## Considered Options

- **Keep ADR 0002.** Every later tuning round would keep trading ranking for deviance and Scenario plausibility, as v2 did.
- **Drop Scenarios and Blocked Opportunities.** They are built, tested and part of the story the notebook tells; an honest "illustrative" label costs less than removing them.
- **Serve Scenario Effects from the PPML benchmark.** It is closer to the observed changes, but it is fitted at Sector level, so it gives no HS6 detail, and it would be a second served model. It stays the benchmark shown beside each Scenario Effect.
- **Select on several measures at once.** v2's rule (deviance, guarded by ranking) was this, and the guardrail passed at −0.0099 against a limit of −0.01 while the ranking fell.

## Consequences

- **The code followed ADR 0002 until the ranking work landed** (spec #65); see *Outcome* below for the model as built.
- **The Scenarios and Blocked Opportunities pages need the illustrative label and the benchmark beside them.** Until then they overstate what the model knows.
- **The selection measure must be fixed before any experiment is read**, and the 2023 test year, already read for v2, is read once more at the end and reported beside v2's figures.
- **Export Potential's size may get worse.** v2's strength is its deviance. A model chosen on ranking may estimate the size of Flows less well, and the write-up reports both.

## Outcome (spec #65, artefact `20261118T185910Z`)

- **The form is a blend.** Export Potential = exp(0.7 · log model + 0.3 · log naive baseline), each floored at $1, with the naive weight chosen on 2022. Every other experiment (a starting point, five configurations, a two-part model, the Zero Cell scope, three Product-detail features) stayed inside the noise floor or below it. The blend beat the model alone on all five folds of 2022 (0.5635 against 0.5206).
- **The monotone constraints stay.** Removing them ranked slightly worse on 2022 (0.5708 against 0.5771, inside the noise), so the open point above is closed without a further ADR. The policy features stay monotone-constrained and tariffs stay clipped to the training range. Scenarios change only policy features and re-predict through the one prediction function. The naive baseline has no policy features, so a Scenario leaves it unchanged, and every Scenario Effect is the model's effect scaled by 0.7 on the log scale.
- **2023, read once:** the UK ranking is 0.555, against 0.523 for the naive baseline, 0.511 for v2 and 0.540 for v1. On this measure the blend beats the naive baseline, the success bar stated in advance. As this ADR expected, size got worse: Tweedie deviance over all Exporters rose from 355.5 (v2) to 433.8. Size-matched Lift is 0.76 (v2 0.79), still below 1. These are associations on one test year.

## Amendment: the Selection Measure is tie-proof (spec #113, ticket #115)

The open review of 2022 (`docs/research/open-review-2022.md`) found that the Selection Measure could be raised by ties. It ranked both sides with average ranks, so flooring Export Potential at $1,000 tied 52% of UK pairs with the tied zeros of Actual Exports and lifted the Blend from 0.5635 to 0.6062 on 2022 while ranking nothing better. That gain is as large as the Blend's whole win over the model alone, and early trees, whose predictions are full of exact ties, could win round-by-round early stopping the same way.

- **Ties in Export Potential are broken in a fixed pseudo-random order**: a stable hash of the Product and the Market (CRC-32 of each, mixed by integer arithmetic), so the measure is the same on every run and in any row order. **Actual Exports keep average ranks** for their ties. Without ties the measure is the ordinary Spearman mean. A tied block broken at random correlates, on average, no better than a randomly ordered one, so tying can only lose information; a part of a ranking already in the order of Actual Exports can never gain by being tied.
- **One function computes it**, `exportspredictor.selection_measure.selection_measure` (per Product: `within_product_correlations`). The experiment loop's selection and round-by-round early stopping, its diagnostics, the evaluation's headline, breakdowns and stability check all call it.
- **The old measure stays as `legacy_selection_measure`, for continuity tables only.** The figures above (0.5635, 0.5206, 0.555 on 2023 and the earlier versions' reads) were made with it, and the pre-registered v1 comparison keeps it. Nothing is chosen on it.
- **What it changes on the saved 2022 five-fold confirmation grid:** the Blend 0.5635 and the model alone 0.5206 are unchanged to four places (their predictions barely tie); the Naive Baseline, whose zeros tie, falls from 0.5332 to 0.5240. Flooring the Blend now lowers it: $100 gives 0.5628, $1,000 gives 0.5431, $3,000 gives 0.5016 and $10,000 gives 0.4248.
- The loop's other past decisions, the Noise Floor and the Win Threshold are re-checked under the fixed measure in a later ticket of spec #113; 2023 is not re-read for this change.
