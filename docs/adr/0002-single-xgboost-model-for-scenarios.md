# One monotone-constrained XGBoost model serves both Opportunities and Scenarios

**Superseded by [ADR 0008](0008-opportunity-ranking-is-the-one-objective.md).** Opportunity ranking is now the one objective and Scenarios are illustrative. The text below is kept as the record of the original decision; the code still follows it until the ranking work lands.

Trade economists would normally estimate Scenario Effects with a PPML structural gravity regression. We use a single gradient-boosted model (XGBoost, Tweedie objective, trained on GPU) for both Export Potential and Scenarios, re-predicting with the policy features flipped. Monotone constraints guarantee that adding a Trade Agreement never lowers Export Potential and that raising a tariff never raises it, and tariff inputs are clipped to the range seen in training. This was chosen over a regression and tree hybrid to keep HS6 training time and project complexity down. The cost is that Scenario Effects are learned associations rather than causal estimates. Event Backtests are what make that trade-off visible.

## Considered Options

- Multiple regression (gravity): explainable, but cannot learn the Product×Country interactions that drive HS6 Opportunities.
- Neural network with entity embeddings: plausible, but harder to explain and has no native monotonicity. Dropped to keep a single model.
- Regression plus trees hybrid: two models to train, reconcile and explain.
