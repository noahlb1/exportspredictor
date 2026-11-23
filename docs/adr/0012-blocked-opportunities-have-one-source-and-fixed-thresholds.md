# Blocked Opportunities have one source and fixed thresholds

Decided in the review of the Scenarios page ([#139](https://github.com/noahlb1/exportspredictor/issues/139), finding F2) and built in spec [#142](https://github.com/noahlb1/exportspredictor/issues/142). It keeps [ADR 0011](0011-blocked-opportunities-need-room-to-grow-and-split-the-value.md) in force: room to grow and the split of the value into a Tariff part and an Agreement part are unchanged.

## Fixed thresholds

A pair is a Blocked Opportunity when its Blocked Opportunity value is at least **20% of its Baseline Export Potential** and at least **$1M**, and it has room to grow (ADR 0011). The two thresholds are constants of `exportspredictor.scenarios` (`BLOCKED_MIN_SHARE`, `BLOCKED_MIN_USD`), the values the artefact metadata already records, and nobody can change them:

- the Scenarios page has no threshold inputs;
- `POST /scenarios` refuses `min_effect_share` and `min_effect_usd` (422), and `GET /scenarios/thresholds` is gone;
- every response that applies the rule still states it in `thresholds`, so prose can say "20% and $1M".

Until now an Advisor could change the thresholds on the Scenarios page, so the same pair could be a Blocked Opportunity on one page and not on the other. At 0% / $0, adding a Trade Agreement with the USA gave 4,381 Blocked Opportunities; the Blocked Opportunities page, at 20% / $1M, gave 1,340. One rule everywhere means a pair an Advisor finds on one page is a Blocked Opportunity on the other.

### Considered options

- **Keep adjustable thresholds.** Rejected: two pages, two definitions, and an Advisor has to understand and tune a definition before running a Scenario.
- **No thresholds** (every pair without a Trade Agreement whose value is above 0 and has room to grow). Rejected, on the review's counts (artefact `20261122T223312Z`, Baseline 2024):

  | Thresholds | Pairs | Value |
  |---|---|---|
  | Fixed (20% / $1M) | 2,128 | $18.52B |
  | None | 599,351 | $22.05B, median value $0.05 |

  Without thresholds the list is almost all noise: the median pair is worth five cents. The fixed thresholds keep 2,128 pairs, under 0.4% of them, and those carry **84%** of the value.
- **Fix the thresholds at 20% / $1M.** Chosen. They are the values every Blocked Opportunity figure in the artefact, the notebook and the Blocked Opportunities page already used, so no published figure changes.

The Blocked Opportunities page keeps its minimum-value filter, worded as a filter on the list ("show only pairs worth at least…"): it narrows the list shown and does not change which pairs are Blocked Opportunities.

## One source

Built in [#148](https://github.com/noahlb1/exportspredictor/issues/148), from the review's findings F1, F3, F4 and F12.

A Blocked Opportunity is a pair with no Trade Agreement in force, measured by adding one (ADR 0011). Every Blocked Opportunity figure is read from one place: the Blocked Opportunity values the artefact precomputes on its Baseline, ranked by `rankings.blocked_opportunities`, which `/opportunities/blocked` serves.

A Scenario **adds an agreement** for a Market when it turns the Trade Agreement flag on for that Market's pairs and the Market has no Trade Agreement in force in the Baseline year (`ScenarioResult.added_agreements`). Custom additions and CPTPP can add one, and so could EU re-join, though in the 2024 Baseline every EU member already has one. Removals, tariff changes, EU Customs Union only and Leave never add one.

- When a Scenario adds an agreement, `POST /scenarios` answers `adds_agreement: true` and lists the Blocked Opportunities of those Markets: the rows `/opportunities/blocked` serves, in its row shape, with the Caveat Tiers the app hides by default hidden, ranked by Blocked Opportunity value and limited by `limit`; `blocked_count` counts them all. The Scenario Builder shows them in the Blocked Opportunities page's Products table and before/after panel.
- Otherwise `adds_agreement` is false and both blocked fields are `null`, never an empty list and 0, and the page shows one neutral line: Blocked Opportunities are measured by adding a Trade Agreement, and this Scenario adds none.

### Why a Scenario lists figures that are not its own Scenario Effects

Until now the Scenarios page applied `is_blocked` to whatever Scenario it ran, so it called pairs Blocked Opportunities that the glossary would not: 4,734 for EU re-join, though every EU Market already has a UK Trade Agreement; 2,223 for EU Customs Union only, which adds none; and 10 for a 10% tariff on the USA, pairs whose tariff that Scenario lowers. Restricting the rule to Scenarios that add an agreement, and reading the rows from the precomputed table, means a pair is a Blocked Opportunity on the Scenario Builder exactly when it is one on the Blocked Opportunities page, with the same value, Tariff part and Agreement part. A test at the API checks this for India.

For a custom addition nothing changes in the figures: the review found the same count, and a $0 maximum difference over the top 100 pairs, for the USA, China, India, the UAE, Saudi Arabia and Indonesia. A custom addition sets exactly what the Blocked Opportunity value sets, the Trade Agreement flag and a known Applied Tariff to 0, so the two are the same re-prediction.

### Why a preset's Scenario Effect for a pair can differ from its value

A preset changes more than the agreement flag. CPTPP also sets CPTPP membership, and EU re-join sets the Customs Union and EU membership. So a preset's Scenario Effect for a pair can differ from that pair's Blocked Opportunity value, which adds the agreement alone. The Scenario Builder lists the value, not the preset's own effect, and its caption says why in one sentence.

A Market's Scenario Effect and its Blocked Opportunity total differ for a second reason, which holds for custom additions too: the Scenario Effect covers every pair the Scenario changes, while the total sums only the pairs that clear the fixed rule. Adding the USA is associated with +$18.2bn, against a Blocked Opportunity total of $13.1bn over 1,340 pairs. Both are Illustrative associations, not forecasts (ADR 0008).

### Considered options

- **Keep applying `is_blocked` to each Scenario's own Scenario Effects.** Rejected: it reports Blocked Opportunities for Scenarios that add no Trade Agreement, and a preset's figures for a pair disagree with the Blocked Opportunities page.
- **List a preset's own Scenario Effects only for the Markets it adds an agreement with.** Rejected: it fixes the first problem but not the second, and keeps two sources for one figure.
- **Read the precomputed Blocked Opportunities of the Markets the Scenario adds an agreement with.** Chosen: one source, parity by construction, and no extra prediction.
