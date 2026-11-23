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

To be completed by [#148](https://github.com/noahlb1/exportspredictor/issues/148): a Scenario that adds a Trade Agreement lists the precomputed Blocked Opportunities of the Markets it adds one with, and why a preset's Scenario Effect for a pair can differ from that pair's Blocked Opportunity value.
