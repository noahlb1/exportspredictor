# Coding standards

Read in review. These are judgement calls; the rules in CLAUDE.md ("Shape" and "Modelling guardrails") and GLOSSARY.md apply too and are not repeated here.

## Where logic lives

- **`api/` only translates HTTP.** It validates, calls `src/exportspredictor`, and shapes the response. Ranking, sorting, filtering, Scenario wording, money formatting and currency conversion belong in the package, so the notebook gets the same behaviour.
- **One definition per constant or option.** Thresholds, ranking orders and preset names are defined once in the package. The API's schemas derive from that definition, and the app reads it from an endpoint (as `/scenarios/thresholds` and `/opportunities/rank-by` do) rather than copying the literal.
- **A query has one entry point.** A new filter or order on rankings goes into `rankings.rank_opportunities`, not into a second sort in the API.

## Names

- **Money carries its unit**: `_usd` or `_gbp` on every parameter, column and field. A bare `min_gap` hides which.
- **Domain names come from GLOSSARY.md**, including for derived columns (a column holding the Blocked Opportunity value says so). Vocabulary from a source's own format (BACI's or HMRC's "partner") is fine where the code describes that source.

## Leakage

- **Check indirect paths.** Beyond the explicit leave-one-out features, ask whether an aggregate built from other rows (Market demand, income shares, RCA) still contains the held-out UK Flow. Note any such path in ADR 0001 or the notebook rather than leaving it implicit.

## Tests

- **Tests run on the fixture, on the CPU.** A test that needs real data, the GPU or the network is a bug; real-data checks are runs evidenced in the ticket, not tests. The suite stays near a minute and a half.
- **A schema change ships with a contract test** in `tests/test_api.py`, and a new page with its README checklist entry (enforced by `tests/test_readme.py`).
