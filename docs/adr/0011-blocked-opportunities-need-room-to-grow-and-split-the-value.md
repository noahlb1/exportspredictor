# Blocked Opportunities need room to grow, and their value is split into a Tariff part and an Agreement part

Supersedes the decision of [#102](https://github.com/noahlb1/exportspredictor/issues/102) ("presentation only, nothing hidden or re-ranked"). Decided in the investigation and grilling of [#112](https://github.com/noahlb1/exportspredictor/issues/112) and built in spec [#133](https://github.com/noahlb1/exportspredictor/issues/133).

#102 kept the Blocked Opportunity rule as it was and handled awkward rows with wording. The page built on that decision was found hard to read, and the cause was mostly the rule, not the wording. Measured on artefact `20261120T133841Z` (Baseline year 2024, default thresholds: 3,834 Blocked Opportunities, $27.4bn, 62 Markets):

- **623 ($6.65bn) were already exceeded.** Their Baseline-year Actual Exports were above Export Potential with a Trade Agreement, including the top two rows. A list called Blocked Opportunities reads as "exports are being held back here", which these pairs contradict.
- **Most of the value is not the tariff.** 764 rows ($5.9bn, including the top row) have an Applied Tariff of 0, yet every row said its value came from "adding a Trade Agreement and removing the tariff". Setting the tariff to 0 on its own gives $9.4bn of the $27.4bn; the rest is the model's association with having an agreement at all.
- **"A Trade Agreement" is one flag.** In the model it is 1 for any WTO-listed free trade agreement or customs union, so the value is the association with an *average* agreement as the model has seen them, with tariffs fully removed, not with any specific deal.

**Measured before Market Reach.** The figures here come from artefact `20261120T133841Z`, which has no Market Reach ([ADR 0010](0010-market-reach-in-the-model-form.md)). The served value and the $1M floor are in dollars after Market Reach, which moves each Market's dollar level with its multiplier, so Market Reach decides much of the Market mix: the first served form gave 2,169 pairs, $18.45bn and 18 Markets under this rule, with the Falkland Islands fourth because of an unbounded offset, which ADR 0010's amendment fixes. Keeping the value after Market Reach is deliberate: it is the same Scenario Effect on every page.

## The decision

- **Room to grow.** A Blocked Opportunity also needs Baseline-year Actual Exports below Baseline-year Export Potential plus its Blocked Opportunity value (its Export Potential with a Trade Agreement). The value itself is **not capped** at that room. Every other part of the rule stays: no Trade Agreement in force, any Applied Tariff (0%, above 0 or unknown), a value of at least 20% of Baseline Export Potential and at least $1M. A missing Baseline-year Actual Exports counts as 0. On the artefact above this leaves about 3,211 pairs, $20.7bn. One rule, `scenarios.is_blocked`, decides it on the Blocked Opportunities page, the Leaderboard and the Scenarios page.
- **The value is split.** The **Tariff part** is the Scenario Effect of setting a known Applied Tariff to 0 (clipped to the training range, [ADR 0006](0006-tariffs-carried-forward-and-zeroed-under-trade-agreements.md)) with the Trade Agreement flag left off, re-predicted out-of-fold through `prediction.export_potential` with the served Model Form. An unknown tariff stays unknown, so its Tariff part is 0, as is a 0% tariff's. The **Agreement part** is the value minus the Tariff part, so the two always sum to the value. Pairs with a Trade Agreement in force have all three at 0. The same function computes the parts in the pipeline and in the rebuild from saved fold models.
- **The 0% and unknown-tariff rows stay.** The page says their rise rests entirely on the model's association with an average agreement, and never speaks of removing a tariff there.
- **The all-Markets headline total stays**, labelled Illustrative, and follows the new rule.

### Why the Agreement part is the remainder

A separate "flag on, tariff kept" run would price the agreement directly, but that combination is the one ADR 0006 warns the model rarely saw: pairs with an agreement in training almost always have low tariffs. The tariff-only input (tariff 0, no agreement) is common in training, so the Tariff part is the well-supported run, and defining the Agreement part as what is left keeps the parts summing exactly to the served value. The #112 flag-only run ($16.1bn) is a rough cross-check, not a figure the app shows.

### Why room to grow, uncapped

Room to grow removes the pairs that contradict the page's reading, with one comparison an Advisor can check on the row. Capping the value at the room (the smaller of the value and the room) would make the value depend on Actual Exports, which no other Scenario Effect does, and the $1M floor would then be applied to a different quantity on this page than the Scenarios page reports for the same pair. Uncapped, the value is the same Scenario Effect everywhere.

## Considered Options

- **Keep #102's presentation-only answer.** It is the page found confusing; wording alone cannot stop an exceeded pair from heading a list of things held back.
- **A tariff-led definition**: a Blocked Opportunity needs a known Applied Tariff above 0, and the thresholds apply to the Tariff part (about 1,045 pairs), with the agreement association moved to the Market level. Rejected. A Trade Agreement is more than tariffs (standards, customs procedures, rules of origin, services), and the model's association with it is real information about the pair's Market even where the tariff is 0. More importantly, the 0% and unknown rows are exactly where better data would help: **if Product-level agreement provision data were added, these pair rows would become informative**, whereas a single agreement-depth index would mostly sharpen the Market-level figure. Dropping the rows now would hide the place that improvement would show. That idea is recorded as [#138](https://github.com/noahlb1/exportspredictor/issues/138).
- **Tariff above 0 and room to grow, value unsplit** (2,607 pairs, $17.0bn). It shortens the list but still reports the agreement association as if it came from the tariff.
- **Rank by the rise as a share** instead of dollars. It answers a different question (which pairs grow most relative to their size) and was left for later; the order by Tariff part gives the tariff-led view without changing the definition.
- **Change the thresholds.** The $1M floor already does most of the selecting (it removes 673,922 of 678,216 pairs without an agreement); tuning it was out of scope.

## Consequences

- **Every figure stays Illustrative and associational** ([ADR 0008](0008-opportunity-ranking-is-the-one-objective.md)): "the model associates an agreement with +X%", never "an agreement causes +X%". The Tariff part and the Agreement part are associations too, not a causal decomposition.
- **The artefact gains a Tariff part column** in its Baseline table, written as a new version from the newest artefact's saved fold models. Nothing is retrained and no test year is read, so the Test-year Ledger ([ADR 0009](0009-test-year-read-once-per-frozen-version.md)) is untouched.
- **The API drops `blocked_opportunity_uplift`**, the percentage uplift shown instead of a dollar level where Actual Exports exceeded Export Potential. Room to grow guarantees Export Potential with a Trade Agreement is above Actual Exports, so the dollar level is always shown.
- **The notebook and model card change their headline**: the count, total and Markets above, the split of the value, and how much the Agreement part varies between Products within a Market.
- **A Market's total on the Blocked Opportunities page still differs from the Scenario Effect** of an agreement with that Market on the Scenarios page: the first sums only the listed pairs, the second covers every pair. The page says so and links to the Scenario.
