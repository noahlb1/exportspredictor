# Decision log: spec #142 (Scenario Builder)

## orchestrator: integration branch and ticket branches
- Options: PR-based close; local integration branch with issues closed by comment
- Chose: local branch `spec-142`, ticket branches `t142/<n>-<slug>`, worktrees under C:\CodingProjectsC\ep-wt\t142-<n>; issues closed with a comment.
- Why: matches spec #133's run (issue tracker closes work by comment and close, not PR).

## orchestrator: exploration subagent
- Options: run one first; let implementers explore
- Chose: no exploration subagent.
- Why: the tickets name the files and seams precisely; a shared note would mostly duplicate them.

## orchestrator: one-shot tickets
- Options: treat a ticket as one-shot (review first); none
- Chose: none is one-shot.
- Why: no ticket reads the 2023 test year, builds the reported artefact or spends a once-only budget.

## 147: which seams to test
- Options: confirm seams with the user (tdd skill); take the spec's Testing Decisions
- Chose: the API over HTTP (`tests/test_api_scenarios.py`) and the page through AppTest (`tests/test_app_scenarios.py`), as the spec names.
- Why: unattended run; the spec already agrees these seams.

## 147: what "the set rate" is compared against, and how
- Options: compare the Baseline tariff with the requested `rate_pct`; compare it with the rate actually applied (the requested rate clipped to the training range), both read from the same `ln(1 + t)` column
- Chose: the applied rate, as `baseline_tariff_pct` against `scenario_tariff_pct` over the changed pairs (`ScenarioResult.tariff_directions`).
- Why: it is exact (no float mismatch between a float32 Baseline 10% and a float64 10.0) and agrees with which pairs count as changed; the two only differ for a rate beyond the training range, where the clipped-tariff note already speaks.

## 147: wording and placement of the page line
- Options: mention the unknown-tariff remainder; only raised and lowered
- Chose: "This sets N pairs to R%: it raises A and lowers B.", as markdown right after the clipped-tariff note (before the caption #149 may cut); rate printed with `:g` (30%, 12.5%).
- Why: it is the ticket's own example; pairs with unknown tariffs are the remainder, never described as "removing" a tariff (ADR 0006, ADR 0011).
- Detail: N is the sum of `markets[].pairs` (every changed pair, unknown tariffs included). README release checklist gains one line for it.

## 146: how "changes nothing" is decided for the preset list
- Options: a new rule on bloc flags; `apply(baseline_features, preset, tariff_range)` returning no rows; running each preset in full
- Chose: `scenarios.offered_presets` keeps a preset unless `apply` (with the artefact's tariff range) returns no rows, the same rows `run` re-predicts and whose emptiness is `ScenarioResult.changes_nothing`.
- Why: one rule for the list and a run, without paying for predictions; a test runs every offered preset and asserts `changes_nothing` is false.
- Detail: the API caches the list per artefact version on `app.state.offered_presets`; `/scenarios/presets` now needs a loaded artefact (503 otherwise), like `/markets`. Every preset name is still accepted by `POST /scenarios` (Leave still answers "changes nothing").

## 146: fixture Baseline offers only CPTPP
- Options: change the fixture world so EU presets change something; accept that the fixture (no EU member Market, UK in no bloc) offers only CPTPP and test the others on a variant
- Chose: accept it; a saved variant with India made an EU member (and the UK too, for Leave) shows all four come back.
- Why: changing the fixture world reaches every other test; the variant tests the "returns by itself" criterion directly.
- Detail: tests/test_app_scenarios.py's stale-controls test switched from choosing EU re-join to switching the Scenario kind, since EU re-join is no longer offered on the fixture.

## 146: EU re-join wording
- Options: compute the description from the Baseline; a general wording true in any Baseline
- Chose: general wording: "The Customs Union and EU membership with every EU member, plus a Trade Agreement with any member that has none in the Baseline year; since 2021 the UK already has a Trade Agreement with each."
- Why: keeps `PresetOut` static (no contract change) and stays true in a pre-2021 Baseline, where the preset still adds agreements.

## 143: what to name the shared panel function, and how to test a pure move
- Options: `panel.blocked_opportunity(row)` / `panel.before_after_panel(row)`; rely on the existing page tests only / add one test at the panel module's seam.
- Chose: `app.panel.blocked_opportunity(row)`, plus one AppTest (`from_function`) that draws it alone against the Blocked-world API.
- Why: the name says what it explains (`before_after` already names the chart), and the new test pins the public function #148's Scenario Builder section will call, while the existing page tests stay unchanged.
- Detail: app/panel.py, app/pages/blocked_opportunities.py (its `_explain` removed), tests/test_app_blocked.py (one test appended).

## 145: Where does "the Markets a Scenario accepts" live, so `/markets` and validation cannot drift?
- Options: compute the set in the API endpoint; add a public `scenarios.accepted_markets(baseline_features)` used by both `Scenario` validation (`_apply`) and `GET /markets`.
- Chose: `exportspredictor.scenarios.accepted_markets`, called by `_apply`'s validation and by `/markets`.
- Why: one definition (UK feature rows in the Baseline year) read by both callers, as the package is the single source of truth.

## 145: Add `scenario` to `MarketOut`, or to a `/markets`-only model?
- Options: add the field to `MarketOut` (embedded in the market views and pair detail too); a subclass `MarketListed(MarketOut)` for `/markets` only.
- Chose: `MarketListed` with `scenario: bool`, used only by `GET /markets`.
- Why: the spec adds the flag to `/markets`; adding it to `MarketOut` would change every response embedding a Market and broke 43 tests.

## 145: Should the `?add=` prefill link also drop Markets a Scenario does not accept?
- Options: filter only the pickers; filter the one Market-name map the page builds, which feeds the pickers and the prefill.
- Chose: filter the page's Market-name map, so the link leaves out such a Market as it does an unknown code.
- Why: a prefilled Market the picker cannot offer would otherwise produce the "Unknown Markets" error the ticket removes.

## 144: names of the fixed threshold constants
- Options: keep DEFAULT_MIN_SHARE / DEFAULT_MIN_USD; rename to say they are the fixed Blocked Opportunity thresholds
- Chose: renamed to `exportspredictor.scenarios.BLOCKED_MIN_SHARE` (0.2) and `BLOCKED_MIN_USD` (1_000_000.0), with `BLOCKED_MIN_SHARE_TEXT` / `BLOCKED_MIN_USD_TEXT` ("20%", "$1M") for prose.
- Why: "default" implied other values could be chosen; the orchestrator asked for clear public names #148 can use.
- Detail: callers updated in api/schemas.py, governance.py, pipeline.py, rankings.py. `is_blocked`, `with_blocked_opportunity_flag` and `ScenarioResult.blocked_opportunities` keep their `min_share` / `min_usd` parameters (defaulting to the constants) so the package can still test the rule; only the API and pages lost them.

## 144: how threshold fields in POST /scenarios are refused
- Options: keep fields and validate they equal the fixed values; drop them from the request model and rely on `extra="forbid"`
- Chose: `ScenarioRequest` no longer inherits `Thresholds`; any threshold field is an unknown field and answers 422.
- Why: simplest, and no client can send even the fixed values, so there is never a second definition in a request.

## 144: tests that needed Scenario Blocked Opportunities at 0% / $0
- Options: keep a way to set thresholds for tests; move the room-to-grow check to the package seam and rewrite page tests against the drill-down table
- Chose: the room-to-grow check for a Scenario moved to tests/test_scenarios.py (package, `with_blocked_opportunity_flag` at thresholds 0); page tests that clicked or read the Scenario's Blocked Opportunity table now use the drill-down table, and one test asserts the fixture shows 0 Blocked Opportunities and "No pairs to list" at the fixed rule.
- Why: the fixture's Scenario Effects are all under $1M (also in the Blocked world, whose fold models are unscaled), so no fixture Scenario yields a Blocked Opportunity at 20% / $1M; #148 replaces the section with precomputed rows and its own parity tests.

## 144: wording of the Blocked Opportunities page's minimum-value filter
- Options: keep "Minimum Blocked Opportunity value"; "Show only pairs worth at least… (USD, nominal)" with help saying it only narrows the list
- Chose: the latter; the help says the rule itself is fixed and stated in the caption, without repeating the figures (the sidebar renders before the read that carries `thresholds`, and the app does not import the package).
- Why: the spec's wording; it keeps the filter from being read as a threshold.

## 144: README checklist lines on thresholds
- Options: leave for #150; update the two lines that describe adjustable thresholds
- Chose: updated README.md's Blocked Opportunities and Scenarios checklist lines to the fixed rule; left every "Scenarios page" name for #150.
- Why: those lines described behaviour this ticket removes.

- Merger, #147: merged t142/147-raised-lowered into spec-142 as 3f91a19 with no conflicts; no decisions needed. ruff, pyright and pytest (1422 passed) green.

- Merger, #143 and #146: merged t142/143-panel into spec-142 as cec488e (no conflicts; ruff, pyright green, pytest 1423 passed), then t142/146-presets as c5b67b5 (no conflicts, the branches touch disjoint files; ruff, pyright green, pytest 1427 passed). No decisions needed, no fix commit.

## Merger: tickets #144 and #145 into spec-142

- #144 merged cleanly (bfbf0ee), but `test_a_line_says_when_the_controls_differ_from_the_result_shown` failed: #144 had made its last step run the `eu_rejoin` preset, which #146 no longer offers on the fixture (only CPTPP). Replaced that step with a Custom Trade Agreements run adding India, which keeps the test's intent (change controls, run, no stale line). Also dropped the removed `/scenarios/thresholds` from CODING_STANDARDS' example. Commit a68337c.
- #145 conflicted only in `tests/test_api_scenarios.py`: kept #144's `test_the_thresholds_read_is_gone` and #145's `/markets` `scenario` flag tests, and dropped #145's `test_default_thresholds_are_served` (the endpoint is gone under #144, ADR 0012). Merge 4d93b03. No #145 test sends thresholds or uses threshold inputs.
## 148: which seams to test
- Options: confirm seams with the user (tdd skill); take the spec's Testing Decisions
- Chose: the API over HTTP (tests/test_api_scenarios.py) and the page through AppTest (tests/test_app_scenarios.py), on the Blocked world (the plain fixture has no Blocked Opportunity at 20% / $1M).
- Why: unattended run; the spec names these seams, and the Blocked world is where India has Blocked Opportunities (whisky, jewellery).

## 148: where "the Scenario adds an agreement" is decided
- Options: in the API endpoint; on the package's ScenarioResult
- Chose: `ScenarioResult.added_agreements` (sorted ISO3s whose pairs `apply` turns the Trade Agreement flag on for, from off in the Baseline features) and its `adds_agreement` property, set in `scenarios.run`.
- Why: the package is the single source of truth; it reads the same changed rows the Scenario re-predicts, so a removal, tariff change, Customs Union only or Leave can never count.

## 148: how the Scenario's rows are read so parity holds by construction
- Options: filter the Scenario's own pairs; call `rankings.blocked_opportunities` with a new `markets` filter
- Chose: `rankings.blocked_opportunities(artefacts, markets=added_agreements, hide=<tiers with hidden_by_default>, limit=request.limit)`, served as `models.BlockedOpportunity` rows, the code path `/opportunities/blocked` uses.
- Why: same function, same rows, same summary; `blocked_count` is its summary count before `limit`.
- Detail: `ScenarioResponse.blocked_opportunities: list[BlockedOpportunity] | None`, `blocked_count: int | None`, new `adds_agreement: bool`.

## 148: the summary sentence when no agreement is added
- Options: leave it counting "0 Blocked Opportunities"; drop the clause when `blocked` is None
- Chose: `ScenarioResult.summary(blocked=None)` drops the clause; #149 removes the figures entirely.
- Why: a tariff or Customs Union Scenario must never read "with 0 Blocked Opportunities" (user story 13).

## 148: the link to the default-hidden Caveat Tiers
- Options: a plain link to the Blocked Opportunities page; a link that opens it with those tiers shown
- Chose: "See the hidden pairs on the Blocked Opportunities page" carrying `<hide_param>=false` for each tier with `hidden_by_default`.
- Why: the Blocked Opportunities page hides them by default too, so a plain link would not show them; the filters module documents this link form.

## 148: the drill-down pair table after the section moved
- Options: keep `pairs_frame`'s Market / Actual Exports options; drop them
- Chose: dropped `with_market` / `with_exports` and the "Actual Exports" column config and format, which only the old Blocked Opportunity table used.
- Why: dead options otherwise; the drill-down is unchanged.
- Detail: page tests that counted every page link now count only a pair box's Leaderboard links (`_pair_links`), since the section and the neutral line add links to the Blocked Opportunities page. `blocked_products_frame(rows, with_market=True)` and a Market column config in `blocked_products_columns` serve the section.

## 148: two remaining uses of the old rule (posted on #142, not built)
- Options: change them here; post with a recommendation and stop that item
- Chose: posted https://github.com/noahlb1/exportspredictor/issues/142#issuecomment-7103348435 and left both as they are.
- Why: (1) `PairEffect.blocked_opportunity` on `top_pairs` is still `is_blocked` on the Scenario's own effect; changing its meaning is an API contract change beyond the spec. (2) Notebook section 8's preset table prints 4,734 / 2,223 / 118 "Blocked Opportunities" from `ScenarioResult.blocked_opportunities`; changing it alters a printed figure and needs a notebook re-run.

## 148: README release checklist
- Options: leave for #150; update the Scenarios lines this ticket changes
- Chose: rewrote the Blocked Opportunities line (add the USA: same count and figures as that page, caption, panel), added the neutral-line check, and dropped the old Blocked table's Actual Exports and Caveats mentions; left "Scenarios" names for #150.
- Why: those lines described behaviour this ticket removes.

## orchestrator: merge #148 without a merger subagent
- Options: merger subagent re-running the suite; merge inline
- Chose: merged t142/148-one-source into spec-142 inline.
- Why: the branch already contained the spec-142 tip, so the merged tree equals the branch tree on which the suite, ruff and pyright passed (1,459).

## 149: seams under test
- Options: ask; take the spec's Testing Decisions
- Chose: the API over HTTP for the summary sentence, and the Scenarios page through AppTest for the boxes, expander, captions and order.
- Why: the spec names both seams; unattended runs take the recommended option.

## 149: the summary sentence without figures
- Options: drop only effect/share/count and keep the Market count and year; drop every figure
- Chose: "The model associates this Scenario with a rise | a fall | no net change in UK Export Potential in the Markets it changes, holding everything else at the Baseline: a learned association, not a forecast." No digit at all; `ScenarioResult.summary(*, year)` loses its `blocked` parameter (the year is still named in the two "changes nothing" sentences).
- Why: each figure (including the Market count, which Change per Market carries) is stated once beside it; direction is kept so the sentence still frames the result, and the "holding everything else at the Baseline" clause from the removed caption moves into it.
- Detail: the package test of thousands separators in the count was removed; the API test now asserts no digit, $, % or £ for an addition, CPTPP and a tariff Scenario. The notebook does not call `ScenarioResult.summary`, so nothing it prints changes and it needs no re-run.

## 149: one Illustrative expander for both pages
- Options: wrap the call on the Scenarios page only; move the expander into `app.ui.scenario_note`
- Chose: `app.ui.scenario_note` now renders inside the expander labelled "<Illustrative>: what the Blocked Opportunity value is read against", and both pages call it.
- Why: the ticket asks for the same expander with the same label; one function keeps the two from drifting. The label is kept word for word, though on the Scenarios page it qualifies the Scenario Effect too.

## 149: the metrics' width once the note left their side
- Options: spread the metrics across the full width; keep them in the left half
- Chose: kept them in the left half (`st.columns([1, 1])[0]`), the expander full width under them.
- Why: two or three metrics spread over the full width read as unrelated; this is the layout Advisors already had.

## 149: README release checklist
- Options: leave for #150; update the Scenarios lines this ticket changes
- Chose: reworded five lines (summary without figures, clickable-tables caption, "changes nothing" as info and invalid as warning, the clipped-tariff info box, the Illustrative expander); left "Scenarios" names for #150.
- Why: those lines described behaviour this ticket changes.
## orchestrator: merge #149 inline
- Options: merger subagent; merge inline
- Chose: merged inline.
- Why: branch parent was the spec-142 tip, so the merged tree equals the tested branch tree (1,464 passed).

## 150: How to check "no tracked file says 'Scenarios page'"
- Options: a one-off grep at review; a test.
- Chose: a test, `tests/test_page_names.py`, scanning every tracked text file except `docs/runs/` for the old name in any case, plural or split across a line break or a `#` comment wrap.
- Why: the rename touches files every ticket edits, and a test stops the old name coming back.

## 150: The menu label and the Blocked Opportunities page's link label
- Options: link label "Scenario: a Trade Agreement with X" kept; "Scenario Builder: a Trade Agreement with X".
- Chose: "Scenario Builder: a Trade Agreement with X", with url_path `Scenario_Builder`; the file stays `app/pages/scenarios.py`.
- Why: the ticket says the link names the Scenario Builder.

## 150: Mentions outside the acceptance phrase
- Options: change only "the Scenarios page"; also change "Scenarios and Blocked Opportunities pages" phrasings.
- Chose: changed the notebook's closing prose, `wording.py`'s comment, a test comment, the README menu item and section heading; left ADR 0008 ("The Scenarios and Blocked Opportunities pages need the illustrative label") and the spec section name `spec #103, "Scenarios"` as they are.
- Why: ADR 0008 records a past decision in the words of its time, and #103's section really is called "Scenarios"; the rest describes the current app.
- Detail: ADR 0012's opening now reads "Decided in the review of the page now called the Scenario Builder"; GLOSSARY.md's new entry avoids "what-if tool, scenario simulator"; the README checklist gained an item for the menu, `/Scenario_Builder`, the subtitle and the India link.

## orchestrator: merge #150 inline
- Options: merger subagent; merge inline
- Chose: merged inline.
- Why: branch parent was the spec-142 tip, so the merged tree equals the tested branch tree (1,466 passed).

## orchestrator: which final-review findings to fix
- Options: fix all; fix all except those awaiting the owner or fixed by the spec's contract
- Chose: fix wording-in-package, shared rule text, shared Blocked Opportunity table+panel block, expander label, Story 6 rule prose, raised/lowered line under a clipped rate, obsolete old-rule API test, docstring split. Leave: top_pairs `blocked_opportunity` and notebook section 8 (posted on #142, awaiting owner); `ScenarioResult.blocked_opportunities` and threshold params (still used by the notebook / `is_blocked` tests the spec keeps); `scenario` and `min_effect_*` field names (API contract named by the spec); page-path literals (minor).
- Why: the spec forbids building items posted for the owner, and the field names are part of its written contract.

## review-fixes: where the Scenario wording lives and how the app gets it
- Options: wording functions in `exportspredictor.wording`; methods on `ScenarioResult` in `exportspredictor.scenarios`, served as new `POST /scenarios` fields; keep in app
- Chose: `ScenarioResult.tariff_directions_text()` and `ScenarioResult.blocked_note(hidden, shown, count)` in `exportspredictor.scenarios`, served as `tariff_directions` and `blocked_note`; the rule sentence `scenarios.BLOCKED_RULE` is served in `/labels` as `blocked_rule`.
- Why: the app reaches the package only over HTTP, the result already holds what the wording needs, and the API knows the hidden tiers and the listed count; `wording.py` cannot import the thresholds without pulling in the model code.
- Detail: `blocked_note` lazily imports `illustrative.ILLUSTRATIVE` (illustrative imports scenarios transitively), as `ppml.py` already does for scenarios.

## review-fixes: one wording of the Blocked Opportunity rule
- Options: year-specific sentence ("2024 Export Potential"); year-free sentence from `BLOCKED_MIN_*_TEXT`
- Chose: one year-free `BLOCKED_RULE` ("... at least 20% of its Baseline Export Potential and at least $1M, and it has room to grow ..."), reused by the Blocked Opportunities page caption, the Scenario Builder's caption and neutral line, `rankings.BLOCKED.description` and both model-card sentences.
- Why: the pages already name the Baseline year beside it, and Story 6 words the thresholds as "20% of Baseline Export Potential and $1M"; the pages now print "$1M" where they printed "$1.0m" (tests updated).

## review-fixes: shared Blocked Opportunities table and panel
- Options: in `app/tables`; in `app/panel`
- Chose: `app.panel.blocked_opportunities(rows, body, key, remembered, noun, with_market)`, called by both pages.
- Why: it draws the panel, which lives in `app.panel`, and `app.panel` already imports `app.tables`.
- Detail: the Scenario Builder now remembers its chosen Blocked Opportunity as (HS6, ISO3), as the Blocked Opportunities page does (was `{"hs6", "iso3"}`); the panel box key is `<table key>_panel_<hs6>_<iso3>`.

## review-fixes: the Illustrative expander's label
- Options: pass the figure name per page; one general label
- Chose: "Illustrative: what these figures are read against" on both pages.
- Why: true on both pages with no per-page argument, and the two pages' expanders stay identical (an existing test compares them).

## review-fixes: the raised/lowered line under a clipped rate
- Options: say the counts are against the edge; name the rate the model reads
- Chose: both: "This sets N pairs to 5000%, which the model reads as 150.0%, the edge of the training range: against that rate it raises R and lowers L."
- Why: the counts are computed against the clipped rate, so naming it makes the line true; the rate read is taken from the changed pairs' Scenario tariff, one rate for every pair.

## review-fixes: the obsolete old-rule API test
- Options: rewrite to the one-source rule; delete
- Chose: deleted `test_a_scenarios_blocked_opportunities_are_its_flagged_pairs`.
- Why: the parity test and the mixed/two-Market tests already cover the one-source rule.
## orchestrator: review fixes add response fields
- Options: revert to page-side wording; keep additive fields
- Chose: kept `tariff_directions` and `blocked_note` on POST /scenarios and `blocked_rule` in /labels.
- Why: CODING_STANDARDS puts Scenario wording in the package and the app reaches it only over HTTP; the fields are additive and easy to drop. Flagged for the owner.
