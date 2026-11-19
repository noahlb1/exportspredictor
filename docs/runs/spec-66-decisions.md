# Decision log: spec #66 (Notebook and app for the marker)

Integration branch: `spec-66-marker` (based on main 0be8997).

## orchestrator: Open a draft PR for the integration branch?
- Options: open a draft PR; close tickets by comment + close as the issue tracker describes.
- Chose: no PR; close each ticket with a comment and report the integration branch.
- Why: docs/agents/issue-tracker.md closes work by comment + `gh issue close`, not PRs, and spec #65 was merged locally. Merging to main is left to the user.

## orchestrator: Is #83 (derive a real artefact with the PPML benchmark) a one-shot ticket needing a pre-run code review?
- Options: review the integration branch first; treat it as redoable.
- Chose: no pre-run review for #83; review before #92 only.
- Why: #83 runs first, when the integration branch has no changes to review, and its real run reads no held-out data and can be redone by deriving another version (~30 min CPU). #92's committed real-data run is the reported output, so it gets the review gate.

## 82: Which tests count as wording tests, and what replaces each?
- Options: delete every test that reads prose; reduce each to the heading, field or number it guarded; leave short glossary terms and API/artefact-served strings alone.
- Chose: reduce to structure where a real thing was guarded, delete only where a behaviour test already covers it. Kept: single glossary terms and guardrail words ("associates", "tendency", "provisional", "Gap Period"), headings, limitation titles (API fields), and strings read from the artefact or the API (verdict text, Event Backtest caveat, headline text). Out of scope (not notebook/model card/README/app captions): API and package tests of served strings (test_api.py, test_evaluation.py, test_app_why.py factor labels) and the Uncertain Opportunity caption check, which the badge-removal ticket deletes with its feature.
- Why: the ticket asks that prose can be rewritten without breaking the build while behaviour tests stay; a term or a served value survives a rewrite, a sentence does not.
- Deleted or rewritten, one line each:
  - tests/test_notebook.py::test_the_write_up_covers_governance: rewritten; topics now found in the Governance section's `###` subheadings (Model card, Data lineage, Bias review, Regulation, Drift), not in body text ("EU AI Act", "HMRC").
  - tests/test_notebook.py::test_the_training_section_explains_the_gap_period_and_why_2020_and_2021_are_left_out: replaced by test_the_write_up_names_the_gap_period; it pinned section "## 5.", a heading and the sentence "2020 and 2021 are left out", all of which the reorder changes. The 2020/2021 reason lives in `gap_period`'s docstring and the glossary.
  - tests/test_notebook.py::test_notebook_runs_in_fixture_mode_and_saves_artefacts: the printed coverage sentence is reduced to the coverage percentage read from the saved fixture artefact.
  - tests/test_readme.py::test_checklist_checks_values_are_in_usd_with_no_pounds: the quoted caption sentence is reduced to "USD" in the checklist; the no-£ check stays.
  - tests/test_readme.py::test_rebuild_section_says_the_pipeline_trains_and_the_notebook_only_reads: renamed test_rebuild_section_gives_the_pipeline_command; keeps the command, drops the phrases "reads the newest artefact" / "the notebook, which trains". That the notebook only reads is the spec's notebook test's job.
  - tests/test_governance.py::test_model_card_summarises_the_v1_comparison_and_the_gain_ladder: dropped the phrase "decides nothing"; the heading and the artefact's verdict text stay.
  - tests/test_governance.py::test_a_model_card_without_a_v1_comparison_says_so: "not been run" replaced by the section heading still rendering (degrades, never hides) and the field being None.
  - tests/test_governance.py::test_model_card_does_not_call_the_applied_tariff_a_missing_input: renamed test_model_card_names_the_year_the_applied_tariff_is_carried_forward_from; checks "Applied Tariff" and `tariffs.LAST_TRAINS_YEAR` instead of a sentence and its negation.
  - tests/test_governance.py::test_model_card_shows_the_test_year_beside_earlier_versions...: the phrase "beats the naive baseline" replaced by the report's own earlier-versions text rendered in the section; the numbers stay.
  - tests/test_app.py::test_page_shows_baseline_year_banner: the sentence "Export Gaps are 2022–2024 averages" replaced by the API's /health baseline_year, gap_period_label and artefact_version appearing in the banner.
  - tests/test_app.py::test_market_page_shows_the_uk_trade_agreement: dropped the markdown phrase "no UK Trade Agreement"; the table's "UK Trade Agreement in force" = No check stays.
  - tests/test_app_blocked.py::test_every_caption_uses_associational_language: dropped the banner phrase "not causes"; the causal-word scan and "associate" check stay.
  - tests/test_app_blocked.py::test_the_caption_says_the_value_removes_the_tariff_with_the_agreement: deleted; the behaviour is tested in tests/test_scenarios.py::test_blocked_values_remove_the_tariff_with_the_agreement.
  - tests/test_app_evaluation.py::test_page_states_the_test_year_and_how_the_models_were_trained: renamed test_page_states_the_test_year; checks the report's test_year, drops "never" and "every non-UK row".
  - tests/test_app_evaluation.py::test_page_states_the_non_uk_training_sample: "30% sample of non-UK rows" reduced to the rate "30%".
  - tests/test_app_evaluation.py::test_page_shows_size_matched_lift_next_to_raw_lift_and_explains_the_difference: "mostly reflects pair size" replaced by the API's growth threshold in dollars; the literal "2021–2022" replaced by the lift's gap_years.
  - tests/test_app_scenarios.py::test_a_scenario_that_changes_nothing_says_so: "changes nothing" replaced by a warning shown and no table.
  - tests/test_app_scenarios.py::test_the_tariff_mode_runs_and_says_when_the_effect_flattens_beyond_the_training_range: dropped "flattens"; the clip warning and its pair count stay.

## 83: Which seams to test the saved PPML benchmark at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions as the agreed seams.
- Chose: the spec's seams on the fixture world: the package command `ppml.add_benchmark` through `artefacts.load` (tests/test_ppml.py), the artefact round trip (tests/test_artefacts.py), the pipeline's opt-in parts (tests/test_pipeline_parts.py), and "parts not run" through the API, the Model Evaluation page and the model card (tests/test_parts_not_run.py, tests/test_governance.py).
- Why: unattended run; the spec names these seams ("The benchmark command is tested on the fixture world through the artefact tests").

## 83: Where in the artefact does the PPML benchmark live?
- Options: a block of evaluation.json; its own optional file `ppml_benchmark.json` (`Artefacts.ppml_benchmark`).
- Chose: its own optional file, read by `artefacts.load` as `Artefacts.ppml_benchmark` (None when absent); `artefacts.copy_version` gained a `ppml_benchmark=` argument. Format stays 3.
- Why: the ticket requires the derived version's evaluation report to be identical to the source's, so the benchmark cannot go into it; an optional file keeps older format-3 folders loading.

## 83: Which command derives the new version, and what does the model association use?
- Options: a `pipeline` subcommand; `python -m exportspredictor.ppml`. For the model's association: read the saved `blocked_opportunity_value_usd`, or recompute it.
- Chose: `python -m exportspredictor.ppml` (`ppml.add_benchmark`), which also rewrites the new version's model_card.md so it reads the saved benchmark. The model association (`ppml.model_agreement_association`) recomputes the add-an-agreement Scenario for every UK pair without one through `scenarios.blocked_opportunity_values` -> `prediction.export_potential` with the artefact's `model_form` (the Blend), out-of-fold, then summarises with `model_agreement_effect`.
- Why: the benchmark lives in `ppml`, as the existing `scenarios.rebuild_blocked_values` command lives beside its logic; the ticket requires the association "computed through the one prediction function on the served Model Form", which reading a stored column would not prove.

## 83: How does "parts not run" learn about the benchmark?
- Options: add a block to the evaluation report; give `evaluation.parts_not_run` a second argument.
- Chose: `parts_not_run(report, ppml_benchmark)` (required second argument), with "PPML benchmark" / `--ppml-benchmark` added to `OPTIONAL_PARTS`; the API and the model card pass `artefacts.ppml_benchmark`. `Parts.ppml_benchmark` added, included in `Parts.every()`.
- Why: the report stays unchanged; a required argument stops a caller silently listing a saved benchmark as not run.

## 83: Must the real benchmark avoid opening the 2023 Flows file?
- Options: keep `baci.scan_flows` (a glob over every year, filtered lazily); scan only the panel's years' files.
- Chose: `ppml.real_sector_flows` scans only `flows_{y}.parquet` for y in `YEARS` (2012, 2015, 2018, 2021, 2024).
- Why: "does not read 2023" is then true at the file level, not only after predicate pushdown.

## 83: How much of the notebook to change?
- Options: switch the notebook to read the saved benchmark now; change only the model-card cell.
- Chose: only the model-card cell (no `gravity_benchmark=` argument any more) and the one prose sentence saying the benchmark is passed in. Section 6 still fits PPML in the notebook.
- Why: #88 ("The notebook reads and does not build") owns replacing the fitting cells; `model_card` no longer takes a benchmark argument, so the cell had to change here.

## 87: Which seams to test "How the model was chosen" at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions seams.
- Chose: the API on fixture artefacts (new tests/test_api_model_choice.py: `/evaluation` serves `model_choice` and `event_backtest_note`, degrades with no experiment results, still hides the v1 comparison) and the Model Evaluation page through AppTest against a live API (new tests/test_app_model_choice.py, asserting against the API's values). New files rather than appending to test_api.py / test_app_evaluation.py, so the parallel #86 does not conflict.
- Why: unattended run; the spec names "the API on fixture artefacts" and "the app page tests" as the seams for this.

## 87: Where does "how the model was chosen" live, and in what shape is it served?
- Options: extend `governance._ranking_summary` and call it from the API; a new package module; compute it in the API.
- Chose: new module `exportspredictor.model_choice` (`how_chosen`, `ranking_summary`, `traded_away`, `event_backtest_note`); governance's private `_ranking_summary` moved there as `ranking_summary` (now also with the Noise Floor SD and the confirmation's folds) and the model card imports it. `/evaluation` gains `model_choice` (recorded, model_form {form, naive_weight, text}, selection_measure, validation_year, noise_floor_sd, win_threshold, experiments[], confirmation {n_folds, winner, v2, naive, difference, win_threshold, beats_v2, folds[]}, text, test_year = the report's `earlier_versions` block, traded_away) and `event_backtest_note`. Additive fields only.
- Why: one source for the page and the model card; a new module keeps away from evaluation.py/governance.py edits made by the parallel tickets; the API holds no logic.

## 87: When does the page say what the Blend traded away?
- Options: always; only when an earlier version records a Tweedie deviance and this version's is higher.
- Chose: the latter (`traded_away` is None otherwise, e.g. the fixture, which has no v2 figures). It quotes deviance over all Exporters (433.8 vs 355.5 on the real artefact) and size-matched Lift (0.76 vs 0.79), and says to treat dollar values with more caution than rankings.
- Why: the sentence must be read from the artefact; claiming a trade-off the numbers do not show would be wrong.

## 87: How is "served Scenario Effects are smaller" shown in the Event Backtest section?
- Options: a sentence only; a sentence with the primary Event Backtest's effect rescaled.
- Chose: an info note (served as `event_backtest_note`, None for the model alone) with the model weight and the primary event's pooled predicted effect both ways (+35% model, +23% served on the real artefact), computed by a new `ModelForm.served_effect` ((1+e)^(1−w) − 1).
- Why: the spec says the served effect is the model's effect scaled by the Blend's model weight on the log scale, computed in the package; `ModelForm` is where #84's Scenario note can reuse it.

## 86: Do ranking rows keep the `uncertain_opportunity` field and the Potential Range column?
- Options: drop the row field and the "low–high" column from ranking pages; keep both and remove only the badge, the caveat label and the filter.
- Chose: keep the row field (API rows and /explain) and the Potential Range column; remove the badge, the `uncertain_opportunity` caveat on /labels, `hide_uncertain` (API query, now 422, and `rankings.rank_opportunities`) and the checkbox. `conformal.UNCERTAIN_OPPORTUNITY_CAVEAT` is deleted as unused.
- Why: the ticket removes the badge and filter only; the artefact's tables stay unchanged (ADR 0007), and removing response fields would change existing API meaning beyond the ticket.

## 86: How does the Model Evaluation page state the share inside the Potential Range without rebuilding the artefact?
- Options: page builds a new line from the report's numbers; reword `conformal.coverage_text` and have `/evaluation` re-word `potential_range.text` at serve time.
- Chose: reword `coverage_text` ("N of the M Opportunities (88%) have UK Actual Exports inside their Potential Range (Uncertain Opportunities): most Export Gaps are within the variation normally seen ...") and recompute it in `/evaluation` from the saved report. The page file is unchanged.
- Why: the package owns the wording, the served artefact's saved text is reworded without a rebuild, and leaving `6_Model_Evaluation.py` untouched avoids conflicts with #87.

## 86: What does the Why? page show for an Uncertain Opportunity?
- Options: keep a caveat warning; show only the Potential Range bar and its plain-words text.
- Chose: no warning (caveats come from /labels, which no longer carries it); `potential_range_text` still says where Actual Exports lie and still names an Uncertain Opportunity in that sentence.
- Why: the Potential Range stays on the Why? page per the spec; the warning was the per-row badge in another form.

## 86: Wording tests changed
- Options: keep the exact-sentence test of `conformal.coverage_text`; reduce it to structure.
- Chose: `tests/test_conformal.py::test_the_coverage_text_states_the_held_out_coverage_and_the_uncertain_share` now checks the figures and that the share is tied to the Potential Range. Removed tests: `test_hide_uncertain_is_off_by_default_...` and `test_hide_uncertain_removes_uncertain_opportunities_from_the_leaderboard_summary` (behaviour removed; replaced by a 422 test); the leaderboard and Why? page tests now assert the badge, filter and warning are absent.
- Why: spec #66 testing decisions (no exact sentences; behaviour removed by the ticket is not a behaviour test being deleted).

## 83: The real run (record of the command, its duration and what it read)
- Options: n/a (record required by the ticket).
- Chose: ran `& C:\CodingProjectsC\ep-wt\t83\scripts\wt.ps1 python -X utf8 -m exportspredictor.ppml` (CPU, backgrounded) at 2026-11-18 23:54 UTC; it derived `artefacts/20261119T001736Z` from `20261118T185910Z` in 1,410 s wall-clock (PPML fit 1,382 s, 17 IRLS iterations, converged; whole benchmark 1,394 s). Log: `C:\CodingProjectsC\ep-notes\spec-66\logs\t83_ppml.log`. Then rewrote that version's model_card.md once more after a one-line wording fix to the card's PPML decision entry (model card only; no other file touched).
- Why: no model was trained (the command only loads the fold models and predicts on the CPU) and 2023 was not read (only flows_2012/2015/2018/2021/2024.parquet are opened). Checked after: every file except metadata.json and model_card.md is byte-identical to the source (models, all tables, evaluation.json); `rounds` is still 1,453; model form Blend, w = 0.3.
- Result: PPML +5.1% (95% interval −0.4% to +11.0%; coefficient 0.0501, s.e. 0.0278); published range +21% to +100%; the model's Trade Agreement association on the Blend over 678,216 UK pairs without an agreement is +37.1% Baseline-weighted (mean per pair +44.9%, median +30.4%).

## 84: Which seams to test the Scenario note at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions seams.
- Chose: the API on fixture artefacts (new tests/test_api_scenario_note.py: `/scenarios` carries `note`, `/labels` carries `illustrative`, the note degrades with no saved benchmark, no Event Backtests, no evaluation) and the Scenarios page through AppTest against a live API (tests/test_app_scenarios.py, asserting against the API's values).
- Why: unattended run; the spec names these two seams. A new API test file keeps away from test_api.py / test_api_scenarios.py edits by the parallel tickets.

## 84: Where does the Scenario note live, and in what shape?
- Options: add it to `model_choice`; a new module `exportspredictor.illustrative`.
- Chose: new module `illustrative` with `ILLUSTRATIVE` (the label word), `ILLUSTRATIVE_TEXT` and `scenario_note(artefacts)` -> {label, text, benchmark, model_association, event_backtest, blend}. Each of benchmark/model_association/event_backtest is {run, text, ...figures}, with text "... not run for this version." when absent. `blend` is `model_choice.event_backtest_note` (reused, not copied; None for the model alone). The API serves it as `ScenarioNote` on `/scenarios` (`note`) and the label as `Labels.illustrative` (a `Caveat`: badge + text). #85 can put the same `scenario_note(artefacts)` on the Blocked Opportunities response.
- Why: the note is about Scenario Effects and Blocked Opportunity values, not how the model was chosen; one function, uniform {run, text} parts make degrading one code path and make the page render without branching on which part is missing.

## 88: Which seams to test the notebook's reporting at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions and the ticket's acceptance criteria as the agreed seams.
- Chose: (1) the new `exportspredictor.reporting` module on the fixture artefact (tests/test_reporting.py: the rows behind each chart and table read from the artefact, "not run" where a part is absent, the SVG being a static image); (2) the fixture-mode notebook run (tests/test_notebook.py: a copy runs in memory and the committed file's hash is unchanged; the outputs name the artefact version; chart outputs are `image/svg+xml` and none is Vega/JavaScript); (3) a structural check of the source that no build/fit/train call (`baci.prepare`, `gravity.build`, `hmrc.build`, `policy.build`, `features.build`, `pipeline.build_artefacts`, `experiments.run`, `ppml.fit_ppml`, `ppml.real_sector_flows`, `artefacts.save`, `governance.write_model_card`, ...) runs outside a fixture-only branch, plus that each main result is shown through `reporting`.
- Why: unattended run; the ticket says the module "is tested on the fixture artefact for what it returns, not how it looks", and no test may execute the notebook on real data.

## 88: How are charts made static images, with no plotting library installed?
- Options: add matplotlib (dev group) and return PNG figures; add vl-convert to render Altair to PNG; hand-written SVG from the package.
- Chose: hand-written SVG. `reporting.Chart` holds the title, the rows it draws (`data`, a Polars DataFrame) and the SVG (`_repr_svg_`), so Jupyter stores it as `image/svg+xml`, which GitHub renders. One chart form (`bar_svg`: horizontal bars, optionally floating, with intervals and dashed reference lines) draws every chart. The notebook's two Altair charts (learning curve, gain-ladder waterfall) are gone: the waterfall is `reporting.gain_ladder`, the learning curve keeps its table only (frozen part).
- Why: installing matplotlib into the shared .venv was refused by the permission classifier during this unattended run; SVG needs no dependency, stays crisp and small, and the tests check the rows, not the look. If the owner prefers matplotlib PNGs, only `bar_svg` changes.

## 88: Where does "the loop's saved results" enter the notebook, given the artefact already records them?
- Options: read `loop.load()` and draw from it; draw from the artefact's `metadata["ranking_experiments"]` and only check it against `loop.load()`.
- Chose: the charts and tables read the artefact (as the spec requires); in real mode a cell reads `loop.load()` and prints whether `loop.ranking_record(loop.load())` equals what the artefact records. In fixture mode it uses the artefact's own record.
- Why: one source for the numbers (the artefact, matching the model card and the API), and the check makes a stale or edited results file visible.

## 88: Does real mode still read raw 2023 data in descriptive cells?
- Options: exclude 2023 from every real-mode read; keep the descriptive data cells as they are.
- Chose: keep them. The data section's Flow counts per year, the HMRC cross-check of 2023 UK totals and the 2023-vs-2024 feature drift still read those files. No cell predicts on or scores 2023; the 2023 result is read from the saved evaluation report only.
- Why: "2023 read once" guards the test-year evaluation; counting rows or comparing totals selects nothing and changes no reported model figure. Cutting them is #89/#90's prose call.

## 88: What else changed in the notebook beyond the build cells?
- Options: replace the build cells only; also fix cells that would write into or misread the real artefact.
- Chose: also (a) the model card cell reads the real artefact's saved `model_card.md` instead of rewriting it (the fixture run still writes its own); (b) `saved_to` is the loaded bundle's own folder, not whatever is newest when the cell runs; (c) the controls cell prints the number of control Exporters per backtest direction, not the full lists; (d) the parameters bullet describing real mode now says it reads and does not build (one sentence; the rest of the prose is #89/#90's); (e) a short "How the model was chosen: the experiment loop" subsection in section 7 holds the experiments table, chart and confirmation until #89 moves them into its section 6; (f) `ppml.comparison_table`'s PPML source label says "fitted once and saved with the artefact" instead of "this notebook".
- Why: a real run must not change an artefact file, and the outputs must come from the version the notebook names.

## 88: The real-mode run is 15 minutes, 14 of them global SHAP importance on 20,000 pairs. Cut it?
- Options: keep 20,000 pairs; save global importance with the artefact (a new derived version); compute it on the GPU; sample 2,000 pairs.
- Chose: 2,000 pairs (`explain.global_importance(bundle, sample=2_000)`, seeded as before).
- Why: measured on artefact 20261119T001736Z, 2,000 pairs take 83 s against 846 s, give the same top-five order and move no feature's mean |SHAP| by more than about 2% (log_demand_loo 1.344 vs 1.327; log_supply_loo 1.100 vs 1.085). No new artefact version or GPU path for a frozen explanatory figure. The prose cell after it still quotes "20,000 pairs" for #90 to rewrite.

## 88: The real-mode run time (record required by the ticket)
- Options: n/a.
- Chose: executed a copy (`nbformat.read` → `NotebookClient.execute` with `EXPORTSPREDICTOR_FIXTURE_MODE=0`, output saved only to the scratchpad) against artefact 20261119T001736Z: **110 s (1.8 min)** end to end, no error. Slowest cells: global importance 85 s, loading the artefact 6 s, Scenario presets 4 s. Before the sample cut, the same run took 880 s. The executed copy (479 KB, with outputs) is at `C:\Users\noahl\AppData\Local\Temp\claude\b611510c36a7154a\31ca4637-d278-4e3d-a859-84078c31c66e\scratchpad\t88_real_executed2.ipynb`.
- Why: the copy printed "Outputs below come from artefact version 20261119T001736Z (source: BACI V202601; ...)" and "The artefact records the experiment loop's saved results: True". No model trained, nothing fitted, no table rebuilt; the artefact folder's files kept their timestamps (model_card.md is read, not rewritten). The committed notebook stays output-free.

## 91: Which seams to test the model card and the README at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions and the ticket's acceptance criteria as the agreed seams.
- Chose: (1) `governance.model_card` / `render_model_card` on the fixture artefacts (tests/test_governance.py): section order by heading, and the numbers (headline, Model Form weight, Uncertain Opportunity share, round counts) read from the artefact, with "not run for this version" on `fixture_default_artefacts`; (2) the README by structure (tests/test_readme.py): the opening names the Blend and the Selection Measure and no longer says "This is **v2**"; the notebook row says it reads saved results. No test pins a sentence.
- Why: unattended run; the ticket says "Tests check the model card's sections and numbers against the artefact, not its sentences".

## 91: What does the model card lead with, and in what order?
- Options: add new sections around the old order; reorder the card as an argument like the notebook.
- Chose: reorder. `## Headline` (the report's `earlier_versions` block, else the headline text; plus `model_choice.traded_away`), `## Model Form` (ModelForm text, the model summary, the round-count disclosure), `## Illustrative Scenario Effects` (`illustrative.scenario_note`, each part or "not run for this version"), `## How the Model Form was chosen`, `## Potential Range` (`conformal.coverage_text`: 904,649 of 1,032,453, 88%), then Metrics, Parts not run, Limitations, Intended/Out-of-scope use, Training scope, Features, Excluded Products, Bias review, Data sources, Source versions, Decisions, and last `## Frozen results` with `### Comparison with v1`. The model summary no longer repeats the Blend sentence. `_ranking_lines` uses the glossary's capitals (Selection Measure, Arm, Win Threshold, Naive Baseline).
- Why: story 53 asks the card to lead as the notebook does; frozen parts stay as short reported results at the end. Tests changed (structure only): the earlier-versions test now reads the `## Headline` section; the "names the form" test checks `card["model_form"]`; the gravity benchmark test builds its benchmark from the fixture's saved one (the Scenario note needs its `years`/`level`/`ci_level`).

## 91: How is the round-count disclosure read from the artefact?
- Options: a fixed sentence; read `metadata["round_scaling"]`.
- Chose: `governance._round_count`: `selected_rounds` present (pipeline since #77) -> "chosen by the Selection Measure" with the rule; `tuned_rounds` present (the served real artefacts) -> "Disclosed deviation": 1,453 rounds = the search's 1,198 (deviance early stopping) x 1.2127, not rebuilt so 2023 stays read once, next rebuild uses the median (median_low) of the confirmation winner's fold rounds, 1,097, scaled the same way; neither -> "not recorded".
- Why: the ticket wants the disclosure read from the artefact; `loop.served_configuration` uses the same median_low rule (not called directly: it needs the loop's full settings and raises for forms it cannot serve).

## 91: Who writes model_card.md now the notebook only reads it (#88)?
- Options: leave it to `python -m exportspredictor.governance` by hand; have `pipeline.main` write it into the version it saves.
- Chose: `pipeline.main` writes it after `artefacts.save` (2 lines; `ppml.add_benchmark` already rewrites it for a derived version). README says so.
- Why: otherwise a full rebuild would leave a version with no card for the notebook to read.

## 91: The real model card regeneration (record)
- Options: n/a.
- Chose: wrote only `artefacts/20261119T001736Z/model_card.md` via `governance.write_model_card(store.load(folder), folder)` (no training, no 2023 read; loading predicts nothing). metadata.json, evaluation.json and ppml_benchmark.json checked byte-identical by SHA-256 before/after; no other file's timestamp changed.

## 91: Figures and names this ticket could not yet know (orchestrator to reconcile after #88 merges)
- README rebuild table's notebook row says "about 2 minutes on the CPU", taken from #88's decision-log measurement (110 s). If #88's final run time differs, update that row.
- CLAUDE.md's `notebooks/` line names `exportspredictor.reporting` (#88's module); it exists only once #88 merges. If #88 renames it, update the line.
- Update: #88 merged into spec-66-marker before this ticket finished; merged in, `exportspredictor.reporting` exists and the 110 s run time stands. `scripts/rebuild.py`'s docstring, which still said the notebook trains and saves the artefacts, was corrected here.
- The README's opening keeps the figures from spec #66 (0.555, 0.523, 0.511, 0.540; weights 0.7/0.3).

## 85: Which seams to test the Blocked Opportunities note at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions seams.
- Chose: the API on fixture artefacts (tests/test_api_scenario_note.py: `/opportunities/blocked` carries a `note` equal to `/scenarios`' note, and degrades with no saved benchmark) and the Blocked Opportunities page through AppTest against a live API (tests/test_app_blocked.py: note beside the totals, heading/metric/CSV say "illustrative", page keeps the label and says "not run" with no benchmark and no Event Backtests), all asserted against the API's values.
- Why: unattended run; the spec names these seams, and #84 used the same ones.

## 85: Does the "illustrative" heading apply only on the Blocked Opportunities page, or on every ranking table?
- Options: only the Blocked page; every ranking table (`app.ui.show_ranking`, shared by the Leaderboard, Product to Markets and Market to Products).
- Chose: every ranking table: "Blocked Opportunity value, illustrative (USD, nominal)", word from `/labels` via `app.ui.illustrative()`. The Blocked page's total metric reads "Total Blocked Opportunity value, illustrative". tests/test_app.py's ranking-heading check updated to the new heading.
- Why: GLOSSARY.md makes every Blocked Opportunity value Illustrative, and the column is built once in `show_ranking`; labelling it on one page only would leave the same figure unlabelled elsewhere (spec story 32: never unlabelled).

## 85: The Illustrative text says "associations", which the Blocked page's associational-language test (requires "associate") rejects. Change the text or the test?
- Options: reword `illustrative.ILLUSTRATIVE_TEXT` (shared with #84's Scenarios page and the notebook/model card); accept the stem "associat" in tests/test_app_blocked.py::test_every_caption_uses_associational_language.
- Chose: the test accepts "associat" (associates, associations); the causal-word scan is unchanged.
- Why: the text is associational already; changing #84's shared wording to satisfy a substring check would ripple into other tickets' outputs.

## 85: Where does the `note` field go in the schema, given ScenarioNote is defined after BlockedOpportunities?
- Options: move the note's classes up; a forward reference plus `BlockedOpportunities.model_rebuild()` at the end of api/schemas.py.
- Chose: forward reference and `model_rebuild()`.
- Why: a two-line change that leaves #84's classes where they are, so a parallel edit to schemas.py merges cleanly.

## 89: Which seams to test the rewritten sections 1–7 at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions seams.
- Chose: (1) the notebook's structure and the fixture-mode run (tests/test_notebook.py): `SECTIONS` lists the numbered headings in order and the numbered headings must equal it; `WORD_LIMIT` (300) applies to every prose cell of the preamble and of the sections in `SECTIONS`; each of those sections names a course concept ("Course:"); `reporting.round_count(bundle)` joins the results the notebook must show through `reporting`. (2) `reporting.round_count` on the fixture artefact (tests/test_reporting.py), including the served artefact's shape (search's 1,198 rounds scaled to 1,453; five-fold confirmation rounds → next rebuild 1,097).
- Why: unattended run; the spec says notebook tests check headings in order and the word limit, not sentences. #90 extends `SECTIONS` to 8–11 and the appendix, and the word test then covers the whole notebook with no other change.

## 89: How is a "word" counted for the 300-word limit?
- Options: whitespace tokens; tokens with a letter or digit.
- Chose: whitespace-separated tokens containing a letter or digit (`_words` in tests/test_notebook.py), so table pipes and `|---|` rules do not count.
- Why: a markdown table's separators are not prose; counting them would penalise a table for its layout.

## 89: Where do the parts of old sections 1–7 that the new sections do not use go, given #90 owns sections 8–11 and the appendix?
- Options: leave them in place inside the new sections; delete them; move them, unchanged, after the seventh section.
- Chose: the old sections after 7 keep their old order but lose their numbers ("## Scenarios and Blocked Opportunities", "## Explaining Export Potential", "## Saving the artefacts", "## Governance", "## Conclusions"), so no number repeats. Old section 7's remainder (sector/region, stability, Potential Range, limitations, real results, Event Backtests, v1 comparison, gain ladder) plus old section 5's sanity check of the top Opportunities sit, unchanged, in a new "## Evaluation detail" after Scenarios. The frozen parts of old sections 4–5 (ablation, learning curve, search) moved unchanged into an "## Appendix: frozen results" at the end, which #90 rewrites. Cut outright: the old title cell's findings list and course table (replaced by inline "Course:" links), old Lift intro prose (condensed into section 7), the flow-count, 2024-provisional, GDP-source and policy-disagreement cells, the old training/evaluation intros and old section 5's conclusion.
- Why: the ticket leaves sections 8–11 to #90 "as they are"; numbering them would pre-empt #90's order, and the frozen parts are spec'd for the appendix. Stale prose in the unrewritten sections (e.g. "v2 does not beat the naive baseline") is #90's to rewrite.

## 89: How is the round-count disclosure read from the artefact?
- Options: type 1,198 / 1,453 / 1,097 into prose; a reporting function.
- Chose: new `reporting.round_count(bundle)`: a two-row table, the served model (what chose its rounds: "Tweedie deviance (the hyperparameter search)" when `round_scaling` records `tuned_rounds`, "Selection Measure" when it records `selected_rounds`; rounds chosen and trained) and the next rebuild (`loop.served_configuration(ranking_experiments).rounds`, i.e. the median of the five-fold confirmation's rounds, before scaling).
- Why: the ticket asks for the disclosure and for figures from the artefact, not prose; the artefact already records both counts, and `loop.served_configuration` is the pipeline's own rule for the next rebuild.

## 89: Where does the notebook load the artefact, now that section 1 shows the headline?
- Options: keep loading in old section 5; load right after the imports.
- Chose: one cell after the imports (in the Parameters preamble) builds the fixture artefact (with the fixture ablation/curve) or loads the newest real artefact and `experiments.load()`, sets `report`, and prints `reporting.provenance`. The appendix's ablation cell reads `experiment_results` from there instead of running `experiments.run` again.
- Why: section 1 must print the headline from the artefact before any detail.

## merge 91: does the README's manual UI checklist still match the Blocked Opportunities page after #85?
The merge was clean, and no checklist item contradicts #85. But nothing checked #85's change: the total and the value column are now headed "illustrative", and an Illustrative note (benchmark, model association, Event Backtest) sits under the summary. Chosen: add one Blocked Opportunities checklist item for the headings and the note, in the merge commit's follow-up. Alternative (leave it) rejected: the checklist is what a release is checked against, so it should cover the new UI.

## 89: The real-mode check of sections 1–7 (record)
- Options: n/a.
- Chose: executed a copy in real mode (`EXPORTSPREDICTOR_FIXTURE_MODE=0`, nbclient, output saved only to the scratchpad: `C:\Users\noahl\AppData\Local\Temp\claude\b611510c36a7154a\31ca4637-d278-4e3d-a859-84078c31c66e\scratchpad\t89_real_executed.ipynb`) against artefact 20261119T001736Z: 112 s, no error. Printed from the artefact and equal to docs/research/opportunity-ranking-experiments.md: 2023 UK ranking 0.555 / Naive Baseline 0.523 (and 0.523 as reported for v2) / v2 0.511 / v1 0.540; Noise Floor SD 0.00444, Win Threshold 0.00887; five-fold confirmation 0.5635 vs 0.5206 (+0.0429), folds +0.040 to +0.045; deviance over all Exporters 433.8 vs 355.5; size-matched Lift 0.76 (0.68–0.83) vs v2 0.79; round count served 1,198 (search, deviance) scaled to 1,453, next rebuild 1,097.
- Why: the ticket requires the headline, the 2023 comparison and the cost figures to match the research note on the real artefact. Nothing was trained or fitted, and no cell scored 2023 (it is read from the saved evaluation report). The committed notebook stays output-free.

## 90: Which seams to test the rewritten sections 8–11 and the appendix at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the spec's Testing Decisions seams.
- Chose: the notebook's structure and its fixture-mode run (tests/test_notebook.py). `SECTIONS` now lists every section after the parameters, appendix included, and the section headings must equal it. The word limit applies to every prose cell. A new test keeps the total within 10% of `TOTAL_WORDS` (10,000). Each of the six `FROZEN_PARTS` must be exactly one `###` entry in the appendix. The fixture run must print the package's Scenario note (its text and each part's text, from `illustrative.scenario_note`) and the Uncertain Opportunity share from the evaluation report. `reporting.ppml_benchmark` and `reporting.gain_ladder` joined the results the notebook must show through `reporting`.
- Why: unattended run. The spec says notebook tests check headings in order and word counts, not sentences. The Scenario note and the share are checked as values read from the package and the artefact.

## 90: Section 10 is called "Limits and governance", but the governance test looked for "Governance" with a capital G. Rename the heading or change the test?
- Options: use the heading "Limits and Governance"; match "governance" in lower case in `test_the_write_up_covers_governance`.
- Chose: the test matches lower case. The five `###` subheadings (Model card, Data lineage, Bias review, Regulation, Drift) are still required.
- Why: headings in the notebook are in sentence case. "Governance" is not a glossary term, so it takes no capital.

## 90: What was cut or moved from the old unnumbered sections?
- Options: keep each old cell and shorten it; rebuild each section around the story and link the rest.
- Chose: rebuild.
  - Old "Evaluation detail" is gone. The Event Backtests moved into section 8. The sanity check of the top Opportunities (top 10, plus the shares table), whisky, the Potential Range and the limitations moved into section 10. The v1 comparison, the gain ladder and the constraint comparison became appendix entries.
  - "Saving the artefacts" is gone. Its fixture-only `artefacts.save` now sits in the model card cell, and its prose went into Data lineage.
  - Cut cells: the PPML parameters dict, the control-count cell, the coverage-by-Exporter tables (`range_coverage_by_exporter`), the per-Exporter coverage table, the top-10 Potential Range table, the full stability dict (its text is printed), the learning-curve plateau table, the search's trial table and its learning-curve re-check, and the v1 comparison's two metric tables (its verdict text is printed).
  - Section 1's conclusion gained one sentence pointing to sections 8–11.
  - The global-importance prose now says 2,000 pairs.
  - Every figure quoted in the new prose comes from the served artefact 20261119T001736Z, not the older v2 artefacts. Exceptions are the gain ladder (20261118T082751Z) and the constraint comparison (20261118T042154Z). They were run for v2 only, so the prose names their artefacts, and the cell under each prints "not run for this version".
- Why: the spec asks for the argument in the body, with cut detail linked rather than moved into another long cell. The removed tables are still in the evaluation report, the model card and the app.

## 90: The v1 comparison on the served artefact calls the Blend "v2", and its verdict is "mixed". Reword it or explain it?
- Options: change `comparison`'s labels; print the verdict as it is and explain it in the prose.
- Chose: print it as it is. The appendix says the code calls the newer version "v2". It also says the result is mixed: the Blend ranks better than v1 (0.555 against 0.540, interval above 0) but is worse on deviance and size-matched Lift beyond their intervals.
- Why: the comparison is a frozen part (spec: not extended), and its text is in the saved evaluation report, so changing the code would not change what the artefact records.

## 90: The total is 9,441 words, not 10,000. Pad it to 10,000?
- Options: add prose to reach 10,000; stop inside the 10% band.
- Chose: stop at 9,441 (largest prose cell 269 words). The test allows 9,000–11,000.
- Why: the spec says "about 10,000 (within 10%)", and the extra words would have been filler.

## 90: The real-mode check of the whole notebook (record)
- Options: n/a.
- Chose: ran a copy in real mode (`EXPORTSPREDICTOR_FIXTURE_MODE=0`, nbclient). Its outputs were saved only to the scratchpad: `C:\Users\noahl\AppData\Local\Temp\claude\b611510c36a7154a\31ca4637-d278-4e3d-a859-84078c31c66e\scratchpad\t90_real_executed.ipynb`. It ran against artefact 20261119T001736Z in 101 s with no error.
  - The prose was checked against these outputs: the Scenario note (PPML +5.1%, −0.4% to +11.0%; Blend association +37.1% over 678,216 pairs; EU–Japan +35% against +2%, served +23%); presets (EU re-join +$38.3bn, 27.8%; CPTPP +$4.7bn, 12.1%); global importance (demand ×3.83, supply ×3.00); the top Opportunity (US$1.4bn, fold 2); Russia fourth for whisky; coverage 85% and 904,649 of 1,032,453 (88%) Uncertain Opportunities, none in the top 1,000; region and income ranks; the search winner; and the gain ladder and constraint comparison "not run for this version".
  - Two sentences were corrected after the run: band coverage, and the Naive Baseline doing better in the weakest Sectors.
- Why: the ticket asks for figures checked on real data. Nothing was trained or fitted, and 2023 was not scored (it is read from the saved report). The committed notebook stays output-free.

## orchestrator: Which pre-#92 code-review findings to fix, defer or keep?
- Options: fix all; fix hard breaches and divergent duplicates; fix only hard breaches.
- Chose: one fix implementer for the hard breaches (duplicate `NOT_RUN`; schema contract tests in tests/test_api.py), the divergent duplicates (round count, primary Event Backtest), the other duplicates (default Model Form, experiment-loop sentence), glossary casing, the `test_year`/`naive` and `_pct` names, the missing `ignore_patterns` entry, the `/evaluation` route re-wording, the README slips, and a note that the Scenario note's figures cover different populations. Deferred to #92: the rebuild script's notebook step lacks `--inplace` and can't run alone. Kept: the `winner`/`v2` field names, interval types for the Data Clumps, and the app/governance note-rendering loops (the app cannot import the package).
- Why: the divergent duplicates could print different numbers in the model card and the notebook, which #92 then commits. The kept items are naming or shape preferences that cost schema churn and fix no wrong output.

## orchestrator: The notebook's real mode still reads 2023 for description (Flow counts per year, the HMRC cross-check, drift 2023 vs 2024). Remove them?
- Options: remove or skip the 2023 reads in real mode; keep them as descriptive data checks.
- Chose: keep them, and flag them in the final report.
- Why: no cell scores, predicts on or selects with 2023, and none of these reads can change a reported model result. The spec's "does not read 2023" protects the once-read test evaluation. Removing the cells would drop the data-quality checks a marker expects. The user can reverse this.

## review1: Which seams to test the review fixes at (the tdd skill asks to confirm them)?
- Options: confirm with the user; test at the existing public seams.
- Chose: the API contract (tests/test_api.py) for the schema fields and renames; `model_choice` for the one round count, the one primary Event Backtest and the shared Model Form and selection-rule wording (read by `governance.model_card` and `reporting.round_count`); `artefacts.copy_version` for the optional JSON files; `evaluation.served_report` for the re-worded Potential Range; `illustrative.scenario_note` for the population sentence. Pure renames and moves (NOT_RUN, percent helpers, glossary casing, README) are covered by the existing suite, with no new test.
- Why: unattended run; these are the seams the findings name, and each new test fails before its fix.

## review1: Where does the one round-count function live, and which rule does it use?
- Options: keep governance's own median of the confirmation's rounds; use `loop.served_configuration` (the pipeline's rule) in one package function.
- Chose: `model_choice.round_count(metadata)`, using `loop.served_configuration(record).rounds` for the next rebuild and `metadata["rounds"]` for the rounds trained; `governance.model_card` and `reporting.round_count` both read it. `chosen_by` is now "Tweedie deviance (the hyperparameter search)" (the notebook's label) rather than "Tweedie deviance".
- Why: the next rebuild's count should come from the rule the pipeline will apply, not a copy of it. Checked on 20261119T001736Z: the table and the card's sentence are unchanged (1,198 chosen on deviance, 1,453 trained, 1,097 next rebuild), so the model card was not regenerated for this.

## review1: The API's contract tests for the new fields were in tests/test_api_scenario_note.py and tests/test_api_model_choice.py. Add more, or move them?
- Options: add a second set of tests to tests/test_api.py; move the existing ones there.
- Chose: move them, unchanged, into tests/test_api.py under two section headers, and delete the two files.
- Why: they already cover `Labels.illustrative`, `BlockedOpportunities.note`, `ScenarioResponse.note`, `Evaluation.model_choice` and `Evaluation.event_backtest_note`; CODING_STANDARDS wants them in tests/test_api.py, and a second copy would duplicate them.

## review1: Can a copied version carry a stale PPML benchmark (copy_version's ignore_patterns omits PPML_BENCHMARK_FILE)?
- Options: drop the benchmark on copy; keep it unless a new one is given, as the evaluation report is kept.
- Chose: keep it unless given. The omission was not a bug: copytree carried the source's benchmark, and a given one overwrote it. Every caller (refresh, compare, constraints, backtests, evaluation, ppml) copies the source's fold models and Model Form unchanged, so the benchmark stays true of the copy. Refactored to one `_OPTIONAL_JSON` table used by save, load and copy_version, with a test that a copy keeps the source's results unless given new ones.
- Why: dropping it would make `pipeline refresh` lose the benchmark; a table means the next optional file cannot be missed in one of the three places.

## review1: Where does the "different populations" sentence (story 29) go, and is the real model card regenerated?
- Options: a separate note part; a sentence ending the Event Backtest part's text.
- Chose: `illustrative.DIFFERENT_POPULATIONS`, appended to the Event Backtest part's text when that part was run (it follows the model's association in the API, the app pages, the notebook and the model card). The API test asserts the package constant appears in the served text; the page tests already assert each served part's text is shown. Regenerated `model_card.md` of 20261119T001736Z in place (no training, no new version): the only changes are that sentence and "Naive Baseline" in the metrics header.
- Why: one sentence, in the package's wording, reaches every place the note is shown without a schema change.

## review1: Rename the API's `naive` field of the test-year ranking too?
- Options: rename it (e.g. `naive_baseline`); keep the key and type it honestly.
- Chose: keep `naive`, typed `list[NaiveBaselineResult]`; the rows are `list[VersionResult]` (this version first); `how_chosen()["test_year"]` and `ModelChoice.test_year` become `test_year_ranking`.
- Why: `naive` is the saved report's own key (`evaluation.earlier_versions`), and the mislabel was the `EarlierVersion` type; renaming the key would need a mapping layer for no gain.

## review1: Fix the other lowercase "naive baseline"/"model form" uses that predate this branch (evaluation.py, prediction.py, the Model Evaluation page's column labels, README checklist)?
- Options: fix all; fix those the finding names plus new text in `git diff main...HEAD`.
- Chose: the named lines and the branch's new text only (artefacts.py, ppml.py, the model card's metrics header, README).
- Why: the finding scopes it to the diff; the older uses change app labels and page tests outside this review.

## 92: Which seams to test the committed outputs and the rebuild script's notebook step at (the tdd skill asks to confirm them)?
- Options: confirm with the user; take the ticket's acceptance criteria as the agreed seams.
- Chose: (1) `scripts/rebuild.py` through its CLI and `STEPS` (tests/test_rebuild.py): the notebook step saves its outputs in place (`--inplace`), and `--only notebook` selects that step alone, checked with `--list --only notebook` so nothing runs; (2) the committed notebook file read as JSON (tests/test_notebook.py): every code cell has outputs, none is an error, a chart is `image/svg+xml`, and the provenance line names a timestamped artefact version whose source is not the fixture. The fixture-mode run test stays as it is (copy in memory, hash unchanged).
- Why: unattended run; the ticket names these checks, and neither test executes the notebook on real data.

## 92: How does the rebuild script run only the notebook step?
- Options: a `--step N` number; a `--notebook` flag; `--only TEXT` selecting the steps whose name contains TEXT.
- Chose: `--only TEXT` (case-insensitive substring of the step name; exits 2 when nothing matches), combinable with `--list`. The refresh command is `uv run python scripts/rebuild.py --only notebook`.
- Why: a step number shifts when a step is added; a substring selector is general and the README command reads plainly.

## 92: The real run that produced the committed outputs (record)
- Options: n/a (record required by the ticket).
- Chose: ran `& C:\CodingProjectsC\ep-wt\t92\scripts\wt.ps1 python -X utf8 scripts/rebuild.py --only notebook` (backgrounded, 20-minute deadline) against artefact 20261119T001736Z, confirmed the lexically newest and the one holding `ppml_benchmark.json`. **103 s** wall-clock, exit 0. The executed `notebooks/exports_predictor.ipynb` is **343,132 bytes (335 KB)**, well within GitHub's notebook rendering. Log: `C:\CodingProjectsC\ep-notes\spec-66\logs\t92_notebook_run.log`.
- Checked: provenance line names 20261119T001736Z (source BACI V202601, the Blend w 0.3, derived from 20261118T185910Z); headline "0.555 ... naive baseline (0.523 ...) ... v2 0.511, v1 0.540" equals the model card and docs/research/opportunity-ranking-experiments.md; 4 SVG charts, no Vega/JavaScript; no error outputs or tracebacks; every code cell but the imports cell has outputs; the largest output is the rendered model card (28 KB Markdown), tables are capped (e.g. 20 rows). The artefact folder's listing (sizes and timestamps) is identical before and after. No model trained; 2023 not scored (read from the saved report; the descriptive 2023 reads the orchestrator kept still run).
- Why: the ticket's acceptance criteria.

## 92: The HMRC cross-check cell prints a Polars warning ("POLARS_MAX_BLOCKING_THREAD_COUNT reached (512), this may indicate a deadlock") on stderr. Fix and rerun?
- Options: leave it in the committed outputs; change the cross-check's scan in the package and rerun.
- Chose: leave it. The cell completes and prints its result; the same warning appeared in #90's real run.
- Why: it is a Polars thread-pool warning, not an error or a stack trace, and it changes no figure; diagnosing Polars' async scan is outside this ticket. A later fix needs only `scripts/rebuild.py --only notebook` rerun.

## review2: which seams the review fixes are tested at (tdd asks to confirm seams)
- Options: confirm with the user (not possible unattended); test at the public functions the findings name.
- Chose: `scripts/rebuild.py`'s notebook step run on a temporary notebook (encoding round trip); the committed notebook's cell sources (mojibake); `model_choice.how_chosen`/`ranking_summary` (experiment count and labels); `evaluation.result_limitations`/`range_limitations` (wording); `evaluation.earlier_versions` and the read-time relabel of saved reports; `comparison.verdict` and the model card's v1 comparison (`governance.render_model_card`); `reporting` helpers (provenance parse, model card excerpt); `ppml.add_benchmark` lineage.
- Why: each is the interface the notebook, the model card or the API reads, so the tests survive refactors.

## review2: the notebook printed "Of 9 experiments" against the prose's seven; which is right?
- Options: edit the prose to say 9; count the note's experiments in the package.
- Chose: seven is right. The artefact records 9 decisions of the rule (`naive baseline`, `select on ranking`, `two-part model`, `scope: UK Zero Cells`, `scope: zero rate`, three `product detail: …`, `monotone constraints`); the research note numbers them 1–7: the blend (1) and the starting point (2) share the `naive baseline` decision, and the scope (5) and Product-detail (6) experiments decide each Arm on its own (7 − 1 + 1 + 2 = 9). New `loop.experiment_numbers`/`experiment_label`/`count_experiments` (in `loop/arms.py`, beside the definitions) map a decision and its Arms to the note's numbers; a decision the note does not number counts as one. `ranking_summary` gains `label` and `n_experiments` per entry (additive, also in the API schema); the lead sentence reads "Of 7 experiments, decided in 9 steps, 1 kept a change (1–2. Naive Baseline: blend)"; the fixture (blend only) reads "Of 1 experiment". The notebook table, the model card table and the app show the labels.
- Why: the finding asked to fix the count in the package rather than the prose; the labels also tie each row to the note's table.

## review2: how to stop the model card calling the Blend "v2" in its frozen v1 comparison, without re-running it
- Options: add a note under the frozen section; edit the saved evaluation.json strings; re-word the saved figures at read time.
- Chose: re-word at read time. `comparison.verdict(..., name=)` names the compared version; `comparison.saved_verdict_text(block, name=)` re-words a saved block from its saved primary and secondary figures (and `secondaries_not_run`); `comparison.relabelled(report)` applies it with `version_name(report)` ("the Blend" when the report's Model Form is a blend, else "v2"). A real-data test checks that re-wording the real block as "v2" reproduces its saved text exactly (bar the Naive Baseline's casing), so nothing but the name changes. The card's "Primary metric" line uses the name too. The gain ladder's steps still say "v2's … configuration", which is true (the Blend kept v2's configuration). The pipeline now passes the name for future runs. The saved evaluation.json is not edited.
- Why: the figures are saved; only the label was stale. Editing a saved result file is harder to audit; a note would leave the contradiction in place.

## review2: Glossary casing in printed output ("naive baseline", "This version (blend)") when the saved report holds the old strings
- Options: change only the producers (the served artefact would still print the old names); edit the saved report; re-word at read time.
- Chose: producers changed (`NAIVE_THIS_RUN`/`NAIVE_AS_REPORTED_FOR_V2` now "Naive Baseline, …", `evaluation.version_name` gives "Blend", `headline`'s and `earlier_versions`' sentences, `comparison.CONDITIONS`, two model card decision lines) and `evaluation.relabelled(report)` re-words a saved report's headline and `earlier_versions` from their saved figures with today's names; `served_report`, `model_choice.how_chosen`, `governance.model_card` and `reporting.ranking_by_version` read through it. The loop's recorded experiment name "naive baseline" is a saved key, so it is shown through `loop.experiment_label` ("1–2. Naive Baseline") instead of renamed.
- Why: no saved JSON keys or files change, and older artefacts read the same way as new ones (the same pattern as `served_report`'s Potential Range wording).

## review2: two "known limitations" contradicted the Blend and the removed badge
- Options: delete them; re-word them.
- Chose: re-word. For a served Blend, the ranking limitation now says the Blend was chosen on this ranking, not on the size of Export Potential, so treat dollar values with more caution than rankings (story 48); a model-alone version keeps "its value is mostly in the size". The Potential Range limitation now says what being an Uncertain Opportunity means for a large pair ("a small gap for its size", not that the model is unsure), with no badge (story 41).
- Why: both limitations still describe real limits; only their stale claims were wrong.

## review2: lineage prose says `copied_from`, but the served version records `ppml_benchmark_from` and its source's `created_at`
- Options: make `ppml.add_benchmark` also set `copied_from` and a fresh `created_at`, and edit the real artefact's metadata.json; fix the prose.
- Chose: fix the prose (cell "Data lineage"): a derived version records which one it came from as `ppml_benchmark_from` (the PPML step, the served version) or `copied_from` (the other derivation steps), and keeps its source's `created_at`, when its models were trained. The lineage cell now prints `ppml_benchmark_from` too. No code or artefact change.
- Why: every derivation step (backtests, constraints, evaluation, pipeline's v1 comparison, ppml) keeps the source's `created_at`, so that is the convention, not a bug; `ppml_benchmark_from` is deliberate and `reporting.provenance` already reads it. Editing the served artefact's metadata is avoidable risk.

## review2: governance embedded the whole model card (about 28 KB)
- Options: show the first section only; show headings only; show the headline plus the list of sections.
- Chose: new `reporting.model_card_excerpt(path)`: the card's title, its Headline section and the names of its sections, naming the file (`<version>/model_card.md` in the artefacts directory). About 1.6 KB on the real card. A test on the committed notebook now caps any one output at 15 KB.
- Why: the spec says governance links to the card for the rest and outputs stay small; the headline is what the notebook leads with too.

## review2: committed outputs show `FIXTURE_MODE` False under `FIXTURE_MODE = True`
- Options: markdown note; code comment.
- Chose: a two-line comment in the parameters cell naming `EXPORTSPREDICTOR_FIXTURE_MODE=0` and `scripts/rebuild.py --only notebook`.
- Why: it sits next to the output it explains; the markdown above already describes the variable.

## review2: the rebuild script garbled the notebook's prose
- Options: PYTHONUTF8=1 in the notebook step's env; call nbclient from the script with explicit encodings.
- Chose: PYTHONUTF8=1 in the step's env. Cause verified: `nbclient/cli.py` opens the input with `input_path.open()` (locale encoding, cp1252) and `nbformat.write` saves UTF-8. A test runs the step's command (with its env) on a temporary notebook holding an en dash, a pound sign and an accented name, and checks they round-trip; it failed before the fix. 16 markdown cells were restored (their sources equal 858bf18's after undoing cp1252); a test rejects mojibake in any cell source. Separately, the real `data/derived/country_map.parquet` holds a garbled name ("Saint Barth00c300a9lemy (escaped)") that shows in the gravity cell's output; that is derived data, left for a follow-up (fixing it means rebuilding the gravity step).
- Why: one line at the step that caused it; jupyter execute stays the documented command.

## review2: refreshing the real model card and the notebook's outputs
- Options: (none needed; done as instructed, once, after every code and source change was committed)
- Chose: rewrote `artefacts/20261119T001736Z/model_card.md` in place with `governance.write_model_card(artefacts.load(folder), folder)` (no training, nothing scored; the folder listing is unchanged except model_card.md, 27,593 → 27,963 bytes; the old card is kept in the scratchpad as `model_card_before_review2.md`). Then `scripts/wt.ps1 python scripts/rebuild.py --only notebook`, backgrounded under a 20-minute `timeout`: exit 0 in **105 s**. The notebook is **312,003 bytes** as stored (316,127 on disk, CRLF) (338,833 at d597c16, with the full card; sizes as stored, LF); largest output 11.3 KB (the limitations table); 0 mojibake in cells and outputs bar the derived-data name noted above ("Saint Barth00c300a9lemy (escaped)", in the gravity cell's output only); no error outputs; provenance names 20261119T001736Z; headline 0.555 vs 0.523; "Of 7 experiments, decided in 9 steps, 1 kept a change". Cell sources are identical to the committed ones. Committed as 5d6780f.
- Why: the finding asked for one refresh after all changes; any later edit would need another run.

## orchestrator (after the run, at the owner's request): matplotlib charts
- Options: matplotlib PNG; matplotlib SVG.
- Chose: matplotlib SVG with text kept as text (`svg.fonttype: none`), a short font list, comments and whitespace stripped, no date, salted ids. Charts are 9-22 KB. A chart image gets its own cap, `reporting.CHART_LIMIT_BYTES` (30 KB); printed outputs and tables keep 15 KB.
- Why: a base64 PNG of these charts is 16-54 KB and blurs on zoom. SVG renders on GitHub, stays crisp, and re-runs write the same bytes. Notebook refreshed in 133 s; sources unchanged.
