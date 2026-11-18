# Decision log: spec #39 (UK Exports Predictor v2)

Unattended /implement-spec run started 2026-11-17. Format per docs/agents/unattended-runs.md.

## orchestrator: The user typed `/imlpement-spec 39` (typo); run it?
- Options: stop and ask; treat as `/implement-spec 39`
- Chose: treat as `/implement-spec 39`, following .claude/skills/implement-spec/SKILL.md by hand
- Why: the intent is unambiguous, and unattended runs take the recommended option.

## orchestrator: Integration branch name, and whether to open a PR
- Options: `spec-39-integration` with no PR; open a draft PR
- Chose: `spec-39-integration` off main, checked out in worktree C:\CodingProjectsC\ep-wt\integration; no PR. Tickets are closed by comment + `gh issue close`.
- Why: matches the v1 run (spec-10-integration, merged locally). docs/agents/issue-tracker.md closes work by comment + close, not PRs. Opening a PR or pushing reaches outside the repo.

## orchestrator: Where the decision log ends up
- Options: leave only outside the repo; copy into docs/runs/
- Chose: keep here during the run; copy to docs/runs/spec-39-decisions.md on the integration branch at the end
- Why: matches docs/runs/spec-10-decisions.md from the v1 run.

## #40: Where the held-out Exporter list goes ("four tables" plus the list)
- Options: a fifth small table `lift_held_out_exporters.parquet`; only in the manifest; a column on the Lift table
- Chose: a fifth table, listed in the manifest with its own row count and checksum
- Why: the spec says "plus the held-out Exporter list itself"; a table keeps every frozen fact checksummed the same way, and Exporters with no rows in 2015–2017 are still listed.

## #40: Which rows the Event Backtest table holds
- Options: post years only (2019–2021) for the treated pairs of each direction; also pre years and control Markets
- Chose: each direction's treated pairs (EU27→JPN, JPN→EU27) in 2019–2021, with `export_potential_on_usd`/`_off_usd`, value and weight
- Why: that is exactly what v1's Event Backtest predicted; the observed difference-in-differences uses observed Flows only, which v2 can read itself (as with the naive baseline).

## #40: What a re-run does
- Options: overwrite and reproduce checksums; no-op that verifies the existing manifest
- Chose: no-op that verifies checksums and row counts (raises if a file no longer matches); a fixture test also shows a fresh freeze reproduces identical checksums
- Why: v1 must never be rewritten after the freeze; the ticket allows either.

## #40: CPU or GPU for the freeze's predictions
- Options: predict on the GPU (faster); predict on the CPU as v1's evaluation and API did
- Chose: CPU (the saved models are set to `device=cpu`); no GPU lock taken
- Why: identical numerics to v1's own evaluation; no training happens.

## #40: Guarding provenance of the real run
- Options: record whatever commit is checked out; refuse unless HEAD is the clean `v1` tag
- Chose: `python -m exportspredictor.freeze` refuses unless the `v1` tag points at HEAD and tracked files are clean; it also checks that `2023_20e7c6bf9547` is the test-model cache key of the v1 artefact's metadata
- Why: the manifest's tag and commit must really be the code that produced the tables.

## #40: Whether to add the freeze to README's rebuild-steps table
- Options: add a row; leave README alone
- Chose: leave README alone; the module docstring documents the command
- Why: the freeze is a one-off tied to the `v1` tag, not a rebuild step, and the ticket asks that the tagged commit change nothing but the freeze command.

## orchestrator: Use a merger subagent for trivial merges?
- Options: always a merger subagent; merge inline when it is a fast-forward or conflict-free
- Chose: merge inline (merge, ruff, pytest) when there are no conflicts; use a merger subagent when a merge conflicts
- Why: a clean merge is three commands, and a subagent adds cost without accuracy. Conflicts need focused context, so they go to a subagent.

## #42: Where `artefact_format: 2` is stamped
- Options: `artefacts.save` adds it to any metadata; the pipeline's metadata dict sets it (from `artefacts.ARTEFACT_FORMAT`)
- Chose: the pipeline sets it in `_train_and_score`'s metadata; `save` writes metadata as given
- Why: every metadata key is set in one place today; `copy_version`/`refresh` carry it through, and a test asserts fixture artefacts are saved as format 2.

## #42: One error type for wrong format and missing table?
- Options: one `UnreadableArtefacts(ValueError)`; separate exceptions per cause
- Chose: one `artefacts.UnreadableArtefacts`, with the message naming the format found ("artefact format 1" / "no artefact_format") or the missing table(s)
- Why: the API treats both the same (not ready, reason in `/health`); callers need no distinction.

## #42: What `/health` returns when not ready, and whether to fall back to an older folder
- Options: fall back to the newest readable folder; stay not ready with a reason
- Chose: stay not ready (503) with `reason` (the load error, also for "no artefacts"); `reason` is omitted from the ready response (`response_model_exclude_none`), so the ready contract is unchanged; retry every `retry_seconds` as before
- Why: the ticket says the API still serves the lexically newest folder and never serves a bundle it cannot read.

## #42: Package commands that load the real v1 artefact
- Options: let `evaluation/backtests/constraints.add_to_latest`, `pipeline refresh`, `governance` refuse v1 folders; special-case v1
- Chose: they refuse (they all go through `artefacts.load`); `tests/test_scenarios_real.py` skips with the reason instead of failing
- Why: v1 is frozen (#40); those commands are meant to run on v2 folders from #58 on.

## #41: How "the old parameter is gone" is enforced for `min_gap_gbp`
- Options: drop the field (FastAPI silently ignores unknown query params); drop it and set `RankingQuery` to `extra="forbid"` so an unknown parameter is a 422
- Chose: `extra="forbid"` on `RankingQuery`; a contract test asserts `min_gap_gbp` gives 422
- Why: a silently ignored `min_gap_gbp` would return an unfiltered ranking to an old client; 422 makes removal visible. Later tickets adding ranking filters (#44, #46, #55) must declare them as `RankingQuery` fields, which they would anyway.

## #41: Where the USD caption, unit label and money formatter live
- Options: literals in `app/ui.py`; constants and `format_usd`/`format_change` in `exportspredictor.currency`, imported by the app
- Chose: `currency.USD_CAPTION`, `currency.USD_UNIT` ("USD, nominal") and `currency.format_usd`/`format_change` ($ only); the app imports them (the package is installed in the app image)
- Why: CODING_STANDARDS puts money formatting in the package and wants one definition per constant; these are pure formatting, not model access, so the "app talks to the model only through the API" rule is untouched.

## #41: Where each page's one USD caption is shown
- Options: per page; inside `ui.banner()`, which every page calls
- Chose: `banner()` shows it right after the Baseline-year banner (only when the API is up)
- Why: every page, including future ones (#46 Leaderboard, #47 Blocked Opportunities), gets exactly one caption for free. The ranking pages' two captions (order + rate) were folded into one "Ranked by …" caption.

## #41: How the HMRC cross-check picks its £ rate
- Options: keep the caller passing `usd_per_gbp`; default it to `currency.USD_PER_GBP[year]`
- Chose: `hmrc.crosscheck(..., usd_per_gbp=None, year=...)` defaults to the year's own Bank of England annual average (explicit rate still accepted); the notebook cell no longer imports `currency`
- Why: the acceptance criterion "converts £ at each year's own rate" becomes a property of the function, tested for 2022 and 2023.

## #41: What `pipeline.refresh` does with v1's `gbp_rate`
- Options: carry it over; drop it
- Chose: drop it from the refreshed metadata
- Why: v2 artefacts must not carry GBP metadata; the API no longer reads it anyway (`_metadata` is version + Baseline year only).

## #41: Table cells and the `$1.2bn` format
- Options: format table cells as strings like "$1.2bn"; keep numeric cells with "(USD, nominal)" headers and use `$1.2bn` in prose, metrics and the Scenario summary
- Chose: numeric cells (sortable; CSV export stays numeric and equal to the screen) with "(USD, nominal)" headers; `$1.2bn` on the Why? values line, the Scenario Effect metric and the Scenario summary sentence
- Why: string cells would break sorting and the CSV round trip; the spec's "written like $1.2bn" fits text, the header carries the unit for tables.

## #48: Which seams the tariff tests use (TDD skill asks to confirm seams)
- Options: confirm with the user; use Seam 2 (package functions) as the spec's Testing Decisions list
- Chose: Seam 2 only: `tariffs.download` (resume, with an injected fetch), `tariffs.panel` + `tariffs.applied_tariffs` on hand-written TRAINS-shaped rows, the fixture world's tariffs, and a real-data skip test
- Why: unattended run; the spec names these package tests and adds no new seam.

## #48: Shape of the "bilateral panel"
- Options: one dense Importer × Exporter × HS6 × year table; three compact tables (schedules, MFN, preferences) plus `applied_tariffs(rows, tariffs, agreements)` that combines them per row
- Chose: three compact tables under `data/derived/tariffs/` and the `applied_tariffs` function (returns applied_tariff, mfn_tariff, tariff_imputed, in percent; #49 takes ln(1+t/100) or as it decides)
- Why: CLAUDE.md forbids the dense grid; EU members share one `EUN` schedule; Customs Union and post-2021 zeroing need the policy table anyway, so they are applied at join time.

## #48: How the raw TRAINS download is cached
- Options: raw XML per reporter-partner-year (~20 GB); one Parquet per reporter-year (parsed, written when all its partners are fetched)
- Chose: one Parquet per reporter-year under `raw/trains/<year>/<ISO3>.parquet`, plus availability.xml, countries.xml and a manifest.json written at the end (it records the pull date)
- Why: the ticket says "cached per reporter-year"; Parquet is ~100x smaller, so lineage checksums stay fast. An interruption loses at most the in-flight reporter-years. Requests use XML (SDMX structure-specific), parsed by regex; 404 = no records.

## #48: Which HS revisions are converted to HS12
- Options: HS17 only (as the ticket says); HS17 and HS07; all
- Chose: HS17 (H5) and HS07 (H3) via the UNSD correlation tables; HS12 as is; H0–H2 reporter-years dropped (gap fill may cover them)
- Why: HS07 is 60 of 137 reporter-years in 2012, so dropping it would leave large gaps; the same code path handles it at no extra cost. H2 and older are ~1 reporter-year a year after 2013.

## #48: Mean over several HS17 codes when a preference covers only some of them
- Options: average MFN and preference separately, then take the lower; average the lower rate per code (preference where it exists, MFN elsewhere)
- Chose: per-code lower rate, averaged: preferential HS12 rate = MFN mean − (sum of per-code savings) / number of codes
- Why: matches the ticket's definition (simple average of the lower rate); averaging separately would give a partial preference full weight.

## #48: Which partner groups are expanded
- Options: all groups (needs TRAINS beneficiary lists, which the WITS API does not serve); EU groups only
- Chose: EU groups only (EUN/918, N04, A36, M02, N63, M13, N55; A37 adds GUF, SMR; A11 adds EFTA), expanded to EU members in that data year (so the UK drops out after 2020); other groups (GSP etc., ~26% of calls) are dropped, so their members pay MFN unless they have their own row
- Why: the ticket asks for EU groups; no membership source for the others without the WITS web login. This overstates tariffs for some developing-country Exporters, never for the UK.

## #48: EU schedule membership and gap-fill ties
- Options: derive EU schedule users from the RTA records; a constant list
- Chose: constant `EU_MEMBERS` (EU-27 plus HRV from 2014 by the 1 July rule, GBR until 2020); gap-fill ties go to the earlier year; 2022–2024 use 2021's schedule and source year; `tariff_imputed` = year > 2021 for every row
- Why: simple and auditable; the UK applied the EU tariff until the end of the transition period, which the political `eu` bloc flag (left 2020) would get wrong.

## #48: Post-2021 Trade Agreements and Customs Unions
- Options: hard-code RCEP, UK–AUS, UK–NZ, UK CPTPP; derive from the policy table
- Chose: derive: any pair with a Trade Agreement in a year after 2021 and none in 2021 gets 0; any `customs_union` pair gets 0 in every year
- Why: covers the named agreements (UK–AUS/NZ show from 2023 in the real table; UK CPTPP starts Dec 2024, so not until 2025 by the 1 July rule) and any other, without a second list to maintain.

## #48: Intra-EU and gap-fill tests on the fixture world
- Options: add an EU Market and earlier years to the fixture world; hand-written TRAINS-shaped inputs in test_tariffs.py
- Chose: hand-written inputs for gap fill, HS17 and intra-EU; the fixture world (2021 schedule, GBR–JPN agreement from 2022) for carry-forward and post-2021 zeroing
- Why: an EU Market or more years would change every fixture artefact test (e.g. the notebook's 15 baseline rows). The fixture's Flows do not depend on tariffs; #49 may want a tariff effect in the Flow rule.

## #43: Where Excluded Products are dropped
- Options: filter in `features.training_rows`/`uk_grid` (rows only, `flows` untouched for totals); filter in `features.build` and the pipeline; filter at read time in `training.read_chunk`
- Chose: inside `training_rows` and `uk_grid` (via `exclusions.drop_excluded`), plus the fixture/real history; `build_features` still gets every Flow, so totals count them
- Why: every caller (fixture pipeline, real build, notebook, tests) gets the exclusion with no second code path, and aggregates cannot accidentally lose them because `flows` is never filtered.

## #43: What the real pipeline does with feature files built before the exclusion
- Options: filter v1-built files at read time; refuse them
- Chose: `features.build` records `excluded_products` (the list's entries) in features.json; `build_real_artefacts` raises "rebuild them" if it is missing or differs. `_train_and_score` also refuses a grid holding an Excluded Product.
- Why: the v1 features on disk include them; #58 rebuilds features anyway (#49/#51 add features), and a silent read-time filter would hide a stale build.

## #43: Shape of the list and of the artefact table
- Options: HS6 list in code; rules (HS2/HS4 prefixes with exceptions) expanded against the Product lookup
- Chose: `exclusions.EXCLUSIONS` = 4 rule entries (codes label, prefixes, exceptions, reason); artefact table `excluded_products.parquet` (hs6, description, sector, reason) = the rules expanded over the Product lookup; added to `artefacts._TABLES`
- Why: the ADR states the list as chapter/heading rules; HS6 rows are what Advisors search and what `/products/excluded` lists.

## #43: How search marks Excluded Products
- Options: hide them; mark them
- Chose: `/products` returns `ProductMatch` (ProductOut + `excluded`, `exclusion_reason`), via `lookups.search_products(..., excluded=)`; `ProductOut` itself is unchanged
- Why: ticket says mark rather than hide; rankings/explain embed `ProductOut` and must not grow fields that are always false there.

## #43: Where the app's Excluded Products list lives
- Options: a new page linked with `st.page_link`; an expander on the ranking pages
- Chose: `app.ui.excluded_products_list()`, an expander (static `st.table`) under the filters on both ranking pages; #46's Leaderboard should call it too
- Why: reachable from the rankings with no extra page numbering to clash with #46/#47, and AppTest-friendly (page_link needs a multipage context). `st.table` keeps `page.dataframe` meaning "the ranking" in existing tests.

## #43: Fixture Excluded Product
- Options: gold 710812; crude oil 270900
- Chose: crude oil 270900, large values, from its own RNG stream (`default_rng([seed, 1])`)
- Why: the other Products' Flows stay identical to before, so existing fixture tests keep their numbers; large values make "totals still count it" visibly move RCA.

## #45: Which seams the Gap Period tests use
- Options: confirm seams with the user (tdd skill); use the four seams in codebase-map §7
- Chose: package seam (test_pipeline, test_features, test_scenarios), API contract (test_api), Streamlit AppTest (test_app, test_app_why), notebook smoke
- Why: unattended run; these are the project's agreed seams and the ticket's acceptance criteria map onto them one to one.

## #52: Where the training scope and UK Zero Cells are applied
- Options: at feature build (`features.training_rows`); at training time (`training.preparer`)
- Chose: `training_rows(flows, rate, uk=baci.UK, exporters=scope)`: UK Zero Cells always kept at weight 1 (the default, so every caller gets the guard), out-of-scope Exporters' rows dropped; `flows` untouched so features still count them. `features.build` records `exporter_scope` and `uk_zero_sampling_rate` in features.json; `build_real_artefacts` refuses files without the scope.
- Why: same pattern as #43's exclusions; smaller feature files; one code path for fixture and real data. The scope comes from new `scope.exporter_scope(country_year, uk)`, ranking every country with a GDP (all are BACI Exporters on real data).

## #52: Where the row budget lives and how the rate is derived
- Options: rows-only constant (110 M); a VRAM model (fixed + rows x (row bytes + feature bytes x F)) calibrated on the GPU check
- Chose: the VRAM model in `training` (`row_budget(n_features)`); `fit_non_uk_rate` returns the largest multiple of 1e-4 whose expected rows plus 4 sd of the hash sample's size fit; `check_row_budget` refuses a given rate. `_train_and_score` counts the training years' rows (all years before the Baseline year, the largest matrix any step builds) and derives (`REAL_NON_UK_RATE = None`) or checks before tuning.
- Why: the spec ties the budget to "the measured rows and feature count"; the 4-sd headroom makes "never exceeds" hold for the realised hash sample, not just its expectation. Per-row cost is measured; the per-feature share is XGBoost's ELLPACK bin index (~1.5 bytes), not measured separately.

## #52: CuPy dependency
- Options: main dependency; dev group; optional extra
- Chose: `cupy-cuda13x[ctk]` in the dev group (xgboost 3.4.1 is a CUDA 13 build; no CUDA toolkit on the machine, so `[ctk]` brings the runtime libraries). Imported lazily only when device is CUDA.
- Why: the API container (`uv sync --no-dev`) never trains and stays lean; local training gets it with the normal `uv sync`.

## #52: Sector subset constant and where the GPU-check numbers are recorded
- Options: define the Sector subset in #53; define it now
- Chose: `training.SECTOR_SUBSET = ("03","22","30","39","61","85","87")` (constraints' 22/30/87 plus fish, plastics, knitted apparel, electrical machinery), reused by #53. The measured GPU check is a package constant `training.GPU_CHECK`, written into every artefact's metadata as `gpu_check` next to `row_budget`, `training_rows` and the rates; `scripts/gpu_check.py` reproduces it.
- Why: the ticket needs the subset now; the check is a one-off measurement that calibrates the budget, so it belongs with the budget constants and travels with each artefact.

## #44: How should the fixture world gain a Flagged Product and a Re-export Hub?
- Options: add a sixth scored Product and a fourth Market (grid 6 x 4); recode an existing Product and add a Market; recode an existing Product and Market.
- Chose: recode the fish Product 030420 as 711319 (gold jewellery, Flagged) with identical Flows, and add Hong Kong (HKG) as a fourth Market drawn from its own random stream, so every other Flow is unchanged (checked: the 240 non-hub, non-Excluded Flows are identical). UK grid / Baseline is now 5 x 4 = 20 (notebook smoke pin, grid-height tests and India-only control lists updated).
- Why: a scored Flagged Product and a hub Market are what the API and AppTest criteria need; recoding the Product keeps per-Product counts (e.g. 5 Blocked Opportunities for India) unchanged. HKG has no fixture Trade Agreement or Trading Bloc, so the Scenario presets behave as before (NLD would join the EU presets, SGP the CPTPP preset).

## #44: Which seams carry the #44 tests?
- Options: API contract only; API + AppTest + package.
- Chose: package (test_exclusions constants, test_features training rows / grid keep flagged rows, test_pipeline Baseline columns), API contract (test_api), Streamlit AppTest (test_app, test_app_why).
- Why: these are the seams the acceptance criteria name; tdd skill asks for confirmed seams and this is an unattended run.

## #44: Where do the constants and caveat wording live?
- Options: features.py; a new module; next to EXCLUSIONS in exclusions.py.
- Chose: exclusions.py (`FLAGGED_PRODUCT_PREFIXES`, `REEXPORT_HUBS`, `FLAGGED_PRODUCT_CAVEAT`, `REEXPORT_HUB_CAVEAT`, `flag_columns()`, `is_flagged_product()`); the app imports the caveat strings from the package.
- Why: ADR 0005 defines all three tiers together; #43 notes suggested it; one definition per constant.

## #44: How are badges shown in the ranking table?
- Options: emoji/icon columns; two boolean columns; one text "Caveats" column.
- Chose: one "Caveats" text column ("Flagged Product", "Re-export Hub", comma-joined, empty otherwise), plus a caption explaining any caveat present, and two "Hide ..." checkboxes (off by default) sent as `hide_flagged` / `hide_hubs` only when ticked. Why? shows one st.warning per caveat.
- Why: readable in the table and the CSV export, testable in AppTest, no emoji.

## #51: Which seams the tests use
- Options: confirm seams with the user (tdd skill); use the spec's agreed seams
- Chose: package leakage guards (new tests/test_relatedness_and_market_trends.py, test_features_build.py), API contract (test_api.py explain labels), real-data test (tests/test_relatedness_and_market_trends_real.py, skips without data)
- Why: unattended run; the spec's Seam 1/Seam 2 map onto the ticket's criteria one to one.

## #51: What "RCA leaves out the row's Importer" removes for Relatedness
- Options: only the Exporter's Flows to the Importer; every Flow into the Importer (Exporter's totals and world totals alike)
- Chose: every Flow into the Importer, so changing any of the target Market's imports leaves a UK row's Relatedness unchanged exactly
- Why: the ticket's guard says "the target Market's own Flows"; keeping the Market in the world totals would leave a (tiny) path. Cost: a dense per-Market computation (about 70 s per real year of training rows, vs a few seconds before).

## #51: The v1 test "no feature is built from the pair's trade in other Products"
- Options: keep it for every feature; exempt Relatedness
- Chose: exempt Relatedness, with a docstring saying why
- Why: the test moves UK exports from Japan to Canada; Relatedness reads the UK's exports to every other Market (so it moves) but never reads Japan's Flows at all, so it cannot carry the pair's trade. The Market-leave-out guard covers it.

## #51: Proximity definition and fitting years
- Options: per-year proximity; pooled over training years; fitted per training window (tuning / test / Baseline)
- Chose: Hidalgo et al. (2007) proximity, min of the two conditional probabilities of RCA > 1, pooled over Exporter-years of `features.PROXIMITY_YEARS` = 2012–2021 (fixture: 2021), one table for every feature year; saved as `features_<v>/proximity.parquet`, years recorded in features.json `proximity_years`
- Why: 2012–2021 is before the validation year (2022), so neither validation, test nor Baseline data reach it, with one feature build serving all windows. Proximity keeps Excluded Products (totals count them, ADR 0005). A Product is not its own neighbour.

## #51: Market import growth and re-export ratio details
- Options: ln vs log1p; null vs value at zeros; cap level
- Chose: growth = log1p(demand_loo[t]) − log1p(demand_loo[t−3]) (the existing `log_demand_loo` scale), missing when `flows` holds no Flows for t−3; re-export = log1p(min(Market's exports / demand_loo, 10)), = log1p(10) when imports from others are 0 but exports are not, missing when both are 0
- Why: log1p avoids −inf for Markets with no imports three years earlier; a cap of 10× separates producers from re-exporters without letting near-zero imports explode the value; 0/0 is honestly unknown.

## #51: How build_features gets the earlier year
- Options: new parameter; `flows` may hold extra years
- Chose: `flows` may hold several years (aggregates join on the row's year); `features.build` passes year t and t−3, the fixture pipeline passes all fixture years
- Why: keeps `build_features(rows, flows, sources)` unchanged for siblings; aggregates were already keyed by year.

## #51: Fixture world
- Options: make fixture Markets export (so the re-export ratio varies); leave Flows unchanged
- Chose: leave Flows unchanged (fixture re-export ratio is 0 for every row); `World.sources` adds proximity fitted on `fixture.PROXIMITY_YEARS = [2021]`; worked examples on hand-built Flows test the ratio
- Why: Markets that export would change Exporter counts, pairs and every fixture number, and #44 is adding a hub Market concurrently; hand-built Flows give exact expected values.

## #51: Stale feature files
- Options: let training fail on missing columns; refuse in build_real_artefacts
- Chose: `build_real_artefacts` refuses feature files whose features.json `features` differs from `FEATURES` ("rebuild them")
- Why: matches #43's refusal; the v1 files lack the new features and #58 rebuilds.

## orchestrator: New v2 notebook or update the existing one?
- Options: update notebooks/exports_predictor.ipynb; create a separate v2 notebook
- Chose: update the existing notebook (pending any answer from the user, who was asked mid-run)
- Why: the spec and #63 speak of "the notebook" as the v2 write-up, and the smoke test runs that file. v1's version remains at the local `v1` tag.

## #45: How pairs missing from one Gap Period year's grid get that year's features
- Options: build the union in the pipeline from saved grids (cannot: a missing pair has no features); build every Gap Period grid over the union in `features.build`; write separate gap grids
- Chose: `features.uk_grid(flows, uk, pairs=)` + `features.grid_pairs(grids)`; `features.build` writes `uk_grid_<y>` for 2022–2024 over the union and records `gap_years` in features.json; `build_real_artefacts` refuses files without it
- Why: one code path for fixture and real, no duplicate files; real features are rebuilt in #58 anyway. Side effect: the 2023 test grid is the union grid too (adds a few $0 pairs whose Product nobody traded in 2023); negligible for the evaluation.

## #45: Baseline table column names
- Options: rename the means (e.g. `gap_period_export_gap_usd`); keep `export_potential_usd`/`actual_exports_usd`/`export_gap_usd` as the Gap Period means and add `baseline_export_potential_usd` for 2024
- Chose: keep the names as the means, add `baseline_export_potential_usd`
- Why: every ranking, API field and CSV already shows these as the Advisor-facing values; only Scenarios, Blocked Opportunity values and the Blocked flag (`rankings.is_blocked`) move to the 2024 column.

## #45: History table shape
- Options: separate yearly-score table; extend `history`
- Chose: `history` = hs6, iso3, year, actual_exports_usd, export_potential_usd, export_gap_usd; every pair in each Gap Period year (incl. $0), earlier years non-zero Actual Exports only with null scores
- Why: the API contract (yearly gaps average to the Export Gap) needs all three years per pair; no new artefact table. Real size ~3.5 M rows.

## #45: Evaluation stability input
- Options: keep passing the Baseline table (now 3-year means); pass the 2024 yearly scores
- Chose: `gap_period.baseline_year_scores(history, baseline_year)` (2024 scores) into `evaluation.evaluate`
- Why: stability compares two single years scored by the same models; a 3-year mean against one year would overstate it.

## #45: App labels
- Options: "Export Gap (USD, nominal)" + caption; "Export Gap, 2022–2024 average (USD, nominal)"
- Chose: the latter for Export Potential, Actual Exports, Export Gap (from the API's `gap_years` via `ui.gap_period_label`); Blocked Opportunity value keeps "(USD, nominal)". Why? page: "Year by year" table (Year, Export Potential, Actual Exports, Export Gap), means + the 2024 Export Potential the SHAP factors explain
- Why: ticket wording; the label is derived, never hard-coded.

## #52: VRAM budget level, from the GPU check
- Options: 8 GiB above idle; 7.5 GiB above idle; a rows-only 110 M constant
- Chose: 7.5 GiB above idle (`VRAM_BUDGET_BYTES`), with 300 MiB fixed plus 72.9 bytes a row at 22 features. That gives 106.2 M rows, and a derived real non-UK rate of 0.63 (22 features). The GPU check ran 107.5 M rows at a peak of 7.67 GiB above idle, 0.29 s a round.
- Why: the idle desktop holds about 1.9 of the 10 GiB, so 8 GiB would leave about 100 MiB. 7.5 GiB is close to the spec's "about 110 M" and keeps about 0.5 GiB of headroom.

## #46: Where Relative Export Gap and Opportunity Type are computed
- Options: at query time in rankings; as Baseline columns in the pipeline
- Chose: Baseline columns `relative_export_gap` and `opportunity_type`, added by `gap_period.with_opportunity_columns` in `pipeline._train_and_score`
- Why: the spec lists both as Baseline table columns; computed once, served instantly. Opportunity Type reads the Gap Period mean Actual Exports (> 0 means the UK exported in at least one Gap Period year), and "any Market" is read from the Product's own UK grid rows (the grid holds every Importer).

## #46: Opportunity Type values and Relative Export Gap edge cases
- Options: codes (expand/new_market/new_product) or the glossary names; relative gap at zero Export Potential
- Chose: the glossary names "Expand", "New Market", "New Product" as stored values and API filter values (`gap_period.OPPORTUNITY_TYPES`); `relative_export_gap` is null where Export Potential is 0 and such pairs are never ranked by `relative`. "Above `min_potential_usd`" is implemented as >=, so the $0 option keeps every pair with a defined ratio
- Why: one definition, readable in CSVs and the API; no division by zero.

## #46: Leaderboard API shape and filters
- Options: separate query for the Leaderboard; generalise `rank_opportunities`
- Chose: `rank_opportunities` with no hs6/iso3 is the Leaderboard (one entry point); `rankings.summarise` sizes the filtered rows before `limit`. `RankOrder` gained `ties` and `min_potential`; `MIN_POTENTIAL_USD_OPTIONS`/`DEFAULT_MIN_POTENTIAL_USD` live in rankings and are served on `/opportunities/rank-by` (only for `relative`). `min_potential_usd` is ignored by other orders. `market` is a Leaderboard-only filter (`LeaderboardQuery`); `opportunity_type`, `trade_agreement`, `min_potential_usd` apply to every ranking. `hide_uncertain` and Potential Range fields are left to #55
- Why: coding standards (one query entry point, constants defined once, app reads them from an endpoint). `blocked` stays in RANK_BY until #47 moves it, so the Leaderboard's toggle shows it meanwhile.

## #46: Home page layout
- Options: the prototype's four variants (leaderboard, quadrant, Market briefs, heatmap)
- Chose: variant A (one Leaderboard): summary metrics, the shared `ui.ranking_filters(market_filter=True)` (toggle + min-potential selector), the shared `ui.show_ranking` table (now with Opportunity Type and Relative Export Gap columns on every ranking page), the Excluded Products expander, CSV; the top 500 rows are sent to the page and the summary covers every filtered pair
- Why: decision #31 chose the Leaderboard; reusing the shared furniture keeps all ranking pages identical. 500 rows keeps the page fast at 1.2 M pairs (query measured at ~0.2 s on a synthetic 1.2 M-pair Baseline).

## #47: Should by-product / by-market keep rank_by=blocked once the Leaderboard drops it?
- Options: keep blocked in RANK_BY for the ranking pages only; remove it from RANK_BY everywhere.
- Chose: remove it from RANK_BY everywhere (gap, relative); `rankings.BLOCKED` is a separate order used only by `/opportunities/blocked`. `rank_opportunities(rank_by="blocked")` still works in the package.
- Why: spec story 34 (Blocked Opportunities on their own page, not mixed into the tables); notes-46 and the codebase map planned it; `rank_by=blocked` on the other views now returns 422.

## #47: What summary and query does /opportunities/blocked take?
- Options: reuse the Leaderboard summary as is; extend it.
- Chose: `BlockedQuery` = `LeaderboardQuery` with `rank_by` fixed to "blocked" (anything else 422); summary = Leaderboard summary + `blocked_opportunities` (count at default thresholds) + `blocked_effect_usd` (total), via `rankings.summarise_blocked`. The response also carries `order` (title, associational description) for the page caption. All pairs are ranked (agreement pairs at 0), not only flagged ones.
- Why: "same filters, summary and row fields as /opportunities/top" kept as a superset; subclassing means #55's `hide_uncertain` on RankingQuery flows through with no extra change.

## #47: Where does the page sit in the sidebar?
- Options: renumber pages 3–5; name it 3_Blocked_Opportunities.py next to 3_Scenarios.py.
- Chose: `app/pages/3_Blocked_Opportunities.py`; Streamlit sorts by (number, label), so it lists before Scenarios as the spec's page order wants.
- Why: no renames of files siblings (#50 Scenarios) are editing.

## #47: Where do the API contract tests for the new endpoint live?
- Options: tests/test_api.py (CODING_STANDARDS); tests/test_api_scenarios.py, which already held the blocked ranking tests.
- Chose: tests/test_api_scenarios.py, replacing the old by-product/by-market blocked tests.
- Why: that file's docstring covers Blocked Opportunity rankings; keeps test_api.py (edited by #55) conflict-free.

## #55: Fold hash for the Potential Range's calibration rows (other Exporters)
- Options: a new hash helper; extend `folds.assign_folds` with a `keys` parameter
- Chose: `assign_folds(pairs, n_folds, keys=UK_PAIR_KEYS)`; calibration rows use `EXPORTER_PAIR_KEYS` = (exporter, hs6, iso3), hashing "exporter|hs6|iso3". UK hashes are unchanged ("hs6|iso3"). `folds.score_by_fold(rows, models)` predicts each row with its `fold` model (shared by `score_out_of_fold`)
- Why: one hash definition; including the Exporter spreads one pair's Exporters over the folds, as the codebase map flagged.

## #55: Sampled Zero Cells in the conformal quantiles and coverage
- Options: unweighted residual quantiles; weighted by the training `weight` (1/rate for sampled non-UK Zero Cells)
- Chose: weighted (inverted-CDF weighted quantile) for the band offsets and the coverage; band edges and the 1,000-row minimum count rows
- Why: calibration rows are the Baseline-year training rows, where Zero Cells are a 5% hash sample; weights make the percentiles estimate the full population of Flows and Zero Cells, as Horvitz–Thompson weights do for training.

## #55: Band construction and merging
- Options: quantile value edges (ties can make uneven bands); equal-count by sorted row position
- Chose: rows sorted by Export Potential, cut at positions b*n/20; while the smallest band has < min rows it merges with its smaller neighbour. Each band's lower edge is its smallest Export Potential (the first band starts at $0); UK pairs take a band by `join_asof` on the Gap Period mean Export Potential. Stored as artefact table `potential_range_bands` (band, min_export_potential_usd, rows, low_offset_usd, high_offset_usd)
- Why: exact equal counts; the merge rule is the ADR's "merges with a neighbour".

## #55: Where held-out coverage is computed and stored
- Options: notebook-only computation; in the pipeline, saved in the artefact
- Chose: the pipeline fits the served bands on all calibration rows, and separately refits on a salted-hash half (min rows halved) and scores the other half; `conformal.coverage_report` + `uncertain_share` are saved as `evaluation["potential_range"]` (so `/evaluation` serves it via `extra="allow"`, and #56/#63 can show it), settings in `metadata["potential_range"]`. The notebook prints `conformal.coverage_text` and the per-band / per-Exporter tables
- Why: the real calibration needs the fold models and the 2024 training file, which only the pipeline has; one computation serves notebook, API and model card.

## #55: Fixture size for conformal bands
- Options: add Exporters to the fixture world; keep it and set a small band minimum
- Chose: keep the world; `pipeline.FIXTURE_MIN_BAND_ROWS = 12` (54 calibration rows -> 3 bands). The held-out half (27 rows) covers 71%; tests accept 80% ± 15 points
- Why: more Exporters would shift every fixture-pinned result and collide with concurrent fixture edits (#48 tariffs). 27 rows give a binomial s.d. of about 8 points, so 71% is consistent with 80%.

## #55: Uncertain Opportunity edges and the app presentation
- Options: strict or inclusive range edges; a separate badge column or the shared Caveats column
- Chose: `uncertain` = Export Gap > 0 and low <= Actual Exports <= high (inclusive; a $0 lower bound makes New Market/Product pairs uncertain, as the spec says). In the app it joins `ui.CAVEATS` (badge in "Caveats", warning on Why?, caption), plus a "Potential Range, 2022–2024 average (USD, nominal)" column of "low–high" text, a "Hide Uncertain Opportunities" checkbox (`hide_uncertain`), and a Vega-Lite range bar on Why? with `conformal.potential_range_text` wording
- Why: reuses #44's caveat furniture with the smallest change to shared ranking code.

## #48: A country listed under two WITS reporter codes in one year
- Options: fetch both into one file; keep one entry
- Chose: keep the entry with the most partners (Sudan 2013 and 2017: 736 over 729, which has MFN only)
- Why: the cache is one file per ISO3-year; the fuller entry holds the same MFN plus preferences.

## #48: Download concurrency
- Options: 4 workers (~2.5 h); 8 workers
- Chose: restarted at 8 workers after 15 min (resume skipped the 218 cached reporter-years); finished in about 1.5 h, 62 MB of Parquet for 1,291 ISO3-years
- Why: still modest load on a free API; the run was time-bound.

## #49: Name and type of the Applied Tariff feature
- Options: `applied_tariff` (percent, as #48 returns it); `log_applied_tariff` = ln(1 + t/100)
- Chose: `log_applied_tariff` = log1p(applied_tariff / 100), Float32, in `features.TARIFF_FEATURES`, appended to POLICY_FEATURES (so last in FEATURES), in CLIPPED_FEATURES, constraint -1. `policy.POLICY_FEATURES` stays the binary flags (+1)
- Why: ADR 0006 defines the feature as ln(1+t); a separate column name avoids confusion with #48's percent `applied_tariff`. Scenarios (#50) need it among the policy columns that may change.

## #49: Which tariff columns ride along as row metadata
- Options: `tariff_imputed` only; `tariff_imputed` and `mfn_tariff`
- Chose: `features.TARIFF_METADATA` = tariff_imputed, mfn_tariff, written to every training and UK grid file, never in FEATURES
- Why: #50 restores the 2021 MFN rate when a Scenario removes an agreement; carrying it now saves a second ~20 min real feature rebuild.

## #49: When the Why? page shows the carried-forward caveat
- Options: whenever the Baseline year is after 2021 (every pair); only when the pair's Baseline-year tariff is also known
- Chose: `Explanation.tariff_caveat` = `tariffs.TARIFF_CARRIED_FORWARD_CAVEAT` when baseline `tariff_imputed` and the tariff input is not missing; otherwise None. Not a ranking badge (`ui.CAVEATS`), since it would mark every row
- Why: a pair with no tariff schedule has nothing carried forward. The caveat text also says that post-2021 agreements are set to 0, so the zeroed pairs are covered.

## #49: Where the history keeps the mark
- Options: Gap Period years only; every history year
- Chose: artefact `history.tariff_imputed` for every row (Gap Period years from the scored grid; earlier years `year > LAST_TRAINS_YEAR`), served as `HistoryOut.tariff_imputed`; Why? year-by-year table gains an "Applied Tariff" column ("Carried forward from 2021" / "From tariff data")
- Why: the ticket says the history keeps the mark; Advisors see per year which rates are real.

## #49: Testing that the -1 constraint has teeth
- Options: property test on fixture artefacts only; plus a model trained on rigged data
- Chose: both. The fixture Flow rule has no tariff effect, so the fixture-artefact property test passes even with +1; `test_a_model_trained_where_tariffs_go_with_more_trade_still_never_rises_with_the_tariff` rigs Flows to rise with the tariff and fails with +1 (checked by mutation)
- Why: the fixture world was left unchanged so pinned results elsewhere don't move.

## orchestrator: Who rebuilds the real feature files (stale after #43/#45/#51/#52/#49)?
- Options: #53 rebuilds them for its 2022 experiments; leave it all to #58
- Chose: #53 rebuilds the real derived features (all years it needs, through `features.build`, to the shared data dir) before its ablation and learning curve. #58 reuses them if they are still current, or rebuilds them.
- Why: #53 cannot run on real data without them, and building once avoids a second 15–20 min build.

## #53: Which seams the tests use
- Options: confirm seams with the user (tdd skill); use the spec's agreed seams
- Chose: package seam (new tests/test_experiments.py: nested zero subsampling, ablation/curve tables, the "hurts" rule on a worked example, refusal of the test year, a file-read spy on the real path over `derived_data_root`), notebook smoke (test_notebook.py checks the metadata carries both tables, all on 2022), and a real-data test (tests/test_experiments_real.py, skips without saved results)
- Why: unattended run; the acceptance criteria map onto these seams one to one.

## #53: How the learning curve varies the zero rate, which is fixed at feature-build time
- Options: rebuild the shared feature files at each of 2/5/10%; build the shared files at 10% and subsample at training time in the pipeline; build separate experiment rows (Sector subset, 2012–2022) at 10% and cut them to 2%/5% with the same hash rule
- Chose: the last. `features.subsample_zeros(rows, built_rate, rate, seed)` uses the same hash expression as `training_rows` (now shared, `_zero_cell_kept`), so samples nest and 10% cut to 5% gives exactly the 5% build's rows (tested). `experiments.build` writes `derived/experiments_V202601/` (training_2012..2022 + uk_grid_2022, subset Sectors only, features from every Flow). The ablation uses the same rows cut to 5%. The shared full build stays at `ZERO_SAMPLING_RATE` for #58.
- Why: one extra CPU build of the subset (~15 min) instead of three full rebuilds; no pipeline change for #58; the experiment rows are self-contained and never read 2023 (each Flows file is read on its own).

## #53: Ablation and learning-curve protocol details
- Options: fixed 1,491 rounds vs early stopping; which rows the deviance is measured on; which rates the ablation uses
- Chose: v1's winning params (`experiments.V1_CONFIG`) with v1's cap (1,500 rounds) and early stopping after 30 on 2022, as v1's own tuning did. Fold 0 of 5 held out, Sectors `training.SECTOR_SUBSET`, FeatureTransform fitted on 2012–2021 subset rows. Validation deviance = weighted Tweedie deviance (p = 1.5) of the 2022 rows (other Exporters at the built 10% zero rate, unsampled, plus UK rows of the held-out fold), the same fixed set for every model. UK Spearman = within-Product Spearman (>= 3 rows per Product) of the held-out fold's 2022 UK grid. The ablation runs at ADR 0004's starting rates (non-UK 0.65, zeros 5%) and drops each of relatedness, market_import_growth, reexport_ratio and log_applied_tariff in turn.
- Why: one fixed validation set makes every point comparable; early stopping matches how v1 chose its rounds; ADR 0001 keeps the UK measure out-of-fold.

## #53: What "hurts" means in the ablation
- Options: worse on deviance alone; worse on either measure; worse on both
- Chose: a feature hurts when the model without it is better on both 2022 deviance and UK Spearman (`experiments.kept_features`); the Applied Tariff is kept regardless
- Why: the ticket keeps each feature "unless it hurts"; requiring both measures to agree avoids dropping a feature on one-fold noise in a single measure.

## #53: Where the results live and how they reach the metadata
- Options: only in the notebook; a results file the pipeline copies into metadata; rerun inside every build
- Chose: `python -m exportspredictor.experiments build|ablation|curve` saves `derived/experiments_V202601/results.json`; `build_real_artefacts` records it as metadata `experiments` (None if absent); `build_artefacts(world, experiment_results=)` records the fixture run the notebook makes. The notebook reads the saved file in real mode.
- Why: the real runs take about an hour of GPU, too long to repeat in every build or notebook run; the file keeps the evidence next to the data it was measured on.

## #53: Notebook placement and the chart
- Options: new numbered sections (renumbering 5–11); `###` subsections at the end of section 4
- Chose: two `###` subsections at the end of section 4 (before its Conclusion, which gains one sentence); the curve is drawn with polars' `.plot` (altair, added to the dev group; streamlit already pulled it in)
- Why: renumbering would clash with sibling notebook edits and with test_notebook's "## 5." lookup; no plotting library was declared.

## #50: Which test seams (tdd skill asks to confirm them)?
- Options: confirm with the user; use the ticket's acceptance criteria.
- Chose: the ticket's seams: package (tests/test_scenarios.py: coupling, clipping, leakage guard over every Scenario kind), API contract (tests/test_api_scenarios.py), Streamlit AppTest (tests/test_app_scenarios.py, test_app_blocked.py).
- Why: unattended run; the acceptance criteria name exactly these seams.

## #50: Which pairs get their tariff changed when an agreement is added or removed?
- Options: every pair of the named Markets; only pairs where the agreement flag really flips.
- Chose: only where it flips. Add: tariff 0 where trade_agreement was 0 (custom, CPTPP preset, Blocked values); EU re-join and EU Customs Union only set 0 for every EU member (a Customs Union is tariff-free in training). Remove/Leave: MFN where trade_agreement or customs_union was 1.
- Why: "adding" an existing agreement must change nothing (the existing changes-nothing summary), and "removing" a non-existent one must not raise a tariff.

## #50: What does "restore the 2021 MFN rate" do when MFN is unknown or below the current tariff?
- Options: set MFN as is (null where unknown); max(current, MFN), keeping the current tariff when MFN is null.
- Chose: max(current tariff, ln(1 + MFN/100)); null MFN keeps the current tariff.
- Why: MFN is the ceiling of agreement rates, so this equals MFN in practice, and it guarantees removal never lowers a tariff (so never raises Export Potential under the -1 constraint). A null would send the pair down XGBoost's default branch, which is not monotone-safe.

## #50: Can a tariff Scenario be combined with a preset or agreement changes, and how are Sectors validated?
- Options: combinable; exclusive. Sectors against the products table in the API; against the Baseline's UK pairs in the package.
- Chose: exactly one kind per Scenario (preset, agreements, tariff); Sectors validated in the package against the HS2 chapters of the Baseline's UK pairs (unknown -> InvalidScenario -> 422). Negative/NaN rate and empty markets are 422 at the schema.
- Why: matches the existing "one kind" rule and keeps one validation place; a Sector with no UK pairs could change nothing anyway.

## #50: What counts in clipped_pairs, and what else does the response gain?
- Options: count only tariff-Scenario pairs; count every Scenario-set tariff beyond the training range. Expose tariffs or not.
- Chose: every pair whose Scenario-set tariff (tariff Scenario, or a restored MFN) lay beyond the range, counted before the unchanged rows are dropped. PairEffect gains baseline_tariff_pct and scenario_tariff_pct (percent, from ln(1+t)); the page shows them in pair tables.
- Why: one honest rule; the acceptance test "zero tariffs" needs tariffs observable through the API, and Advisors need to see the rate the effect is for.

## #56: The imputed/observed split when every test-year row is imputed
- Options: split by `tariff_imputed` as specified (observed side empty on 2023); redefine "imputed" (e.g. gap-filled or post-2021-zeroed rows) so both sides have rows; also evaluate an earlier, observed year
- Chose: split by `tariff_imputed` exactly, always reporting both sides (`by_tariff`: observed, imputed; an empty side has rows 0 and null metrics), plus `tariff_split_text` saying plainly that every 2023 row is carried forward
- Why: `tariffs.applied_tariffs` marks every year after 2021 as imputed, so the 2023 test year has no observed rows (fixture and real alike). Redefining the mark or reading other years goes beyond the ticket and would blur ADR 0006's meaning; the honest empty side makes the cost of carrying forward explicit.

## #56: Where Excluded Products are dropped from the metrics
- Options: rely on training rows/grids already dropping them; also drop them inside `evaluation.accuracy`
- Chose: `accuracy` calls `exclusions.drop_excluded` itself, so every metric row (overall, Sector, region, income, tariff split) is guarded
- Why: one choke point every metric passes through; cheap, and it holds even if a caller passes unfiltered rows (e.g. the v1 comparison).

## #56: Event Backtest "off" state where the MFN rate is missing
- Options: leave the tariff null; keep the row's own tariff; drop the row
- Chose: keep the row's own Applied Tariff (only the flag changes)
- Why: the state changes as little as possible where nothing is known; "on" always zeroes the tariff (ADR 0006).

## #56: Potential Range coverage text on the Model Evaluation page
- Options: compute the sentence in the app; add `text` to `evaluation["potential_range"]` from `conformal.coverage_text`
- Chose: pipeline adds `text` (one line in `_evaluate`); the page shows it with a per-Exporter coverage table
- Why: the app talks only to the API and should not re-derive package wording.

## #53: test_freeze_real compares the frozen v1 tables with the live feature files, which the v2 rebuild replaced
- Options: leave the two tests failing; skip them once the files are no longer v1's; check the frozen tables against their own manifest
- Chose: (per the orchestrator's note) the manifest: the UK grid's row count equals the manifest's, pairs are unique and folds match `assign_folds`; the test rows' count equals the manifest's, all are 2023, Flows weigh 1 and Zero Cells 50. `freeze.verify` still checks every checksum. The frozen tables are untouched.
- Why: the spec says v1 is never re-loaded after the freeze, and the v2 rebuild overwrites v1's feature files by design.

## #57: Which pairs, weights and yearly values make up a Gap-Period Lift pair, given Zero Cells are sampled per year?
- Options: pairs with a row in every gap year only; the union of the gap years' rows; the last gap year's (2017) rows with their weights, earlier years filled in where present
- Chose: the 2017 rows and weights are the pairs (unbiased Horvitz–Thompson, as v1). Mean Actual Exports is over all three years, a missing row counting as $0 (every positive Flow is a row). Mean Export Potential is over the years the pair has a row. So the Export Gap is exactly the mean of the three yearly gaps whenever all three rows exist. `backtests.gap_period_pairs` takes rows shaped like the frozen v1 `lift` table, so #60 scores v1 with the same function
- Why: training rows hash-sample Zero Cells per year (the key includes the year), so most sampled 2017 zeros have no 2015/2016 row; requiring all three years would drop most zeros and break the weights, and a union would need inclusion probabilities that vary by pair.

## #57: Where the held-out Exporter list comes from at build time
- Options: keep hashing a share of Exporters; read v1's frozen `lift_held_out_exporters` table
- Chose: `freeze.lift_held_out_exporters(folder)` reads it (checked against the manifest) and `backtests.lift_exporters(list, scope=, uk=)` restricts it to v2's scope (error if UK listed or none in scope). `backtests.run` takes `lift_held_out=` (required). Real build: `build_real_artefacts(v1_frozen=None)` defaults to `freeze.v1_folder()` = artefacts/20261117T022550Z/v1_predictions. Fixture: `pipeline.FIXTURE_V1_LIFT_EXPORTERS = ("FRA", "MEX")` (MEX out of scope), so FRA is held out. `LiftSettings.held_out_share` and `gap_year` are gone; `gap_years` and `before` replace them, and the report has `gap_years` and `held_out_from: "v1"`.
- Why: the spec (decision #38) requires both models scored on the same Exporters; a required argument stops a caller from silently falling back to a different list.

## #57: Fixture Gap Period for Lift
- Options: extend the fixture world to more years; use a two-year Gap Period (2021–2022) in FIXTURE_BACKTESTS and test three years via `fixture_training(baseline_year=2025)`
- Chose: the latter. The fixture artefact's Lift averages 2021–2022; a package test trains on 2021–2023 and checks three-year means on the fixture, and a literal-number test checks the mean of three yearly gaps and the growth thresholds
- Why: the fixture has 2021–2024 only and the pipeline trains on years before 2024; growing the world would ripple through every pinned fixture count.

## #57: LiftSettings leakage guard
- Options: rely on `train_backtest_model(before=outcome_year)`; require training years to end with the Gap Period
- Chose: `LiftSettings` raises LeakageError if a training year is after the last gap year or the Gap Period reaches the outcome year; the Lift model trains with `before=settings.before` (last gap year + 1), and `lift_backtest` refuses a model trained after the Gap Period
- Why: the ticket's guard is "no year after 2017", stricter than "before 2023".

## #57: TDD seams (normally confirmed with the user)
- Options: as the ticket lists; add more
- Chose: the ticket's: package (`backtests.gap_period_pairs`, `lift_exporters`, `LiftSettings`, `lift_backtest` on the fixture; `freeze.lift_held_out_exporters` on the synthetic frozen folder), API contract (`/evaluation` lift fields), Streamlit AppTest (page 5)
- Why: unattended run; these are the existing seams the ticket's criteria name.

## #53: How a feature the ablation drops leaves the model
- Options: delete its computation from `build_features`; keep computing it but leave it out of FEATURES
- Chose: `features.ALL_FEATURES` (everything built, the ablation's full model), `features.ABLATED_FEATURES = ["market_import_growth"]`, `FEATURES = ALL_FEATURES minus ABLATED_FEATURES`; CLIPPED_FEATURES only those in FEATURES. The feature files still carry the column; features.json lists FEATURES, so the shared files were rebuilt again. test_api's explain test now asserts the growth label is absent.
- Why: on 2022 (fold 0, Sector subset, v1's configuration) the model without Market import growth had lower deviance (412.0 vs 416.2) and higher UK Spearman (0.612 vs 0.600), so it "hurts" under the rule. Keeping the computation keeps #51's leakage tests and makes re-adding it a one-line change if a later run disagrees. Relatedness and the re-export ratio each cost deviance when dropped (426.4, 418.7) and stay; the Applied Tariff stays by rule (416.4 without it).

## #53: The final sampling rates from the learning curve
- Options: zero rate 2% / 5% / 10%; non-UK rate fixed (e.g. 0.3) or derived from the budget
- Chose: `features.ZERO_SAMPLING_RATE = 0.05` (unchanged) and `pipeline.REAL_NON_UK_RATE = None` (the largest rate the VRAM budget allows, about 0.73 for 2012–2021 rows at 5%)
- Why: deviance still fell 6% from non-UK 0.3 to 0.65 (no plateau), so more non-UK rows help; at 0.65, 5% zeros had the lowest deviance (412.1 vs 413.6 at 2%, 413.8 at 10%) with UK Spearman within 0.003 of 2%, while 10% cost UK Spearman (~0.60 vs ~0.61) and drops the budget's rate to 0.59. UK Spearman is flat (< 0.01) across the grid. Rates unchanged means the rebuilt shared feature files stay current for #58.

## #53: Posting the result on the ticket
- Options: leave it to the orchestrator; comment on #53 (not close)
- Chose: comment on #53 with the kept features, rates and plateau verdict (an acceptance criterion); the issue stays open for the orchestrator
- Why: the ticket asks for the comment; the brief forbids only closing.

## #60: Where the v1/v2 comparison lives and who runs it
- Options: compute it in the fixture and real builds now; a module plus fixture wiring, real wiring left to #61
- Chose: new `comparison.py` (`load_frozen`, `common_keys`, `feature_rows`, `scored_rows`, `compare`, `verdict`). `pipeline._evaluate(v1=comparison.Inputs)` adds `report["v1_comparison"]`, scored by the same test-year fold models. `build_artefacts` runs it by default on the synthetic frozen v1 (`pipeline.fixture_v1_frozen`); `build_real_artefacts` passes nothing, so the real comparison (2023 read once) stays with #61.
- Why: the ticket forbids reading real 2023; v2's features on v1's real Zero Cell keys need the real feature build, which is #61's run.

## #60: Condition 2, "v2 beats the naive baseline by more than v1 did"
- Options: margins on the primary metric over the same naive on the common UK rows; v1's margin from v1's own (all-Product) evaluation
- Chose: margins on the primary metric, both against the naive baseline computed by v2 code on the identical common rows (`verdict(...)["margins_over_naive"]`)
- Why: the spec scores all three on exactly the same rows. On identical rows this reduces to the point estimate v2 > v1, which can still fail when condition 1 holds only narrowly; it is kept as its own check, as pre-registered.

## #60: What "a secondary metric worsens beyond its own interval" means
- Options: paired bootstrap of every difference; per-model intervals
- Chose: deviance and RMSE-log: the paired bootstrap (over Products, seeded) of v2 − v1 lies above 0. Size-matched Lift: v2's Lift lies below v1's 95% interval (same held-out Exporters, same seed and resamples, so the Exporter resamples coincide). Event distance: v2's |predicted − observed DiD| exceeds v1's by more than half the observed interval's width, in the event's first year only (2019 for EU–Japan); later years are shown with the event's caveat and do not count. Raw Lift is shown, never counted.
- Why: v2's per-pair Lift and event rows are not kept in the report, so only the accuracy metrics can be paired cheaply; the observed DiD interval is the event metric's own noise yardstick.

## #60: v1's frozen rows outside the common definition
- Options: use v1's frozen Lift and event rows as they are; restrict them like the common rows
- Chose: drop Excluded Products from v1's Lift and event rows and keep v1's Lift rows only for v1's held-out Exporters within v2's scope (refused if v2's Lift held out a different list). The naive baseline uses every Exporter's test-year Flows (world totals), not only in-scope ones.
- Why: like for like with v2, which never trains on or scores Excluded Products; world totals are what the naive rule describes.

## #60: The synthetic frozen v1 and writing it
- Options: freeze fixture-trained v1 models (as test_freeze does); a synthetic table generator
- Chose: `fixture.frozen_v1_tables` (Actual Exports plus a floor times seeded log-normal noise; keeps the Excluded Product, adds MEX rows outside scope, every other Zero Cell at weight 2) and `freeze.write(folder, tables, fields)` split out of `freeze.freeze`; the `v1_frozen` conftest fixture now writes the full set with manifest
- Why: the fixture build must stay seconds long; one write path means the fixture manifest is checked exactly like the real one.

## #60: TDD seams (normally confirmed with the user)
- Options: as the spec lists; more
- Chose: the spec's: package (`comparison` on the synthetic frozen folder: identical keys, seeded bootstrap, verdict rule with a case per failing condition, manifest refusal), API contract (`/evaluation` has no `v1_comparison`, via `evaluation.NOT_SERVED`), notebook smoke (the new subsection prints the verdict in fixture mode)
- Why: unattended run; these are the existing seams the ticket's criteria name.

## #54: Where the search runs and how the pipeline gets the tuned configuration
- Options: the pipeline runs the full search itself; a separate resumable CLI (`python -m exportspredictor.search`) whose saved results the real pipeline reads
- Chose: new `search.py` (Optuna TPE + Median pruner via `optuna_integration.XGBoostPruningCallback`, v1 enqueued as trial 0). `search.run(subset, everything, plan, folder)` runs stage 1, stage 2 and the learning-curve re-check, saving `data/derived/search_V202601/{study.sqlite3,results.json}` after every step. `build_real_artefacts(search_results=None)` loads the saved results (refuses with a "run python -m exportspredictor.search" message if absent); `build_artefacts` (fixture) runs a tiny `FIXTURE_SEARCH` on its own tuning years. `pipeline.tune`, `REAL_CONFIGS`, `FIXTURE_CONFIGS` and the `configs=`/`max_rounds=` arguments are gone; metadata `tuning` is replaced by `search` (the whole results: trial table, confirmation, winner, v1, guardrail, capped trials, re-check) and `round_scaling`.
- Why: the search is ~3 h on the GPU and must be resumable; the pipeline's hour-long run should not repeat it. Same pattern as #53's experiments.

## #54: The "final rates" for the search
- Options: v1's 0.3; #53's ablation 0.65; derived from the VRAM budget
- Chose: zeros at `features.ZERO_SAMPLING_RATE` (0.05, subsampled from the 10% experiment rows in stage 1), and one non-UK rate for both stages, `search.final_non_uk_rate`: the largest rate at which the full 2012–2021 training rows plus the whole (unsampled) 2022 validation year fit `training.row_budget` (the validation matrix sits on the GPU beside the training matrix in stage 2).
- Why: the pipeline derives its rate from the budget (#53), and stage 2 must fit VRAM with its validation matrix; counting 2023 (as the pipeline does) would read the test year.

## #54: Interrupted trials on resume
- Options: Optuna heartbeat + RetryFailedTrialCallback (needs a grace period to pass); mark RUNNING trials failed at start and re-enqueue their configuration
- Chose: the latter (`study.tell(number, state=FAIL)` then `enqueue_trial(params, user_attrs={retry_of, v1_config})`); the 2 h budget is the sum of finished trials' durations across runs, so a resumed run only spends what is left; stage-2 rows and the re-check are kept in results.json and skipped when done (re-check redone if the winner changes).
- Why: one worker, so any RUNNING trial at start was killed; deterministic and testable without waiting for a heartbeat grace period.

## #54: Spearman guardrail against v1's configuration
- Options: retrain v1's configuration in stage 2 as a 4th model; compare on stage 1 where v1 is trial 0
- Chose: stage 1: `spearman_vs_v1` = winner's stage-1 UK Spearman − trial 0's; `loses_to_v1` if below −0.01. Stage-2 Spearman of the confirmed models is also logged.
- Why: the ticket budgets 3 confirmation models; both stage-1 numbers come from the same rows, fold and year.

## #54: Round scaling for the fold models (and test/backtest models)
- Options: scale only fold models; scale each model family by its own row ratio
- Chose: `metadata["rounds"]` = `search.scaled_rounds(tuned, training_rows, tuning_rows)` with rows = expected rows at the pipeline's non-UK rate for 2012–2023 vs 2012–2021 (counted by the pipeline); the test-year models, backtests and constraint comparison keep using `metadata["rounds"]` as before. `round_scaling` in metadata records tuned rounds, years, rows, ratio.
- Why: the spec names the fold models' ratio; the other families already reused the fold models' rounds, and per-family scaling is scope creep.

## #54: optuna as a dependency
- Options: main dependencies; dev group
- Chose: dev group (like CuPy), imported inside `search` functions only
- Why: the API container never trains.

## #54: TDD seams (normally confirmed with the user)
- Options: as the ticket lists; more
- Chose: package `search` on the fixture (trial 0 is v1, trial table fields, winner = best confirmed, kill-and-resume via a left-RUNNING trial, round scaling literal), pipeline metadata (`search`, `round_scaling`, rounds = scaled), real-pipeline refusal without a saved search, notebook smoke (existing test_notebook)
- Why: unattended run; these are the ticket's acceptance criteria.

## #54: The fixture search and fixture-model-sensitive tests
- Options: run the fixture search on the real `SPACE` (depth 4–12, min_child_weight to 1000) and re-pin the API/app/notebook tests that assume fixture outcomes; give the fixture search its own small space
- Chose: `search.FIXTURE_SPACE` (depth 2–3, min_child_weight 1–10, the rest as `SPACE`), eta 0.2, 4 trials, 100-round cap, 1 XGBoost thread; `SearchPlan.space` (None = `SPACE`). v1's trial 0 (depth 8) is still enqueued first (Optuna warns it is outside the fixture space). Fixture training rows and the experiments' training chunks are sorted by KEYS so `subsample` draws reproducible rows.
- Why: deep trees on ~70 rows gave fixture models whose outcomes (coverage ~0.6, all Opportunities uncertain, no positive JPN gap, no size-matched Lift) broke 4 existing fixture tests; `build_features` row order is not deterministic, so with subsample < 1 the fixture model changed run to run.

## #58: How the real run survives a failure late in a multi-hour build
- Options: run `build_real_artefacts` as it was (any crash after the fold models retrains everything); cache fold models and test-year models under the data root
- Chose: `_train_and_score(cache=)` saves the Baseline fold models (as `evaluation.TestModels`, plus `training.json` with the training seconds) under `data/derived/fold_models/<features.json hash>/<baseline year>_<settings hash>`; `_evaluate(test_models_cache=)` saves the test-year models at `evaluation.test_models_dir(root, metadata)`, where `evaluation.real_test_models` already looks. A re-run reads both. Backtest models were already cached.
- Why: the first real run of a lot of new code is likely to hit a late bug; ~2 h of GPU training should not be repeated for it. #61 also reuses the cached test-year models for the v1 comparison and ladder.

## #58: Who trains on real data, the notebook or the pipeline CLI
- Options: the notebook calls `build_real_artefacts` (as v1 did) and saves; the pipeline CLI trains and saves, and the notebook reads the newest artefact in real mode
- Chose: the CLI trains (`python -m exportspredictor.pipeline`, now its own `scripts/rebuild.py` step, with the experiments and search steps before it); the notebook's real mode does `artefacts.load_latest(artefacts_dir())`, and its save cell saves only in fixture mode.
- Why: the ticket says the notebook "reads the results through the package"; one training run, one artefact version, and the run can go in the background with a log per environment.md.

## #58: Spec metadata "thresholds" and "training dates"
- Options: leave them implicit in code constants; record them
- Chose: new metadata `thresholds` {blocked_min_share, blocked_min_usd, relative_gap_min_potential_usd, potential_range_coverage, potential_range_bands, potential_range_min_band_rows}. "Training dates" are covered by `created_at` plus `training_years`/`tuning_years`/`validation_year`/`test_year`.
- Why: the only spec *Artefacts* metadata item not yet recorded.

## #58: A cheap real smoke run before the full build
- Options: go straight to the multi-hour build; run the whole real pipeline first with the tuned config cut to 3 rounds and non-UK rate 0.05, saved to a scratch folder
- Chose: the smoke run. It reads 2023 through a throwaway 4-round model; nothing is chosen from its numbers and they are not looked at beyond "it ran".
- Why: catches real-data-only bugs (shapes, memory, missing columns) in minutes rather than after hours of GPU time.

## #58: "SHAP background data" in the artefact (spec *Artefacts*)
- Options: add a background sample table; record none
- Chose: none. Explanations use XGBoost's exact TreeSHAP (`pred_contribs`, path-dependent), which needs no background data; the fold models and `baseline_features` are all `explain` reads.
- Why: a table nothing reads would only add size; noted for #59/#63 in notes-58.md.

## #58: Executing the whole notebook in real mode
- Options: run `jupyter execute` with FIXTURE_MODE=0 to check every real cell; exercise only the real-mode cells this ticket changed against the artefact
- Chose: the latter. Real mode re-runs gravity/policy/feature builds on the shared data dir (features ~15 min, ~14 GB RAM, while siblings run) and writes `model_card.md` into the newest artefact folder, which is #63's governance step.
- Why: the side effects reach shared data and the artefact; the fixture notebook test covers structure, and the changed real cells were run against `20261118T042154Z` by a script.

## #58: The headline does not beat the naive baseline on real 2023
- Options: tune or re-run to chase the headline; report it as is
- Chose: report as is (UK within-Product Spearman 0.511 vs naive 0.523); the served headline text already states it, and the notebook and the #58 comment say it plainly.
- Why: 2023 is the test year, read once; changing anything after seeing it would be test-set tuning. #61's ladder will attribute the change against v1.

## #61: Test seams (TDD asks to confirm them with the user)
- Options: confirm with the user; take the existing seams
- Chose: the package seam (`comparison.ladder` on the fixture's frozen v1; rung models trained on a feature subset via `evaluation.train_test_models(features=)`), the fixture artefact's `v1_comparison` block, the new-version writer on a saved fixture artefact, the model card, and the existing notebook smoke test.
- Why: unattended run; these are the seams the codebase already tests the comparison at.

## #61: Configuration of ladder rungs A and B
- Options: v1's params and v1's round count (1491); re-early-stop v1's params on 2022 with v2's rows
- Chose: v1's params (`experiments.V1_CONFIG`) and v1's 1,491 rounds, v1's 22 features (A) or v2's 25 features (B), on v2's feature files, scope, zero rate and derived non-UK rate (0.5894), years 2012–2022, seed 0, 5 folds.
- Why: "v1's configuration" literally; re-stopping would be a new tuning step. The round count is part of the configuration.

## #62: Test seams (TDD asks to confirm them with the user)
- Options: confirm with the user; choose the package seams
- Chose: `ppml.fit_ppml` checked against an independent dense dummy-variable Poisson Newton fit with its own pair-clustered sandwich; `ppml.sector_flows` on a hand-built Flow table; `ppml.model_agreement_effect` and `ppml.comparison_table` on hand-worked numbers; the fixture world end to end; the existing notebook smoke run.
- Why: unattended run; these are the public functions the notebook calls.

## #62: PPML estimator
- Options: pyfixest `fepois` (new dependency: numba, formulaic, etc.); a dense GLM (statsmodels: infeasible with about 1 M pair effects); hand-rolled IRLS with the fixed effects swept out by weighted alternating projections (the ppmlhdfe method)
- Chose: hand-rolled IRLS + alternating projections in NumPy (`exportspredictor.ppml.fit_ppml`), with ppmlhdfe's warm start, dropping all-zero groups and singletons, and SEs clustered by Exporter–Importer pair (G/(G−1)).
- Why: no new dependency; about 100 lines, verified exactly against a dense dummy-variable fit in the tests; the real panel (5 M rows) is a few hundred MB.

## #62: Fixed effects and data for the "3-way" PPML at Sector level
- Options: Exporter-year, Importer-year and Exporter–Importer effects pooled over Sectors (algebraically the same coefficient as aggregate gravity); the same three effects each within a Sector (Exporter×Sector×year, Importer×Sector×year, Exporter×Importer×Sector)
- Chose: within-Sector effects. Data: non-Excluded Products, the model's top-100 Exporter scope with every Importer, self-trade dropped, 3-year intervals 2012/2015/2018/2021/2024, every ever-traded Exporter–Importer–Sector cell in every year (zeros included, the dense grid never built). The Trade Agreement dummy is the model's own `trade_agreement` (WTO RTA-IS, 1 July rule). Customs unions are not a separate regressor.
- Why: pooling Sectors under country-level effects makes the Sector aggregation pointless (Poisson sums within cells). Intervals follow the Advanced Guide. Same scope and agreement definition as the model, so the comparison is like for like.

## #62: What "the model's mean Trade Agreement Scenario Effect" is
- Options: re-run a Scenario over all Markets; read the saved Blocked Opportunity values (`blocked_effect_usd`: add TA + zero tariff, out-of-fold, 2024 Baseline) over every UK pair without an agreement
- Chose: the saved values, reported Baseline-weighted (sum effect / sum Baseline, the PPML-comparable number) and as the mean and median per pair.
- Why: no prediction or model loading is needed; it is exactly the artefact's served "add an agreement" answer.

## #62: Literature range
- Options: a single headline number; a range of cited estimates
- Chose: six estimates, from Yotov et al. 2016 PPML without pair effects (0.191, +21%) to Baier & Bergstrand 2007 (about a doubling, ln 2). Figures checked in the PDFs; sources in `docs/research/rta-effects.md`. Phased-in totals (0.99) are left out because the benchmark has only a contemporaneous dummy.
- Why: the specification differences (pair effects, intra-national flows, phasing) move the estimate a lot; a sourced range is honest about that.

## #62: Where the notebook section goes
- Options: a new top-level section; a subsection of section 6 (Scenarios) after the monotone-constraint comparison
- Chose: `### A gravity benchmark for the size of the Trade Agreement Scenario Effect` in section 6, plus one sentence in section 6's Conclusion.
- Why: it is about Scenario size; it keeps clear of #61's edits in section 7 and avoids renumbering sections.

## #61: How `v1_comparison` gets into the real artefact
- Options: update `20261118T042154Z/evaluation.json` in place; write a new format-2 version folder that copies it file for file with the block added; retrain via a full `build_real_artefacts`
- Chose: a new version (`pipeline.add_v1_comparison`, CLI `python -m exportspredictor.pipeline compare`), via `artefacts.copy_version`; v2's cached 2023 test models (rung C) are reused and the ladder rungs are cached under `data/derived/ladder_models/`. `build_real_artefacts` now also runs the comparison (with the ladder) so a full rebuild produces the same block.
- Why: existing artefact folders are never modified (the brief), and this is the established pattern (`evaluation.add_to_latest`, `backtests.add_to_latest`, `refresh`). The new folder is the lexically newest, so the API serves it; the block is stripped from `/evaluation` by `NOT_SERVED`.

## #61: A real smoke run before the GPU ladder
- Options: start the multi-hour run directly; run the real comparison first with a throwaway 3-round ladder (no cache, nothing saved)
- Chose: the smoke run, then rungs A/B trained by a separate script into the ladder cache while the smoke finished scoring on CPU, then `pipeline compare` reads them.
- Why: catches real-data bugs before GPU hours; nothing is chosen from the smoke's numbers.

## #59: How `/health` returns "format 2"
- Options: infer it from a successful load (only format 2 loads); add an explicit field
- Chose: `Health.artefact_format`, read from the served metadata (fixture test updated to expect 2); `/evaluation` does not carry it.
- Why: the ticket asks `/health` to return it, and an explicit field tells an operator which layout is served without reading code.

## #59: Where the real-data serving tests go and what "Exporter scope matches the metadata" compares
- Options: extend test_scenarios_real.py; a new tests/test_api_real.py through the API on the real root
- Chose: new tests/test_api_real.py (TestClient on `artefacts_dir()`, loads once via the API, skips when nothing format 2 is served or the artefacts are fixture ones): `/health` format 2 + gap years; every Excluded Product 404s with its reason and none appears in the served Baseline, history or Leaderboard for Sectors 27/71/97; the served `exporter_scope` has 100 Exporters incl. GBR and equals both `scope.exporter_scope(derived/country_year)` and `features.json`'s scope. The container check is a real-root healthcheck test in test_containers.py.
- Why: the API is the main seam; the scope check needs an independent source (the derived GDP table), not the metadata against itself.

## #59: Running `podman compose` from a worktree
- Options: run it from the main checkout (old code); a junction `./artefacts`; an untracked override file with the absolute artefacts path
- Chose: `-f compose.yaml -f <scratchpad>/compose.real.yaml` mapping `C:/CodingProjectsC/exportspredictor/artefacts:/artefacts:ro`; not committed.
- Why: worktrees have no ./artefacts; the override changes nothing in the repo and serves this branch's code.

## #59: API-down page hung for about a minute in the containers
- Options: shorter httpx connect timeout in app/client.py; `dns_search: [dns.podman]` for the app service; a podman-only URL `http://api.dns.podman:8000`
- Chose: `dns_search: [dns.podman]` in compose.yaml (+ a compose-as-data test). Verified live: with the API stopped, `api` now fails in 0.06 s (was 60 s) and every page shows the message; with it running, the app reaches it as before.
- Why: the root cause was the host's DNS search domain: with the API container stopped, a bare `api` resolved to a public host on the internet and the app tried to connect to it (two 30 s timeouts). A timeout alone would still send requests outside; a podman-only URL would break Docker.

## #59: Latency expectations with the depth-12, 1,453-round models
- Options: loosen `test_a_bloc_scenario_returns_within_a_few_seconds_on_cpu` (< 5 s); keep it
- Chose: keep it. Measured: EU re-join 1.4–2.1 s via the API (CPU, in process and in the container), 4.0 s end to end in the Scenarios page; Leaderboard 0.06–0.54 s per API call (median 0.26 s) in the container.
- Why: it holds with margin.

## #59: README rows outside the serving checklist that are stale for v2 (rebuild table: "the notebook trains", 30 minutes)
- Options: fix them here; leave them to #63 ("README v2")
- Chose: leave them to #63; #59 changed only the Run section and the checklist.
- Why: scope, and #63 owns the README v2 pass.

## #63: Which seams the governance and write-up tests use
- Options: test the model card through `governance.model_card`/`render_model_card` (package seam) plus the README and notebook smoke tests; test private helpers
- Chose: the package seam (card dict and rendered Markdown, `range_coverage_by_exporter`, `data_lineage`), `tests/test_readme.py` (rebuild table against `scripts/rebuild.py` STEPS) and the existing notebook smoke run. No new seam.
- Why: the existing four seams cover it; the user could not be asked (unattended).

## #63: How the model card gets the PPML benchmark, which is never saved in an artefact
- Options: hard-code +5.1% / +58% in governance; save the PPML result into an artefact; pass it in from the notebook
- Chose: `model_card(artefacts, gravity_benchmark=)` / `write_model_card(..., gravity_benchmark=)`, passed by the notebook from section 6 (`gravity_fit.as_dict()`, `model_agreement`). The literature range is computed from `ppml.LITERATURE`. Without it the card simply omits that limitation.
- Why: the ticket's point is that numbers are read, not typed in; the spec says the PPML benchmark is notebook-only and never served, so it must not enter the artefact.

## #63: How "too wide / too narrow" is judged in the Potential Range coverage by Exporter
- Options: list every Exporter; a tolerance around the 80% target
- Chose: `governance.RANGE_COVERAGE_TOLERANCE = 0.05` (5 percentage points); the card lists the first 10 of each group. Real: 34 of 99 near target, 65 too wide (all small Exporters; Spearman −0.98 with log rows), none too narrow; lowest CHN 78.6%.
- Why: a simple, stated threshold; under-coverage is the harmful direction and it is reported separately.

## #63: The notebook's real run rebuilds the shared gravity/policy/feature tables; could that invalidate the cached fold models (data/derived/fold_models/1c15443b8a7d)?
- Options: run as is; skip the rebuild cells; snapshot the derived tables, run, compare by content and restore the originals byte for byte
- Chose: snapshot → run → content comparison (row order ignored) → restore the snapshot. The cache key is the SHA-256 of features.json, which depends only on settings (years, scope, rates, seed, features, exclusions, gap years), so it cannot change; restoring also keeps the Parquet bytes the cached models were trained on.
- Why: the acceptance criterion needs the real top-to-bottom run, and this route leaves every cache exactly as it was whatever the rebuild does.

## #63: Where to state the Potential Range's fixed-dollar width
- Options: model card only; model card + notebook section 7 + conclusions; also `evaluation.LIMITATIONS` (served by the API/app)
- Chose: a computed model-card limitation (from the artefact's `potential_range_bands` top band: $2.3m, $22.9m wide, Uncertain only under the $10.3m lower offset), a bullet in section 7 and the conclusions. `evaluation.LIMITATIONS` is left alone.
- Why: the brief asks for the card and the write-up; changing served text belongs to a separate app change and the code review.

## orchestrator: How to fix negative Trade Agreement Scenario Effects on pairs with a missing tariff (code review)
- Options:
  - (A) When adding a Trade Agreement, set the Applied Tariff to 0 only where it is observed or carried forward, and leave a missing tariff missing. Then re-score the Blocked values and presets into a new artefact version, with no retraining.
  - (B) Impute missing tariffs before training and retrain every model.
  - (C) Clip negative effects at 0.
- Chose: A
- Why: Spec #39 promises non-negative Trade Agreement Scenario Effects. XGBoost's monotone constraints don't cover the missing-value branch, so a missing tariff set to 0 can lower the prediction, and all 22,135 negative pairs have a missing tariff. A keeps the monotone guarantee without retraining after the test year was read. B would re-tune the model after reading 2023. C hides the cause. A missing tariff is unknown, so "zeroing" it is not a defined policy change. The write-up states the rule.

## orchestrator: Which code-review findings the fix pass takes
- Options: (A) every hard and spec finding plus the cheap smells; (B) only the bugs.
- Chose: A, except splitting `pipeline.py` (Divergent Change) and the `RankingFilters` type (Data Clumps), which are large refactors with merge risk at the end of the run. Those two are logged as follow-ups.
- Why: the skill says fix all issues the review raises; these two are judgement-call smells whose fix is a refactor, not a defect fix.

## review: Which test seams the fix pass uses
- Options: confirm seams with the user (tdd skill); use the existing four seams (package, API contract, notebook smoke, Streamlit AppTest) plus a new source-level app check
- Chose: the existing seams, plus `tests/test_app_shape.py` (AST: no app file imports `exportspredictor`/`api`; unique page numbers) and `tests/test_app_format.py` (app formatting prints as the package's). A real-data guard was added to `tests/test_scenarios_real.py`.
- Why: unattended; these are the seams the code already tests at. No such app-import test existed on main.

## review: Where the "added agreement zeroes only a known tariff" rule lives (S1)
- Options: patch each of `_changes` and `blocked_effects`; one expression shared by all
- Chose: `scenarios.AGREEMENT_TARIFF` (0 where the tariff is known, null where null), used by custom adds, all presets, and `blocked_effects`, which is now `apply` of a Scenario adding an agreement with every Market. `backtests.agreement_states` (the Event Backtest's on state) uses it too.
- Why: one definition, so Blocked values and Scenarios cannot drift. The real artefact's stored Event Backtest was NOT recomputed (it would need the backtest feature rows, including 2023 for UK–AUS/NZ); its numbers and its `states` sentence come from the old rule. Logged as a follow-up.

## review: Rename the stored Baseline columns or map them at load time (H3)
- Options: rename in the new artefact and map old names in `artefacts.load`; keep stored names and map to new names at load only
- Chose: the new version `20261118T100524Z` stores `blocked_opportunity_value_usd` and `uncertain_opportunity`; `artefacts.load` renames `blocked_effect_usd`/`uncertain` in older format-2 folders (`_RENAMED_BASELINE_COLUMNS`). API fields, the blocked summary key and package names follow (`blocked_opportunity_values`, `with_blocked_opportunity_values`). `hide_uncertain` (a query parameter) is kept.
- Why: the new version was being written anyway; old folders stay readable.

## review: How the app gets what it imported from the package (H1, H2)
- Options: serve everything (including formatted strings) from the API; serve wording/options and keep pure formatting in the app
- Chose: `GET /labels` (USD caption and unit, Opportunity Types, the three caveats with badges, `last_tariff_year`, the carried-forward caveat); `gap_period_label` on /health and every value-carrying response; `potential_range_text` on /explain. Money and year formatting is presentation, so `app/format.py` holds the app's copies of `format_usd`/`format_change`/`years_text`/`number_text`, and a test keeps them identical to the package's. Every hard-coded 2021 in the app now reads `last_tariff_year`.
- Why: CLAUDE.md "Shape" forbids the import; wording and options have one definition in the package; a formatter cannot be served.

## review: Shape of the ranking-with-summary entry point (H5)
- Options: `rank_opportunities` returns a summary too; a thin `rankings.ranking(...) -> Ranking(rows, summary)` over it
- Chose: `rankings.ranking(artefacts, limit=, **filters)`: summarises every filtered pair (`summarise_blocked` for the BLOCKED order), then cuts to `limit`. The API's `top`/`blocked` share `_leaderboard`; `_filters` maps a query once. The Excluded Product 404 text is `exclusions.not_ranked_message`.
- Why: keeps `rank_opportunities`' return type (notebook and other callers) unchanged.

## review: Which limitations the API serves (S3)
- Options: only the fixed-width one; everything derivable from the artefact
- Chose: `evaluation.known_limitations(artefacts)` = result limitations (headline vs naive, Lift, Event Backtest, sampling) + the Potential Range fixed-width one + `LIMITATIONS`. `_result_limitations`/`_range_limitations` moved from governance to evaluation; the model card adds the PPML benchmark one, which no artefact holds, so the API does not serve it.
- Why: the panel should match the model card where it can.

## review: Shared report formatting helpers (smells)
- Options: leave the per-module `_years`/`_f`/`_number`/`_rate`; one module
- Chose: `exportspredictor.wording` (`years_text`, `number_text`, `rate_text`). `comparison._number` and `governance._number` are now thin specs over `number_text`; the model card prints a missing metric as "n/a" (was "–"). Gap Period label is `gap_period.label`.
- Why: one definition each.

## review: Notebook's Blocked cell flagged against the Gap Period mean
- Options: leave; use the Baseline year's Export Potential as everywhere else
- Chose: `is_blocked("blocked_opportunity_value_usd", "baseline_export_potential_usd")`.
- Why: found while renaming; the rule is defined against the Baseline year (rankings, API).

## review: How the new real artefact was made (S1)
- Options: `pipeline.copy_version` then patch; `scenarios.rebuild_blocked_values`
- Chose: `rebuild_blocked_values` on the newest `20261118T082751Z` (CPU, 18 s): writes `20261118T100524Z` with the same models and tables, recomputed Blocked values, `blocked_values_from` in metadata. Presets recomputed on it (CPU). `model_card.md` re-rendered with #63's PPML fit (5.1406%, −0.4427% to +11.0370%) and the recomputed model side. Scripts and logs: `spec-39-notes/logs-review/`.
- Why: the existing tool for exactly this; no training, no 2023 reads, no existing folder touched.

## orchestrator: Which worktrees to clean up at the end
- Options: remove every worktree under ep-wt; remove only this run's implementer worktrees (40–63, review)
- Chose: remove this run's implementer worktrees and their merged `ticket-*` branches. Keep `integration` (it holds the result branch) and `prototype-landing` (not created by this run).
- Why: the skill says clean up implementer worktrees. The integration branch is the deliverable, and the prototype worktree belongs to other work.

## orchestrator: Untracked files in the main checkout (`NUL`, `bash.exe.stackdump`)
- Options: delete them; leave them
- Chose: leave them and report them
- Why: they appeared during the run, apparently from tooling, but deleting files this run did not knowingly create is the user's call.
