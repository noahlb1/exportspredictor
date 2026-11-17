# Decision log for spec #10

The choices taken without the user during the unattended `/implement-spec #10` run (17 Nov 2026), on the integration branch `spec-10-integration`. Each entry gives the options considered, the one chosen and why. Choices that settled into lasting design also live in `docs/adr/`; this file is the full record, kept so any of them can be reviewed or reversed.

---

## orchestrator: how to install uv (not on this machine)
- Options: winget; Astral install script; the user installs it.
- Chose: winget (the user picked this before the decision log existed).
- Why: user-scope and verified by hash. uv then installed Python 3.12.

## orchestrator: podman is not installed
- Options: write the compose files and skip the live run; install podman.
- Chose: write the files and skip the run (the user picked this).
- Why: installing podman needs a WSL machine. The user will run `podman compose up` manually.

## orchestrator: worktree isolation fails on this machine
- Options: fix the git config; make worktrees by hand.
- Chose: `git worktree add` into `C:\CodingProjectsC\ep-wt\<ticket>` on branches `impl/<ticket>-<slug>`.
- Why: the harness rejected `c:\` vs `C:\` path casing. Making the worktrees by hand avoids that without changing the repo config.

## orchestrator: draft PR or not
- Options: open a draft PR; close issues directly.
- Chose: no PR. Each ticket is closed with a comment, per docs/agents/issue-tracker.md.
- Why: the tracker closes work through issues, and the user didn't ask for a PR.

## orchestrator: exploration subagent
- Options: run one; skip it.
- Chose: skip it.
- Why: docs/research/ already gives the source URLs for BACI, Gravity, RTA-IS and HMRC.

## #11: which seams to test (tdd skill asks to confirm seams with the user)
- Options: ask the user; take the seams from spec #10's Testing Decisions.
- Chose: the spec's seams: fixture builder, package pipeline/artefacts round trip, API contract (TestClient), Streamlit page (AppTest against a live local API), notebook smoke run.
- Why: the spec already records the agreed seams; the run is unattended.

## #11: where the fixture builder lives
- Options: under tests/ (spec wording); in the package as `exportspredictor.fixture`.
- Chose: the package, used by a session-scoped pytest fixture in tests/conftest.py.
- Why: the fixture-mode notebook (user story 40) and local dev also need it, and notebooks import logic only from the package.

## #11: where artefact folders live
- Options: under the data dir; a top-level `artefacts/` (already gitignored).
- Chose: `<repo>/artefacts`, overridable with env var `EXPORTSPREDICTOR_ARTEFACTS_DIR`; the data root stays `EXPORTSPREDICTOR_DATA_DIR`.
- Why: .gitignore already reserves `artefacts/`; compose (#23) can mount it as its own volume.

## #14: which seams to test (tdd skill asks to confirm)
- Options: test each parser/helper; test only the public functions of a new `exportspredictor.gravity` module.
- Chose: four seams in `gravity`: `map_baci_countries` (BACI code -> ISO3, overrides, unresolved list), `fill_country_year` (the GDP/population missing-value rule), `gravity_pairs` (time-invariant pair features), and `build` end to end on tiny local files with download off.
- Why: these hold the logic the acceptance criteria name; HTTP downloads are thin and checked by a real run instead.

## #14: GDP/population missing-value rule
- Options: WDI only, leave gaps null; WDI then carry nearest year; WDI -> IMF WEO -> CEPII Gravity -> interpolate/carry nearest year -> null.
- Chose: the last. Each value records its source.
- Why: WDI omits Taiwan (a top-20 trader) and some territories; IMF WEO covers Taiwan through 2024, Gravity covers some small territories to 2021. Nulls that remain are left to XGBoost's native missing handling.

## 23: test seams for containers and one-command run
- Options: confirm seams with the user (tdd skill); pick them.
- Chose: compose.yaml parsed as data, the API healthcheck command run against a live local API, the rebuild script's CLI and `run(steps)`, app pages under AppTest with the API down or not ready, and the README checklist checked against page titles.
- Why: podman cannot run here, so these are the closest observable boundaries.

## 23: how the rebuild script is written
- Options: PowerShell script; bash script; Python script with an ordered STEPS list.
- Chose: Python (`uv run python scripts/rebuild.py`), one `Step` per line, notebook step last.
- Why: works on Windows and in containers alike, and is testable; parallel tickets add one line each.

## 23: one image or two for API and app
- Options: one shared Containerfile/image; separate images per service.
- Chose: one image, with compose choosing the command per service.
- Why: both import the same package and lockfile; one build, less to maintain.

## #12: which test seams (tdd skill asks to confirm)
- Options: test BACI parsing internals; test at public seams only.
- Chose: four seams: `baci.prepare()` on a tiny synthetic BACI zip (schema, leading zeros, idempotency, lookups); the API contract for `/products` and `/markets`; the Streamlit page via AppTest; the notebook fixture smoke run. Real-data ACs (2012–2024 coverage, memory) evidenced by a run.
- Why: these match the spec's seams, and the real 1.3 GB zip can't run in the test suite.

## #12: module layout for data loading
- Options: one `exportspredictor.data` module (spec's sketch); one module per source.
- Chose: `exportspredictor.baci` (download/convert/scan, `python -m exportspredictor.baci`) and `exportspredictor.lookups` (Product search).
- Why: #13 and #14 add their own sources in parallel; per-source modules avoid merge conflicts in one shared file.

## #12: Flow storage format
- Options: keep BACI's raw columns (t,i,j,k,v,q, numeric country codes, thousand USD); normalise.
- Chose: `year` Int16, `exporter`/`importer` Enum of ISO3, `hs6` Enum of zero-padded codes, `value_usd` Float64 in USD; `q` dropped.
- Why: matches the fixture world's column names and units (value_usd, ISO3), keeps memory compact (~2.5 GB peak), and the spec says q is unused.

## #12: Market regions and "smartphone" search
- Options: regions from WDI (#14) / a static table / World Bank country API; search via description only / a curated alias list.
- Chose: World Bank country API (regions + income level, also handy for #15's high-income share), with overrides for S19 (Taiwan proxy) and R20; 19 small territories get "Unclassified". A small curated alias table maps everyday words (smartphone, laptop, car, ...) to HS6 codes.
- Why: HS 2012 descriptions never say "smartphone", so an alias table is the only way to meet that AC; World Bank regions are the standard grouping.

## #14: BACI countries missing from CEPII Gravity (BLM, ATF)
- Options: leave their pair features missing; use a nearby territory as a stand-in; drop their Flows.
- Chose: stand-ins: Saint Barthelemy -> Guadeloupe, French Southern Lands -> Reunion (`GRAVITY_PROXIES`).
- Why: both trade (about $2.6 bn over 13 years); the stand-ins share language, colonial ties and region. Easy to change in one dict.

## #13: TDD seams (normally confirmed with the user)
- Options: test only through `policy` public functions (read_rta_export, pair_agreements, bloc_membership, attach_policy, build, gravity_crosscheck); also test private parsing helpers.
- Chose: public functions only, on a committed extract of 10 real WTO records (tests/data/wto_rta_extract.csv), plus the same tests on the full export when it is in the data dir.
- Why: the acceptance criteria are behaviour on known dates; real records make the expected values independent of the code.

## #13: UK's EU Trading Bloc exit date
- Options: WTO's date (31 Dec 2020, end of transition); the political exit (31 Jan 2020).
- Chose: 31 Jan 2020 for the `eu` bloc flag only (hard-coded, cited in policy.py); WTO dates for agreements and the Customs Union.
- Why: the ticket requires "out of the EU from 2020 but in the EU Customs Union for 2020"; no WTO date column carries the political exit.

## #13: Partial-scope agreements (WTO type "PSA")
- Options: count them as Trade Agreements; exclude them.
- Chose: exclude.
- Why: they cover short product lists (e.g. GSTP); Gravity's rta_type has no pure-PSA code, so its fta_wto excludes them too. Including them cost ~13k pair-years of disagreement.

## #13: Members listed only as current signatories, with no entry date
- Options: start them at the agreement's entry into force; drop them.
- Chose: start at entry into force, except in the records that define a bloc's membership (EU Treaty, EFTA, etc.), where accession records give the real date.
- Why: otherwise Croatia would be in the EU from 1958; elsewhere (e.g. EU–OCT territories) dropping loses real members.

- Merge #13: git put the policy rebuild step before gravity; reordered to BACI → gravity → policy (policy cross-checks gravity_extract). Changed `.gitignore` `data/` → `/data/` so `tests/data/wto_rta_extract.csv` (#13 test fixture, ignored on its branch) could be committed.

## #15: revealed comparative advantage vs the two leakage rules
- Options: leave the whole target Market out of RCA (Exporter total minus its total exports to that Importer); leave out only the row's own Flow.
- Chose: leave out only the row's own Flow, from the Product, Exporter and world totals.
- Why: removing the whole Market puts the pair's trade in other Products into the feature (total bilateral trade), which the spec excludes on purpose. With only the own Flow removed, the feature never sees the target and carries nothing pair-specific; both are asserted in tests/test_features.py.

## #15: which transforms "fitted on training years only" covers
- Options: no fitted transforms (trees need none); clip ranges + Sector encoder; target encodings.
- Chose: `FeatureTransform` = per-feature clip ranges (continuous features) + the set of Sectors seen in training, fitted on training years and saved in metadata. Supply/demand aggregates are computed within each year (leave-one-out), never across years.
- Why: gives the guard a real object to test, mirrors ADR 0002's clipping, and changes no tree prediction. Target encodings would add a UK leakage path.

## #15: zero sampling rate
- Options: tune now; pick a default and let model training tune it.
- Chose: default 0.02 for real data (~1.9 M zeros/yr, about one per six non-zero Flows); 0.5 for the fixture. Relevance is defined within each year. Sampling is a hash of the cell keys and seed.
- Why: the rate is a training hyperparameter and belongs to #16's validation-year tuning; the build takes 90 s to redo with another rate.

## #16: which seams to test (tdd skill asks to confirm)
- Options: the spec's seams (package leakage guards, API contract, notebook smoke run); add more.
- Chose: the spec's seams, plus the pipeline (`build_artefacts` / `build_real_artefacts` on a tiny data root) for the Baseline table and metadata.
- Why: the spec lists them; the pipeline is the notebook's single call.

## #16: train on all ~150 M rows, or a sample of non-UK rows
- Options: all rows on GPU; XGBoost external memory; a reweighted hash sample of non-UK rows.
- Chose: keep 30% of non-UK rows (weight x 1/0.3), every UK row; `pipeline.REAL_NON_UK_RATE`, recorded as `non_uk_sampling_rate`.
- Why: all rows (152 M) built in 200 s but filled the 10 GB of VRAM at the first round and slowed to a crawl. External memory on Windows is untested here. The sample keeps the run to about an hour and is unbiased (Horvitz–Thompson, as for zeros).

## #16: what the artefact's history table holds
- Options: Actual Exports and out-of-fold Export Potential per year; Actual Exports only.
- Chose: non-zero UK Actual Exports, 2019–2024 (`history.parquet`: hs6, iso3, year, actual_exports_usd).
- Why: #19 needs the Actual Exports history; scoring earlier years with the 2012–2023 fold models would be in-sample in time. #20 can score the 2023 grid itself for its stability check.

## #16: test-year (2012–2022) models
- Options: train them here; leave them to #20.
- Chose: leave to #20. This ticket tunes on 2022 and refits on 2012–2023 for the Baseline; the test year is never read.
- Why: #20 owns the 2023 metrics and asserts the test year is untouched.

## #16: Taiwan's Market name
- Options: keep BACI's "Other Asia, nes"; show "Taiwan".
- Chose: `iso3` TWN, name "Taiwan" (`lookups.market_lookup`).
- Why: BACI's label is a statistical euphemism Advisors would not recognise.

## #19: write a new real artefact version for explanations?
- Options: write a new version with copied models (brief's default for new tables); write nothing new.
- Chose: nothing new. Explanations are computed at request time from the existing `baseline_features`, `history` and fold models.
- Why: XGBoost's TreeSHAP needs no background data, so #19 adds no table; a copy would only duplicate 154 MB. Verified on the real artefact 20261117T002435Z.

## #19: how ranking rows reach the Why? page
- Options: a LinkColumn with `/Why?hs6=..&iso3=..`; row selection plus `st.switch_page`.
- Chose: LinkColumn (shared helper `app.ui.why_link` / `why_column` for #17 to reuse).
- Why: one column per ranking table, testable with AppTest, and the link is shareable/bookmarkable.

## #18: what do the bloc presets flip, beyond the Trade Agreement and Customs Union flags?
- Options: (a) only `trade_agreement`/`customs_union` as listed in #8; (b) also the UK's bloc membership flag (`eu_o`, `cptpp_o`) on the member Markets' rows; (c) also `<bloc>_o` on every UK row (all Markets).
- Chose: (b). EU re-join = TA + Customs Union + `eu_o` on EU Markets' rows; CPTPP = TA + `cptpp_o` on CPTPP Markets' rows; Leave = switch `<bloc>_o`, TA, Customs Union off on the members' rows for each bloc the UK is in.
- Why: (b) reproduces the policy pattern of intra-bloc pairs the model trained on, and keeps a bloc Scenario to the bloc's ~140k rows as the spec expects. (c) would attribute effects to third Markets. Easy to change in `scenarios._changes`.

## #18: removing a Trade Agreement in a custom Scenario
- Options: switch off only `trade_agreement`; also switch off `customs_union`.
- Chose: both.
- Why: a Customs Union implies a Trade Agreement, so leaving one on would be an impossible combination.

## #18: an invalid Market in a Scenario body: 404 or 422?
- Options: 404 (as for unknown Markets in paths); 422.
- Chose: 422, as for every other invalid Scenario.
- Why: the spec says an invalid Scenario returns 422; the Market is part of the body, not the resource path.

## #17: which USD→GBP rate for £ values
- Options: Bank of England annual average (XUAAUSS) for the Baseline year; HMRC yearly average rates; year-end spot; a live rate.
- Chose: Bank of England XUAAUSS, Baseline-year annual average (2024: $1.2783 per £), copied into `exportspredictor/currency.py` and recorded in artefact metadata `gbp_rate` {usd_per_gbp, year, source}.
- Why: an official UK source, an annual average matches BACI's annual Flows, and a fixed table keeps builds offline and reproducible.

## #17: how the minimum-Gap filter takes £, and what region/Sector mean on each view
- Options: `min_gap` in USD only; a currency switch param; separate `min_gap` (USD) and `min_gap_gbp`.
- Chose: both `min_gap` (USD) and `min_gap_gbp`; the app sends £. Both endpoints accept sector, region, min_gap(_gbp) and limit as row filters. In the app the filter that is fixed by the page's subject narrows its picker instead (Sector narrows the Product search; region narrows the Market list).
- Why: the app needs £ before it knows the rate; one shared filter set keeps #18/#19 changes in one place.

## #17: seams for TDD (unattended)
- Options: per spec's Testing Decisions.
- Chose: API contract (TestClient), package pipeline metadata/refresh, Streamlit AppTest against a live API. CSV checked by recording the `st.download_button` data and comparing it with the on-screen table (Why? link column excluded: navigation, not data).
- Why: those are the spec's agreed seams.

## #20: which seams to test (tdd skill asks to confirm)
- Options: the spec's seams; add more.
- Chose: the package (`evaluation` metrics, naive baseline, headline, stability; the pipeline's evaluation on the fixture, incl. a test that multiplies every 2023 Flow by 1,000 and checks the search results don't move), the artefact round trip, the `/evaluation` API contract, the Model Evaluation page under AppTest, and `evaluation.add_to_latest` on the tiny real-layout data root.
- Why: matches the spec's seams; the real-data numbers are evidenced by a run.

## #20: headline metric
- Options: Tweedie deviance; RMSE of log(1+v); Spearman rank correlation within each Product, UK out-of-fold pairs.
- Chose: Spearman within Product on the UK's full 2023 grid, out-of-fold. The page shows all three metrics for both scopes.
- Why: Advisors read the model as a ranking of Markets for one Product; deviance has no plain-English scale.

## #20: how non-UK rows are scored for the "all Exporters" metrics
- Options: one fold model; the mean of the five fold models; hash non-UK rows into folds too.
- Chose: UK rows out-of-fold, non-UK rows by the mean of the five test-year fold models (every one of them trained on all non-UK training rows).
- Why: no fold model has seen a 2023 non-UK row, so all are valid; the mean uses all five instead of an arbitrary one.

## #20: naive baseline details
- Options: plain share × imports; leave-one-out version.
- Chose: (Exporter's exports of the Product to others) × (Importer's imports of it from others) / (world exports of it minus the Flow), i.e. the Flow itself left out, as the model's features do. Predictions are floored at $1 for the Tweedie deviance.
- Why: otherwise the naive baseline would see the target. The $1 floor keeps a zero naive prediction of a real Flow finite.

## #20: stability check of the 2024 Baseline
- Options: compare with the test-year models' 2023 scores; score the 2023 UK grid with the Baseline fold models.
- Chose: the Baseline fold models on the 2023 grid (out-of-fold), comparing within-Product Spearman of Export Potential and Export Gap and the overlap of the top 1,000 Opportunities.
- Why: same models, so only the data year changes; #16's note suggested it.

## #20: where limitations live and how the evaluation is stored
- Options: limitations in the artefact; in the package, served by the API.
- Chose: `evaluation.json` in the artefact folder (optional field `Artefacts.evaluation`; #21 adds `lift` / `event_backtest` keys; the `Evaluation` schema allows extra fields). Limitations are `evaluation.LIMITATIONS`, added by `/evaluation` at serve time.
- Why: the limitations are text, not results; editing them shouldn't need a retrain.

## #20: wording of the sanctions limitation
- Options: "non-tariff Barriers"; "non-tariff restrictions".
- Chose: "Sanctions and other non-tariff restrictions are invisible".
- Why: GLOSSARY defines Barrier as a tariff or the absence of a Trade Agreement only; widening that term is a glossary decision for the user.

## #20: how the headline words a narrow win
- Options: "better than the naive baseline" for any win; flag small margins.
- Chose: a win smaller than 0.05 in rank correlation reads "only slightly better than the naive baseline" (`evaluation.NARROW_MARGIN`). On real data, UK pairs score 0.54 vs 0.52.
- Why: the ticket asks the page to be plain about whether the model beats the baseline; a bare "better" would oversell a 0.02 gap.

## #21: what "grow materially" means for Lift
- Options: any rise; a share threshold only; a share and a dollar threshold.
- Chose: the Flow rises by at least 50% and at least $100,000 from 2017 to 2023 (constants in `backtests`), with a size-matched Lift reported next to the plain one.
- Why: a share alone counts $10 -> $20 rises; a dollar floor alone favours big pairs. The size-matched ratio shows whether the gap says more than "big pairs grow".

## #21: Event Backtest control group and treated set
- Options: hand-picked Markets; a data rule.
- Chose: the 10 largest high-income Markets (by pre-period imports from the same Exporters) with no Trade Agreement with any of them in any window year; EU = the EU27 (UK excluded); pre-period 2016–2018; Japan->EU checked the same way.
- Why: a stated rule is reproducible and auditable; the UK left during the post-period.

## #21: bootstrap units for the intervals
- Options: resample pairs; resample clusters.
- Chose: Lift resamples Exporters (1,000 times); the difference-in-differences resamples Products.
- Why: one Exporter's pairs move together, so pair-level resampling would overstate precision.

## #21: UK–Australia/NZ secondary check model
- Options: train another model on 2012–2022; reuse #20's cached test-year fold models (2012–2022).
- Chose: reuse #20's test-year fold models, UK pairs out-of-fold; pre-period 2021–2022, post 2024 only.
- Why: same training years, no extra GPU run, and ADR 0001 is kept for UK pairs.

## #22: which seams to test (tdd skill asks to confirm)
- Options: the spec's seams only; add package seams for the new modules.
- Chose: `hmrc` (fixed-width reader on a synthetic archive in HMRC's layout, allocation, crosscheck decomposition, country-code map), `governance` (model card sections/numbers from the artefact, lineage SHA-256, PSI worked example), `explain.global_importance` (against per-pair explain_pair), `evaluation.accuracy` by income, and the notebook (parameter cell tagged, every section introduced and concluded, governance topics, fixture run).
- Why: these hold the logic the acceptance criteria name; real numbers are evidenced by runs.

## #22: HMRC OTS source and concordance
- Options: OData API with server-side groupby; the 2023 annual bulk archive; product concordance from UN Stats HS2022->HS2012 conversion table vs compare only at HS2/Market level.
- Chose: bulk archive (one 36 MB file, fully revised) and the UN conversion table (one HS2012 target per HS2022 code). HMRC country codes map through the World Bank list plus overrides (TW, XS, XK, NA, XC/XL->ESP); unattributed value is kept with null keys, not dropped.
- Why: one reproducible download; the spec asks for HS6 aggregation and an HS2012 concordance; keeping unattributed value makes the totals reconcile.

## #22: what stands in for "reporting quality" in the bias review
- Options: World Bank income level; a BACI-derived completeness score (e.g. 2024/2023 Flow counts per Market); region only.
- Chose: income level, added as `by_income` to the evaluation accuracy breakdowns (re-ran `python -m exportspredictor.evaluation` + backtests on the cached models -> 20261117T015239Z).
- Why: standard, published, already in the BACI lookups; no per-country reliability score ships with BACI.

## #22: where the model card lives
- Options: a new artefact version; a file written into the existing artefact folder; only in the notebook.
- Chose: `model_card.md` written into the artefact folder (notebook: the folder it just saved; CLI: the newest folder), generated from metadata + evaluation; owner printed as UNASSIGNED.
- Why: course: generate cards from code and version them with the model; a document is not a table the API loads, so a new version would only duplicate 154 MB.

## review-fixes: how the API recovers when artefacts are missing at startup (spec review (c)3)
- Options: retry the load at most once per interval when a request arrives; an explicit `POST /reload` endpoint; both.
- Chose: a retry interval (`create_app(retry_seconds=30)`, `api.main.RETRY_SECONDS`). `/health` stays 503 until a load succeeds; each failed attempt is logged once.
- Why: `podman compose up` can start the API before the first artefact exists, and an interval recovers without anyone calling an endpoint or restarting, while a failed load no longer scans the disk on every request. An unauthenticated reload endpoint would be one more thing to secure.

## review-fixes: how the monotone-constraint comparison keeps its real-data cost small (spec review (a)2)
- Options: retrain all five fold models without the constraint (~30 min); train one constrained and one unconstrained model on a few Sectors and one held-out fold; compare on the fixture only.
- Chose: `constraints.REAL_PLAN` = Sectors 22, 30, 87, held-out fold 0, the artefact's tuned params, 1,491 rounds, 30% non-UK sample. It took 70 s on the GPU. The fixture plan uses every Sector and fold. Stored as `evaluation.monotone_constraints`; `python -m exportspredictor.constraints` re-adds it to the newest artefact.
- Why: the question is what the constraint does to Scenario Effects, which two otherwise identical models answer directly. Scoring only the held-out fold keeps ADR 0001. Three large UK Sectors give thousands of pairs on both sides of a Trade Agreement.

## review-fixes: where the app gets ranking orders and default Blocked Opportunity thresholds (standards review)
- Options: keep copies in the app; read them from the API's OpenAPI schema; add two small endpoints.
- Chose: `GET /opportunities/rank-by` (from `rankings.RANK_BY`) and `GET /scenarios/thresholds` (from `scenarios.DEFAULT_MIN_*`), like `/scenarios/presets`.
- Why: the app may not import the package, so one source means the API serves it. Reading OpenAPI would couple the app to schema internals.

## review-fixes: the Scenario drill-down's per-Market query (spec review (c)2)
- Options: a new endpoint that re-runs the Scenario for one Market; an optional `market` field on `POST /scenarios` that limits `top_pairs` to that Market.
- Chose: the optional `market` field (422 if unknown). The page asks for it when a Market is chosen (cached per body).
- Why: one Scenario request type, and per-Market totals stay identical. It costs one more Scenario run (0.5–1.4 s on the real artefact).

## review-fixes: test seams for the review fixes (tdd skill asks to confirm seams)
- Options: test at the API contract, the package functions and the Streamlit pages (the existing seams); add new seams.
- Chose: the existing seams only: `TestClient` contract tests, `exportspredictor.{rankings,sources,constraints,downloads}` functions, AppTest pages.
- Why: these are the seams earlier tickets agreed on; no new ones were needed.
