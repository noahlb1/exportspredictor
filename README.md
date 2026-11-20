# UK Exports Predictor

A model that learns from global HS6 goods trade to estimate the UK's **Export Potential** for each Product and Market. It ranks **Opportunities** and **Blocked Opportunities** and simulates trade-policy **Scenarios**. A FastAPI service serves the model, and Advisors use it through a Streamlit app. Terms follow [GLOSSARY.md](GLOSSARY.md).

Results are associations the model has learned, not causes.

The served model is the **Blend**: Export Potential blends an XGBoost model (weight 0.7) with the Naive Baseline (weight 0.3) on the log scale. It was chosen on one measure, the **Selection Measure** (how well it ranks Markets for each Product on held-out UK pairs in 2022), by an experiment loop in which a change was kept only if it beat the Win Threshold (ADR 0008, [research note](docs/research/opportunity-ranking-experiments.md)). On 2023, read once, it ranks UK Markets at 0.555, above the Naive Baseline (0.523), v2 (0.511) and v1 (0.540). The cost is accuracy on size: its Tweedie deviance is higher than v2's, so dollar values deserve more caution than rankings. Scenario Effects and Blocked Opportunity values are **Illustrative**: upper-end associations shown beside a PPML gravity benchmark and the Event Backtests, never forecasts.

The model underneath is v2's: it trains on the top 100 Exporters into every Market, leaves out Excluded Products, uses the Applied Tariff, Relatedness and the re-export ratio as features, averages Export Gaps over the 2022–2024 Gap Period and gives each pair a Potential Range. The notebook and the model card (`model_card.md` in the artefact folder) give the detail.

## Run the app (one command)

You need [podman](https://podman.io/) with `podman compose`, and a trained artefact folder in `./artefacts` (see [Rebuild the data](#rebuild-the-data-and-artefacts); a fixture run takes seconds).

```powershell
podman compose up --build
```

Then open <http://localhost:8501>.

- The API starts first, on <http://localhost:8000>. It reports healthy only once it has loaded the artefacts: up to a minute for the real artefact, whose five fold models take 1.7 GB. `/health` names the artefact version, its format (3) and the Gap Period.
- The app starts only after the API is healthy.
- `./artefacts` is mounted read-only into the API container, so a new artefact version needs only `podman compose restart api`, not a rebuild.
- Stop with `Ctrl+C`, or run `podman compose down`.

If the app says the API is unavailable, check `podman compose logs api` and <http://localhost:8000/health>. The usual causes are an empty `./artefacts` folder, or a newest folder that is not artefact format 3 (such as a v1 folder, or a format-2 folder from before the served Model Form was recorded): the API serves only the lexically newest folder, and `/health` gives the reason it refused it.

## Development with uv

Python is managed with [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                                # install dependencies, including dev tools
uv run pytest                          # run the tests (includes a fixture-mode notebook run)
uv run uvicorn api.main:app --reload   # API on http://localhost:8000
uv run streamlit run app/Home.py       # app on http://localhost:8501
```

While the API runs, http://localhost:8000/docs lists every endpoint with its parameters and response fields, and lets you call each one from the browser to see the JSON the app reads. FastAPI generates the page itself.

Settings are environment variables:

| Variable | Default | Used by |
| --- | --- | --- |
| `EXPORTSPREDICTOR_DATA_DIR` | `<repo>/data` | data download and build steps, notebook |
| `EXPORTSPREDICTOR_ARTEFACTS_DIR` | `<repo>/artefacts` | pipeline (writes the artefacts and the model card), notebook (reads; writes only its own fixture artefacts), API (reads) |
| `EXPORTSPREDICTOR_API_URL` | `http://localhost:8000` | app |

The layout:
- `notebooks/` is the write-up;
- `src/exportspredictor/` holds all data, feature, Scenario and prediction logic;
- `api/` serves saved artefacts;
- `app/` talks to the API only over HTTP.

See [CLAUDE.md](CLAUDE.md) for the modelling guardrails.

## Rebuild the data and artefacts

```powershell
uv run python scripts/rebuild.py --list   # show the steps
uv run python scripts/rebuild.py          # run them all, in order
uv run python scripts/rebuild.py --only notebook   # refresh the committed notebook's outputs only
```

The script downloads every source and builds the derived Parquet in `EXPORTSPREDICTOR_DATA_DIR`. It then makes every modelling choice on the 2022 validation year (the feature ablation, the learning curve and the hyperparameter search), and `python -m exportspredictor.pipeline` trains the v2 fold models, evaluates them on 2023 and saves a new artefact version (format 3) in `EXPORTSPREDICTOR_ARTEFACTS_DIR`. That version serves the Model Form the experiment loop chose (`python -m exportspredictor.loop`, whose saved results the pipeline reads and records), and the pipeline writes `model_card.md` into its folder. To add the PPML gravity benchmark, which the script does not run, `python -m exportspredictor.ppml` derives a new version from the newest one with the benchmark saved beside it and its model card rewritten (about 25 minutes on the CPU, no training, the 2023 Flows not read). Last, it runs the notebook on the real data and saves the outputs into the committed file: the notebook reads the newest artefact and saved results, and trains, fits and builds nothing. `--only notebook` runs that step alone, and it is the one way to refresh the committed outputs; rerun it after any edit to the notebook. The script stops at the first step that fails. Long steps cache their results under `data/derived/`, so a re-run resumes rather than starting again.

The steps are one ordered list, `STEPS`, in [scripts/rebuild.py](scripts/rebuild.py). The times are from an RTX 3080 (10 GB) with a Ryzen 7 5800X and 32 GB of RAM.

| Step | What it runs and what it costs |
| --- | --- |
| BACI Flows and lookups (download, convert to Parquet) | `python -m exportspredictor.baci`; idempotent, about 1.3 GB download, 30 s convert, 2.5 GB peak memory |
| Gravity and WDI features (download CEPII Gravity, WDI, IMF WEO; build Parquet) | `python -m exportspredictor.gravity`; needs the BACI step first (country codes); about 210 MB download, 40 s build |
| Trade Agreement and Trading Bloc features (WTO RTA-IS download, build) | `python -m exportspredictor.policy`; skips the download if `raw/wto/AllRTAs.xlsx` exists; about 200 KB download, 2 s build |
| Applied Tariffs (TRAINS download via WITS, HS17/HS07 -> HS12, EU schedule, gap fill) | `python -m exportspredictor.tariffs`; downloads UNCTAD TRAINS for 2012–2021 (about 1,300 files, 62 MB, resumable) and the UN HS 2017 and HS 2007 -> HS 2012 tables into `raw/trains/` and `raw/unsd/`, then writes `derived/tariffs/` (ADR 0006) |
| Feature table (training rows with sampled zeros, UK grid; per year) | `python -m exportspredictor.features`; needs the four steps above; writes `derived/features_V202601/` (top-100 Exporter scope, every UK Zero Cell, 5% of other Zero Cells; about 5.7 GB), about 15 minutes, 14 GB peak memory |
| HMRC OTS 2023 exports for the BACI cross-check (download, HS 2022 -> 2012) | `python -m exportspredictor.hmrc`; downloads HMRC's 2023 bulk export archive (36 MB) and the UN HS 2022 -> HS 2012 table into `raw/hmrc/` once, then writes `derived/hmrc_ots_2023.parquet` in a few seconds; used only by the notebook's cross-check, never by the model |
| Experiment rows (Sector subset, 2012–2022) | `python -m exportspredictor.experiments build`; the rows the ablation and the learning curve share |
| Feature ablation on 2022 (GPU) | `python -m exportspredictor.experiments ablation`; 2–4 minutes a model; results in `derived/experiments_V202601/results.json` |
| Learning curve on 2022 (GPU) | `python -m exportspredictor.experiments curve`; nine models, about half an hour |
| Hyperparameter search on 2022 (GPU, resumable, about 3 h) | `python -m exportspredictor.search`; Optuna TPE, the study in `derived/search_V202601/study.sqlite3`, so a killed run resumes |
| Experiment loop on 2022: noise (GPU, resumable) | `python -m exportspredictor.loop noise`; trains v2's configuration with three seeds on fold 0; the Noise Floor and the Win Threshold every later Arm must beat; every Arm is judged on the Selection Measure, results in `derived/loop_V202601/results.json`, models cached |
| Experiment loop on 2022: naive (GPU, resumable) | `python -m exportspredictor.loop naive`; the blend with the Naive Baseline (weight chosen from 0.1 to 0.9) and the starting-point model; scoring the blend trains nothing; every Arm is judged on the Selection Measure, results in `derived/loop_V202601/results.json`, models cached |
| Experiment loop on 2022: select (GPU, resumable) | `python -m exportspredictor.loop select`; five configurations from the saved search and v1, each with its own blend weight; every Arm is judged on the Selection Measure, results in `derived/loop_V202601/results.json`, models cached |
| Experiment loop on 2022: two-part (GPU, resumable) | `python -m exportspredictor.loop two-part`; a binary model of any Flow times a size model on non-zero Flows (two models); every Arm is judged on the Selection Measure, results in `derived/loop_V202601/results.json`, models cached |
| Experiment loop on 2022: scope (GPU, resumable) | `python -m exportspredictor.loop scope`; UK Zero Cells sampled like other Exporters', then a 2% zero rate (two models); every Arm is judged on the Selection Measure, results in `derived/loop_V202601/results.json`, models cached |
| Experiment loop on 2022: product-detail (GPU, resumable) | `python -m exportspredictor.loop product-detail`; three Product-level features, each added on its own (three models); every Arm is judged on the Selection Measure, results in `derived/loop_V202601/results.json`, models cached |
| Experiment loop on 2022: constraints (GPU, resumable) | `python -m exportspredictor.loop constraints`; the best Arm without the monotone constraints (one model); every Arm is judged on the Selection Measure, results in `derived/loop_V202601/results.json`, models cached |
| Five-fold confirmation of the loop's best form on 2022 (GPU, resumable, about 1 h) | `python -m exportspredictor.loop confirm`; the best Arm against v2 on every held-out UK pair of five folds; the pipeline serves the form the loop chose and records these results in the artefact |
| Train the fold and test-year models, evaluate and save the artefacts (GPU, about 2.5 h) | `python -m exportspredictor.pipeline`; trains the five fold models (about 75 minutes), scores the 2022–2024 Gap Period and the Potential Range, trains and evaluates the 2023 test-year models (about 75 minutes), reads the v1 comparison's primary metric from the frozen v1 tables (no training), and saves a new artefact version. Before it evaluates, it records the 2023 read in the Test-year Ledger (`docs/test-year-ledger.csv`) and refuses a configuration the ledger already holds, unless `--read-test-year-again` is passed, which is recorded too (ADR 0009); commit the ledger with the run. The Event Backtests (Lift included), the monotone-constraint comparison and the gain ladder each run only when asked for: `--event-backtests`, `--constraint-comparison`, `--gain-ladder`, or `--every-part` for all three (together about two more hours). The evaluation, the Model Evaluation page and the model card show a skipped part as not run for this version. Fold, test-year, backtest and ladder models are cached under `data/derived/`, so a re-run only scores. `python -m exportspredictor.pipeline compare` adds the v1 comparison with its gain ladder to a copy of the newest version, which must hold the Event Backtests |
| The notebook (reads the saved artefact) | runs `notebooks/exports_predictor.ipynb` with `EXPORTSPREDICTOR_FIXTURE_MODE=0` and saves its outputs in place (`--only notebook` runs this step alone). It reads saved results only (the newest artefact, its model card and PPML benchmark, and the experiment loop's results) and trains nothing; about 2 minutes on the CPU |

## Manual UI checklist

Run this before a release, against `podman compose up` with real artefacts, in a browser window 1280 px wide or wider (the app is designed for 1280 px and wider; phones are unsupported). Each item names its page and what to click. Tick each item.

**UK Exports Predictor (every page)**
- [ ] The menu lists four pages, Leaderboard, Blocked Opportunities, Scenarios and Model Evaluation, with no group headings, and the app opens on the Leaderboard.
- [ ] The logo tile (a white container ship on a blue square) sits alone at the top of the sidebar, shows in a collapsed sidebar, and is the browser-tab icon.
- [ ] The app follows the computer's light or dark mode: in light mode, near-black text on a light page with a sidebar that stands apart; in dark mode, light text on a warm near-black page. Check captions, charts and the info, warning and error boxes in both.
- [ ] Click something slow (e.g. a filter on the real Leaderboard): after a short delay the logo's ship rides the waves in the middle of the page until the app is done.
- [ ] Every page shows, under its heading and introduction, the grey line "Figures are associations the model has learned, not causes or forecasts.", and, after a divider at the foot of the page, a grey footer stating the Baseline year (provisional data), that Export Gaps are 2022–2024 averages, "Values are nominal US dollars, as reported by BACI", and the model version. There is no blue banner, and no page shows a pound sign or a currency switch.
- [ ] Stop the API (`podman compose stop api`) and reload: within a few seconds, every page shows a clear "API is unavailable" message instead of an error or a spinner. Start it again with `podman compose start api` and wait for <http://localhost:8000/health> to answer before reloading.
- [ ] Narrow the browser window to about 700 px (half a laptop screen) and open each page: no page shows an error, though columns may stack or cut long names.

**Leaderboard**
- [ ] The Leaderboard opens without searching, listing UK Product–Market pairs by Export Gap, largest first, with no Excluded Product (e.g. no crude oil or gold) among them.
- [ ] On the real artefact (about 1.16 M pairs), the Leaderboard loads, and responds to each filter change, within about a second.
- [ ] The filters sit in the sidebar: Product search, Market, Sector (by name, e.g. "Beverages, spirits and vinegar"), Market region, Opportunity Type, UK Trade Agreement in force, minimum Export Gap, Hide Flagged Products, Hide Re-export Hubs, Hide Sanctioned Markets, Hide Restricted Pairs and Rank by, then **Reset filters**.
- [ ] One line above the table names the filters in force (or says there are none), "N pairs shown of M" and the total positive Export Gap ("2022–2024 average"); it changes with the filters. There is no row of summary metrics. A caption under it says Export Potential is a median-type estimate that sums to about N% of Actual Exports (N from the served artefact: about 80% before spec #113's rebuild), so totals are relative sizes, not dollars the UK could add.
- [ ] Each row shows its Opportunity Type (Expand, New Market or New Product), its Relative Export Gap, whether a UK Trade Agreement is in force (e.g. whisky: Japan Yes, India No), and any Flagged Product, Re-export Hub, Sanctioned Market or Restricted Pair badge.
- [ ] Export Potential, Actual Exports and Export Gap appear side by side, each headed "2022–2024 average"; values are in USD: column headers say "(USD, nominal)".
- [ ] Switch "Rank by" to Relative Export Gap: a minimum Export Potential selector appears ($0, $100.0k, $1.0m, $10.0m, $100.0m; default $1.0m), pairs below it disappear, and pairs tied at 100% are ordered by Export Gap.
- [ ] Search for a known HS6 code (e.g. `220830`, whisky) and for words (e.g. `smartphone` finds `851712`): the Product is chosen, and the table ranks its Markets by Export Gap, largest first. A prefix matching several Products (e.g. `8703`) offers each to choose from.
- [ ] Search text that matches nothing (e.g. `zzz`) shows a clear "No Product matches" warning.
- [ ] Search an Excluded Product (e.g. `gold`): it is marked "(Excluded Product)", and choosing it shows why it is not ranked and points to the Excluded Products section of Model Evaluation, instead of a ranking.
- [ ] Choose a Market (e.g. Japan): the table ranks its Products by Export Gap. The region filter narrows the Market list.
- [ ] The Sector, region, Market, Opportunity Type, Trade Agreement, minimum-Gap and hide filters each narrow the rows; a filter matching nothing says so.
- [ ] Search an aircraft Product (e.g. `880230`): every row carries the "Flagged Product" badge, and its detail panel's warning says "volatile: UK exports swing widely from year to year"; banknotes and stamps (`490700`) say "one-off: few, large contracts".
- [ ] Search a jewellery Product (e.g. `711319`): every row carries the "Flagged Product" caveat badge, and the Netherlands, Hong Kong and other Re-export Hubs carry "Re-export Hub". "Hide Flagged Products" and "Hide Re-export Hubs" are off by default; ticking one removes exactly those rows.
- [ ] Search whisky (`220830`): Saudi Arabia, Kuwait, Libya and Iran carry the "Restricted Pair" badge, and Russia, Belarus, North Korea and Syria carry "Sanctioned Market". "Hide Sanctioned Markets" and "Hide Restricted Pairs" are off by default; ticking one removes exactly those rows. The Blocked Opportunities page has the same two hide filters.
- [ ] Each filter you set appears in the page's URL (Product as its HS6 code, Market as its ISO3 code; defaults left out). Copy the URL into a fresh browser tab: the same filters and rows appear. A link naming one Product and one Market (e.g. `?product=220830&market=JPN`) leaves that one pair.
- [ ] Set some filters, open another page from the menu and come back: the filters are still set.
- [ ] **Reset filters** restores the full ranking and clears the filters from the URL.
- [ ] Each Leaderboard row shows its Potential Range as "low–high". No ranking page (Leaderboard, Blocked Opportunities) shows an "Uncertain Opportunity" badge or a filter for it; the Flagged Product and Re-export Hub badges and their hide filters are unchanged.
- [ ] Export the Leaderboard to CSV and open it, unfiltered and with one Product or one Market chosen: the columns and rows match the table on screen.
- [ ] The page opens on **Table and detail**: a three-column table (Product, Market, Export Gap, all three in view at 1280 px) beside a detail panel already explaining the top row. Click any value of another row: that whole row is tinted, and the panel explains its pair and starts from its top.
- [ ] Switch to **Wide table**: the table shows the wide columns (Applied Tariff % among them), the same pair stays selected and tinted, and its detail sits under the table. Switch back: still the same pair.
- [ ] Open `?product=902214&market=JPN` and switch to **Wide table**: Actual Exports of $18,882 show as `0.0` (a value under $50,000 rounds to 0.0 in $ millions), and the column still sorts as numbers.
- [ ] Choose the Market USA (5,138 pairs) and turn on **Show all**: a row click answers in about a second.
- [ ] The first visit shows three steps (Read the ranking, Open a row, Narrow it down) above the table; **Got it** closes them, and **What is this?** beside the heading brings them back.

**Blocked Opportunities**
- [ ] Open Blocked Opportunities from the sidebar: it lists only Blocked Opportunities (about 3,800, not every pair), with no "Rank by" toggle; the Leaderboard does not offer that order.
- [ ] The line above the tables gives the filters in force, the number of Blocked Opportunities and their total value, labelled illustrative; one caption gives the unit ($ millions), the Baseline year, the thresholds (20% and $1M) and that other thresholds are on the Scenarios page, then the same median-type note as the Leaderboard's, with its N%.
- [ ] Left, the Markets table (Market, Blocked Opportunities, Total value) is largest total first, with the top Market chosen. Click any value of another Market: its row is tinted, and its line and Products table replace the first.
- [ ] The Products table (Product, Blocked Opportunity value, Uplift %, Applied Tariff %, Potential with agreement, Actual Exports) lists every Blocked Opportunity of the Market, largest first, with the top one chosen. Where 2024 Actual Exports exceed Export Potential (e.g. passenger cars `870324` to the USA), Uplift % holds the value as a percentage of Export Potential and Potential with agreement is blank; elsewhere it is the other way round. The Potential with agreement tooltip names Export Potential with a Trade Agreement and says it is illustrative; the Applied Tariff tooltip says a 0 means the value rests on the Trade Agreement alone, and a blank that it is not known.
- [ ] The panel under it opens with the Barrier, in a box headed by the pair, then the pair's HS6 code and caveats, then Baseline Export Potential, Export Potential with a Trade Agreement and Actual Exports, then the facts (the Potential Range among them, no Relative Export Gap), factors and years. Where Actual Exports exceed Export Potential (e.g. cars to the USA), the middle figure is "Uplift with a Trade Agreement" (e.g. +51%), the Barrier and the facts give the value as that uplift, and the Barrier box adds one line on why; no dollar figure with a Trade Agreement appears. Elsewhere none of this changes.
- [ ] The Export Gap appears nowhere on the page, apart from the footer every page shares.
- [ ] The Sector, region, minimum value and hide filters, and a Product search, each narrow the Markets, the line and the CSV; Reset filters restores them, and they are kept apart from the Leaderboard's filters. There is no Market, Opportunity Type or Trade Agreement filter.
- [ ] The Illustrative note sits in an expander under the tables, with the PPML gravity benchmark, the model's association and the Event Backtest, or says one was not run for this version.
- [ ] Every caption describes the value as what the model associates with adding a Trade Agreement, not as a cause or a forecast.
- [ ] Export the CSV and open it: every filtered Blocked Opportunity, across all Markets, in full dollars.

**Scenarios**
- [ ] Run a Trading Bloc preset Scenario (e.g. EU re-join). It returns within a few seconds, and the summary shows the total change, the change per Market and the top Products gaining, phrased as associations.
- [ ] Run the CPTPP preset: it shows a non-zero Scenario Effect (the 2024 Baseline has no UK CPTPP membership).
- [ ] Blocked Opportunities unlocked by the Scenario are listed, and changing the thresholds changes the list.
- [ ] The builder sits in a box at the top and the page shows no result until **Run Scenario** is pressed. Change a control after a run: a line says the controls differ from the result shown.
- [ ] Click any value of a Market row in Change per Market: that row is tinted, and the pairs most affected in that Market are listed under it, largest effect first; the top Market is selected on open.
- [ ] Click any value of a pair in the drill-down or in Blocked Opportunities (no pair is tinted before): that row is tinted, and a box under that table shows its caveats, its Baseline and Scenario Export Potential, the Scenario Effect and both Applied Tariffs. Its link opens the Leaderboard on that Product and Market.
- [ ] The tables are in $ millions with the unit stated once above them, and the Blocked Opportunities table has a Baseline-year Actual Exports column. The caption above the tables ends with the median-type note on summed Export Potential, with its N%.
- [ ] The drill-down and Blocked Opportunities tables have a Caveats column: add a Trade Agreement with Saudi Arabia, and its alcoholic beverages read "Restricted Pair".
- [ ] A Scenario that changes nothing (e.g. adding an agreement with Japan, which already exists, or the Leave preset in 2024) says so.
- [ ] In tariff mode, set a tariff above the highest one in training for one Market (e.g. 5000% for the USA; on the real artefact the training range ends at 3,000%): the Scenario Effect is never a rise, and the page says the effect flattens beyond the training range. Removing a Trade Agreement (e.g. with Japan) shows the restored 2021 MFN tariff in the pair tables.
- [ ] After a run, the page states the EU–Japan Event Backtest result (predicted against observed) beside the Scenario Effect.

**Leaderboard: the detail panel**
- [ ] The panel opens with the pair, its HS6, Sector and region, and its caveats: for a Flagged Product (naming its reason: Sector 71, volatile or one-off) or a Re-export Hub (e.g. `711319` to Hong Kong) a warning each, for a Sanctioned Market (e.g. any Product to Russia) a warning naming the regime and the year it became broad, for a Restricted Pair (e.g. whisky to Saudi Arabia) a warning naming what the Market prohibits, and for a pair whose Applied Tariff is carried forward (e.g. `300490` to India) that warning.
- [ ] Export Potential, Actual Exports and Export Gap show as three figures ("2022–2024 average"), each with a tooltip in the API's wording.
- [ ] The facts start with the Applied Tariff, then UK Trade Agreement in force and the Blocked Opportunity value (illustrative), then the Opportunity Type with its meaning, the Relative Export Gap and the Potential Range; the Potential Range's tooltip describes it in plain words (comparable pairs, never a confidence interval). There is no Potential Range bar and no Uncertain Opportunity warning.
- [ ] A pair with no UK Trade Agreement (e.g. whisky to India) shows **The Barrier**: its Applied Tariff and the amount the model associates with adding an agreement, marked Illustrative, as a warning when the pair is a Blocked Opportunity. A pair with an agreement (e.g. whisky to Japan) shows none.
- [ ] The factors are described in plain words (no raw feature names), as a bar chart (raised in blue, lowered in red, values readable in light and dark mode) and a table, each marked as raising or lowering Export Potential, phrased as associations; **All factors** lists every one.
- [ ] The year-by-year chart shows Actual Exports against Export Potential; **Figures by year** shows the last six years in $ millions with each year's Applied Tariff mark, and the three 2022–2024 yearly gaps average to the Export Gap.
- [ ] Open a link naming one Product and one Market (e.g. `?product=220830&market=JPN`) in a fresh browser tab: the panel explains that pair.

**Model Evaluation**
- [ ] Four figures open the page: the rank correlation for UK pairs with the Naive Baseline's beside it, the share of Actual Exports inside the Potential Range against its 80% target, size-matched Lift, and ranking stability. Every section below is folded except "How accurate is the model?" and "Known limitations".
- [ ] The "Excluded Products" section, just above "Known limitations", says how many Products are left out and why, lists each reason with its number of Products, and holds the full list (HS6, Product, Sector by name, reason) in an expander. No ranking page (Leaderboard, Blocked Opportunities) shows an Excluded Products list.
- [ ] The plain-English accuracy headline renders, green if the model beats the naive baseline and amber with a plain warning if it does not, with the test year and the non-UK training sample stated below it.
- [ ] The metric tables (model against naive baseline) render for UK pairs and for all Exporters.
- [ ] "Where the model is weakest" lists the weakest Sectors and regions; switching "Rows evaluated" changes the tables.
- [ ] "Observed against carried-forward tariffs" splits the test rows by Applied Tariff (observed, carried forward from 2021) for both scopes, with a sentence saying how they split (on 2023 every row is carried forward).
- [ ] The stability sentence renders, followed by the Potential Range's held-out coverage against its 80% target, the share of Opportunities whose Actual Exports lie inside their Potential Range (87% on the served version), and a coverage table per Exporter.
- [ ] The known-limitations panel (the result limitations such as "Does not beat a naive ranking for UK pairs" when it does not, goods only, tariffs carried forward from 2021, Excluded Products, the top-100 Exporter scope, nominal US dollars, what the Potential Range assumes, associations not causes, provisional 2024, the 2012 edition, reporting quality, sanctions) renders, with no "no tariffs" entry.
- [ ] The Lift section states how many times more often the largest Export Gaps grew, with a 95% interval, worded as a tendency, shows size-matched Lift next to raw Lift with the explanation that raw Lift mostly reflects pair size, and shows the growth share for each tenth of Export Gaps.
- [ ] The Event Backtest shows, for EU exports to Japan and Japan's exports to the EU, a table of predicted against observed change per year, names the control Markets, says the Applied Tariff is 0 with the agreement on and the MFN rate with it off, and states the COVID caveat; the UK–Australia/New Zealand secondary check follows.
