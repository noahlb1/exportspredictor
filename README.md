# UK Exports Predictor

A model that learns from global HS6 goods trade to estimate the UK's **Export Potential** for each Product and Market. It ranks **Opportunities** and **Blocked Opportunities** and simulates trade-policy **Scenarios**. A FastAPI service serves the model, and Advisors use it through a Streamlit app. Terms follow [GLOSSARY.md](GLOSSARY.md).

Results are associations the model has learned, not causes.

This is **v2**. It trains on the top 100 Exporters into every Market, leaves out Excluded Products, adds the Applied Tariff, Relatedness and the re-export ratio as features, averages Export Gaps over the 2022–2024 Gap Period and gives each pair a Potential Range. Under a comparison rule fixed before the 2023 test year was read, v2 does **not** beat v1 on the headline UK ranking measure (0.511 against 0.540, and below the naive baseline's 0.523), though it estimates the size of Flows better; the notebook and the model card give the detail. Spec #65 then chose the model on that ranking alone (ADR 0008): the served Export Potential blends the v2 model with the naive baseline, and on 2023, read once, it ranks UK Markets at 0.555, above the naive baseline (0.523), v2 and v1, at the cost of a higher Tweedie deviance ([research note](docs/research/opportunity-ranking-experiments.md)).

## Run the app (one command)

You need [podman](https://podman.io/) with `podman compose`, and a trained artefact folder in `./artefacts` (see [Rebuild the data](#rebuild-the-data-and-artefacts); a fixture run takes seconds).

```powershell
podman compose up --build
```

Then open <http://localhost:8501>.

- The API starts first, on <http://localhost:8000>. It reports healthy only once it has loaded the artefacts: up to a minute for the real v2 artefact, whose five fold models take 1.7 GB. `/health` names the artefact version, its format (2) and the Gap Period.
- The app starts only after the API is healthy.
- `./artefacts` is mounted read-only into the API container, so a new artefact version needs only `podman compose restart api`, not a rebuild.
- Stop with `Ctrl+C`, or run `podman compose down`.

If the app says the API is unavailable, check `podman compose logs api` and <http://localhost:8000/health>. The usual causes are an empty `./artefacts` folder, or a newest folder that is not artefact format 3 (such as a v1 folder, or a format-2 folder from before the served model form was recorded): the API serves only the lexically newest folder, and `/health` gives the reason it refused it.

## Development with uv

Python is managed with [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                                # install dependencies, including dev tools
uv run pytest                          # run the tests (includes a fixture-mode notebook run)
uv run uvicorn api.main:app --reload   # API on http://localhost:8000
uv run streamlit run app/Home.py       # app on http://localhost:8501
```

Settings are environment variables:

| Variable | Default | Used by |
| --- | --- | --- |
| `EXPORTSPREDICTOR_DATA_DIR` | `<repo>/data` | data download and build steps, notebook |
| `EXPORTSPREDICTOR_ARTEFACTS_DIR` | `<repo>/artefacts` | pipeline (writes), notebook (fixture artefacts and the model card), API (reads) |
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
```

The script downloads every source and builds the derived Parquet in `EXPORTSPREDICTOR_DATA_DIR`. It then makes every modelling choice on the 2022 validation year (the feature ablation, the learning curve and the hyperparameter search), and `python -m exportspredictor.pipeline` trains the v2 fold models, evaluates them on 2023 and saves a new artefact version (format 3), serving the model form the experiment loop chose (`python -m exportspredictor.loop`, whose saved results it reads and records) in `EXPORTSPREDICTOR_ARTEFACTS_DIR`. Last, it runs the notebook on the real data: the notebook reads the newest artefact and trains nothing, and it writes `model_card.md` into that artefact's folder. The script stops at the first step that fails. Long steps cache their results under `data/derived/`, so a re-run resumes rather than starting again.

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
| Train the fold and test-year models, evaluate and save the artefacts (GPU, about 2.5 h) | `python -m exportspredictor.pipeline`; trains the five fold models (about 75 minutes), scores the 2022–2024 Gap Period and the Potential Range, trains and evaluates the 2023 test-year models (about 75 minutes), reads the v1 comparison's primary metric from the frozen v1 tables (no training), and saves a new artefact version. The Event Backtests (Lift included), the monotone-constraint comparison and the gain ladder each run only when asked for: `--event-backtests`, `--constraint-comparison`, `--gain-ladder`, or `--every-part` for all three (together about two more hours). The evaluation, the Model Evaluation page and the model card show a skipped part as not run for this version. Fold, test-year, backtest and ladder models are cached under `data/derived/`, so a re-run only scores. `python -m exportspredictor.pipeline compare` adds the v1 comparison with its gain ladder to a copy of the newest version, which must hold the Event Backtests |
| The notebook (reads the saved artefact) | runs `notebooks/exports_predictor.ipynb` with `EXPORTSPREDICTOR_FIXTURE_MODE=0`. It reads the newest artefact and trains nothing; it rebuilds the gravity, policy and feature tables (about 15 minutes, 14 GB), fits the PPML gravity benchmark (about 30 minutes on the CPU) and writes `model_card.md` into the newest artefact folder |

## Manual UI checklist

Run this before a release, against `podman compose up` with real artefacts. Tick each item.

**UK Exports Predictor (home page)**
- [ ] The page loads, and the banner states the Baseline year, that it is provisional, that Export Gaps are 2022–2024 averages, and the model version.
- [ ] Every page has one caption, "Values are nominal US dollars, as reported by BACI", and no page shows a pound sign or a currency switch.
- [ ] Stop the API (`podman compose stop api`) and reload: within a few seconds, every page shows a clear "API is unavailable" message instead of an error or a spinner. Start it again with `podman compose start api` and wait for <http://localhost:8000/health> to answer before reloading.
- [ ] The Leaderboard opens on Home without searching, listing UK Product–Market pairs by Export Gap, largest first, with no Excluded Product (e.g. no crude oil or gold) among them.
- [ ] On the real artefact (about 1.16 M pairs), the Leaderboard loads, and responds to each filter change, within about a second.
- [ ] The summary row shows the number of pairs and the total positive Export Gap ("2022–2024 average"); both change with the filters.
- [ ] Each row shows its Opportunity Type (Expand, New Market or New Product), its Relative Export Gap, whether a UK Trade Agreement is in force, and any Flagged Product or Re-export Hub badge.
- [ ] Switch "Rank by" to Relative Export Gap: a minimum Export Potential selector appears ($0, $100.0k, $1.0m, $10.0m, $100.0m; default $1.0m), pairs below it disappear, and pairs tied at 100% are ordered by Export Gap.
- [ ] The Sector, region, Market, Opportunity Type, Trade Agreement, minimum-Gap and hide filters each narrow the rows; a filter matching nothing says so.
- [ ] "Excluded Products" under the filters lists each Excluded Product with its reason.
- [ ] Each Leaderboard row shows its Potential Range as "low–high"; Uncertain Opportunities carry the "Uncertain Opportunity" badge. "Hide Uncertain Opportunities" is off by default; ticking it removes exactly those rows and leaves the order of the rest unchanged.
- [ ] Export the Leaderboard to CSV and open it: the columns and rows match the table on screen.
- [ ] Click **Why?** on a Leaderboard row: the Why? page opens for that pair and shows its Opportunity Type.

**Product → Markets**
- [ ] Search for a known HS6 code (e.g. `220830`, whisky) and for words (e.g. `smartphone` finds `851712`): the Product's code and description appear, with Markets ranked by Export Gap, largest first.
- [ ] Export Potential, Actual Exports and Export Gap appear side by side, each headed "2022–2024 average".
- [ ] Search text that matches nothing (e.g. `zzz`) shows a clear "No Product matches" warning.
- [ ] Open "Excluded Products" under the filters: each Excluded Product (e.g. `270900` crude oil, `710812` gold) is listed with its reason. Searching one (e.g. `gold`) marks it "(Excluded Product)", and choosing it shows why it is not ranked instead of a ranking.
- [ ] Values are in USD: column headers say "(USD, nominal)".
- [ ] Each row says whether a UK Trade Agreement is in force with the Market (e.g. Japan: Yes, India: No).
- [ ] The Sector filter narrows the Product search; the region and minimum-Gap filters narrow the ranking.
- [ ] Search a jewellery Product (e.g. `711319`): every row carries the "Flagged Product" caveat badge, and the Netherlands, Hong Kong and other Re-export Hubs carry "Re-export Hub". "Hide Flagged Products" and "Hide Re-export Hubs" are off by default; ticking one removes exactly those rows.
- [ ] Export the ranking to CSV and open it: the columns and row count match the table on screen.

**Market → Products**
- [ ] Pick a Market (e.g. Japan): Products are ranked by Export Gap, and the page says whether a UK Trade Agreement is in force with it.
- [ ] The region filter narrows the Market list; the Sector and minimum-Gap (USD) filters narrow the ranking, and a filter matching nothing says so.
- [ ] Values are in USD: column headers say "(USD, nominal)".
- [ ] Export the ranking to CSV and open it: the columns and row count match the table on screen.

**Blocked Opportunities**
- [ ] Open Blocked Opportunities from the sidebar: every UK pair is ranked by Blocked Opportunity value, largest first, with no "Rank by" toggle; the Leaderboard and ranking pages no longer offer that order.
- [ ] Pairs with a Market that has a UK Trade Agreement in force (e.g. Japan) have a value of 0; rows clearing 20% and $1M are marked as Blocked Opportunities.
- [ ] The summary shows the number of pairs, the number of Blocked Opportunities and the total value of those Blocked Opportunities (not of every pair); all three change with the filters. No pair has a negative value: an added agreement never lowers Export Potential.
- [ ] The Sector, region, Market, Opportunity Type, Trade Agreement, minimum-Gap and hide filters each narrow the rows; a filter matching nothing says so.
- [ ] Every caption describes the value as what the model associates with adding a Trade Agreement, not as a cause or a forecast.
- [ ] Export the table to CSV and open it: the columns and rows match the table on screen.

**Scenarios**
- [ ] Run a Trading Bloc preset Scenario (e.g. EU re-join). It returns within a few seconds, and the summary shows the total change, the change per Market and the top Products gaining, phrased as associations.
- [ ] Run the CPTPP preset: it shows a non-zero Scenario Effect (the 2024 Baseline has no UK CPTPP membership).
- [ ] Blocked Opportunities unlocked by the Scenario are listed, and changing the thresholds changes the list.
- [ ] Pick a Market in the drill-down: its most affected Product–Market pairs are listed, largest effect first.
- [ ] A Scenario that changes nothing (e.g. adding an agreement with Japan, which already exists, or the Leave preset in 2024) says so.
- [ ] In tariff mode, set a tariff above the highest one in training for one Market (e.g. 5000% for the USA; on the real artefact the training range ends at 3,000%): the Scenario Effect is never a rise, and the page says the effect flattens beyond the training range. Removing a Trade Agreement (e.g. with Japan) shows the restored 2021 MFN tariff in the pair tables.
- [ ] The page states the EU–Japan Event Backtest result (predicted against observed) above the Scenario choice.

**Why?**
- [ ] On **Product → Markets** and **Market → Products**, click **Why?** on a ranking row: the Why? page opens for that Product and Market.
- [ ] The factors are described in plain words (no raw feature names), each marked as raising or lowering Export Potential, phrased as associations.
- [ ] A year-by-year table shows the last six years' UK Actual Exports (0 for years with none) and, for 2022–2024, each year's Export Potential and Export Gap; the three yearly gaps average to the Export Gap shown above it.
- [ ] Open the Why? page from the sidebar, pick a Product and a Market: the same explanation appears.
- [ ] Open Why? for a Flagged Product or a Re-export Hub (e.g. `711319` to Hong Kong): its caveats are shown under the heading.
- [ ] The Why? page shows a range bar with Export Potential and Actual Exports marked on the Potential Range, described in plain words (comparable pairs, never a confidence interval); an Uncertain Opportunity carries its caveat.

**Model Evaluation**
- [ ] The plain-English accuracy headline renders, green if the model beats the naive baseline and amber with a plain warning if it does not, with the test year and the non-UK training sample stated below it.
- [ ] The metric tables (model against naive baseline) render for UK pairs and for all Exporters.
- [ ] "Where the model is weakest" lists the weakest Sectors and regions; switching "Rows evaluated" changes the tables.
- [ ] "Observed against carried-forward tariffs" splits the test rows by Applied Tariff (observed, carried forward from 2021) for both scopes, with a sentence saying how they split (on 2023 every row is carried forward).
- [ ] The stability sentence renders, followed by the Potential Range's held-out coverage against its 80% target and a coverage table per Exporter.
- [ ] The known-limitations panel (the result limitations such as "Does not beat a naive ranking for UK pairs" when it does not, the Potential Range's fixed width in dollars, goods only, tariffs carried forward from 2021, Excluded Products, the top-100 Exporter scope, nominal US dollars, what the Potential Range assumes, associations not causes, provisional 2024, the 2012 edition, reporting quality, sanctions) renders, with no "no tariffs" entry.
- [ ] The Lift section states how many times more often the largest Export Gaps grew, with a 95% interval, worded as a tendency, shows size-matched Lift next to raw Lift with the explanation that raw Lift mostly reflects pair size, and shows the growth share for each tenth of Export Gaps.
- [ ] The Event Backtest shows, for EU exports to Japan and Japan's exports to the EU, a table of predicted against observed change per year, names the control Markets, says the Applied Tariff is 0 with the agreement on and the MFN rate with it off, and states the COVID caveat; the UK–Australia/New Zealand secondary check follows.
