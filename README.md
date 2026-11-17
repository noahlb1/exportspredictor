# UK Exports Predictor

A model that learns from global HS6 goods trade to estimate the UK's **Export Potential** for each Product and Market. It ranks **Opportunities** and **Blocked Opportunities** and simulates trade-policy **Scenarios**. A FastAPI service serves the model, and Advisors use it through a Streamlit app. Terms follow [GLOSSARY.md](GLOSSARY.md).

Results are associations the model has learned, not causes.

## Run the app (one command)

You need [podman](https://podman.io/) with `podman compose`, and a trained artefact folder in `./artefacts` (see [Rebuild the data](#rebuild-the-data-and-artefacts); a fixture run takes seconds).

```powershell
podman compose up --build
```

Then open <http://localhost:8501>.

- The API starts first, on <http://localhost:8000>. It reports healthy only once it has loaded the artefacts.
- The app starts only after the API is healthy.
- `./artefacts` is mounted read-only into the API container, so a new artefact version needs only `podman compose restart api`, not a rebuild.
- Stop with `Ctrl+C`, or run `podman compose down`.

If the app says the API is unavailable, check `podman compose logs api`. The usual cause is an empty `./artefacts` folder.

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
| `EXPORTSPREDICTOR_ARTEFACTS_DIR` | `<repo>/artefacts` | notebook (writes), API (reads) |
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

The script downloads every source, builds the derived Parquet in `EXPORTSPREDICTOR_DATA_DIR`, then runs the notebook, which trains, evaluates and saves a new artefact version in `EXPORTSPREDICTOR_ARTEFACTS_DIR`. It stops at the first step that fails.

The steps are one ordered list, `STEPS`, in [scripts/rebuild.py](scripts/rebuild.py). Each data ticket adds its own lines there:

| Step | Status |
| --- | --- |
| BACI Flows and lookups (download, convert to Parquet) | in place: `python -m exportspredictor.baci`; idempotent, about 1.3 GB download, 30 s convert, 2.5 GB peak memory |
| Gravity and WDI features (download CEPII Gravity, WDI, IMF WEO; build Parquet) | in place: `python -m exportspredictor.gravity`; needs the BACI step first (country codes); about 210 MB download, 40 s build |
| Trade Agreement and Trading Bloc features (WTO RTA-IS download, build) | in place: `python -m exportspredictor.policy`; idempotent (skips the download if `raw/wto/AllRTAs.xlsx` exists), about 200 KB download, 2 s build |
| Feature table (training rows with sampled zeros, UK grid; per year) | in place: `python -m exportspredictor.features`; needs the three steps above; writes `derived/features_V202601/`, about 3.4 GB, 90 s build, 8 GB peak memory |
| HMRC OTS 2023 exports for the BACI cross-check (download, HS 2022 -> 2012) | in place: `python -m exportspredictor.hmrc`; downloads HMRC's 2023 bulk export archive (36 MB) and the UN HS 2022 -> HS 2012 conversion table into `raw/hmrc/` once, then writes `derived/hmrc_ots_2023.parquet` in a few seconds; used only by the notebook's cross-check, never by the model |
| Train, evaluate and save artefacts (notebook) | in place: runs the notebook with `EXPORTSPREDICTOR_FIXTURE_MODE=0`; tunes on 2022, trains five fold models on the GPU (about 30 minutes on an RTX 3080) and saves a new artefact version. To train without the notebook: `python -m exportspredictor.pipeline`. The run also precomputes the Blocked Opportunity values; to recompute only those from the newest artefact's saved fold models, run `python -m exportspredictor.scenarios` (seconds, CPU, writes a new artefact version) |

## Manual UI checklist

Run this before a release, against `podman compose up` with real artefacts. Tick each item.

**UK Exports Predictor (home page)**
- [ ] The page loads, and the banner states the Baseline year, that it is provisional, that Export Gaps are 2022–2024 averages, and the model version.
- [ ] Every page has one caption, "Values are nominal US dollars, as reported by BACI", and no page shows a pound sign or a currency switch.
- [ ] Stop the API (`podman compose stop api`) and reload: every page shows a clear "API is unavailable" message instead of an error. Start it again with `podman compose start api`.
- [ ] The Leaderboard opens on Home without searching, listing UK Product–Market pairs by Export Gap, largest first, with no Excluded Product (e.g. no crude oil or gold) among them.
- [ ] On the real artefact (about 1.2 M pairs), the Leaderboard loads, and responds to each filter change, within about a second.
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
- [ ] Search text that matches nothing (e.g. `999999` or `zzz`) shows a clear "No Product matches" warning.
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
- [ ] The summary shows the number of pairs, the number of Blocked Opportunities and their total value; all three change with the filters.
- [ ] The Sector, region, Market, Opportunity Type, Trade Agreement, minimum-Gap and hide filters each narrow the rows; a filter matching nothing says so.
- [ ] Every caption describes the value as what the model associates with adding a Trade Agreement, not as a cause or a forecast.
- [ ] Export the table to CSV and open it: the columns and rows match the table on screen.

**Scenarios**
- [ ] Run a Trading Bloc preset Scenario (e.g. EU re-join). It returns within a few seconds, and the summary shows the total change, the change per Market and the top Products gaining, phrased as associations.
- [ ] Run the CPTPP preset: it shows a non-zero Scenario Effect (the 2024 Baseline has no UK CPTPP membership).
- [ ] Blocked Opportunities unlocked by the Scenario are listed, and changing the thresholds changes the list.
- [ ] Pick a Market in the drill-down: its most affected Product–Market pairs are listed, largest effect first.
- [ ] A Scenario that changes nothing (e.g. adding an agreement with Japan, which already exists, or the Leave preset in 2024) says so.
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
- [ ] The stability sentence and the known-limitations panel (goods only, no tariffs, associations not causes, provisional 2024, the 2012 edition, reporting quality, sanctions) render.
- [ ] The Lift section states how many times more often the largest Export Gaps grew, with a 95% interval, worded as a tendency, and shows the growth share for each tenth of Export Gaps.
- [ ] The Event Backtest shows, for EU exports to Japan and Japan's exports to the EU, a table of predicted against observed change per year, names the control Markets, and states the COVID caveat; the UK–Australia/New Zealand secondary check follows.
