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
| Trade Agreement and Trading Bloc features (WTO RTA-IS) | arrives with #13 |
| Gravity and WDI features (download CEPII Gravity, WDI, IMF WEO; build Parquet) | in place: `python -m exportspredictor.gravity`; needs the BACI step first (country codes); about 210 MB download, 40 s build |
| Train, evaluate and save artefacts (notebook) | in place; runs in fixture mode until real-data mode lands |

## Manual UI checklist

Run this before a release, against `podman compose up` with real artefacts. Tick each item.

**UK Exports Predictor (home page)**
- [ ] The page loads, and the banner states the Baseline year, that it is provisional, and the model version.
- [ ] Stop the API (`podman compose stop api`) and reload: every page shows a clear "API is unavailable" message instead of an error. Start it again with `podman compose start api`.

**Product → Markets**
- [ ] Search for a known HS6 code (e.g. `220830`, whisky) and for words (e.g. `smartphone` finds `851712`): the Product's code and description appear, with Markets ranked by Export Gap, largest first.
- [ ] Export Potential, Actual Exports and Export Gap appear side by side.
- [ ] Search text that matches nothing (e.g. `999999` or `zzz`) shows a clear "No Product matches" warning.
- [ ] Export the ranking to CSV and open it: the columns and row count match the table on screen. (CSV export arrives with #17.)

**Market → Products** (arrives with #17)
- [ ] Pick a Market: Products are ranked by Export Gap, with Sector, region and minimum-Gap filters working.
- [ ] Values show in £, with the conversion rate and year stated.
- [ ] Export the ranking to CSV and open it.

**Scenarios** (arrives with #18)
- [ ] Run a Trading Bloc preset Scenario (e.g. EU re-join). The summary shows the total change, the change per Market and the top Products gaining, phrased as associations.
- [ ] Blocked Opportunities unlocked by the Scenario are listed, and changing the thresholds changes the list.
- [ ] A Scenario that changes nothing (e.g. adding an agreement that already exists) says so.

**Why?** (arrives with #19)
- [ ] Open "Why?" for a Product–Market pair: the factors are described in plain words, and the recent Actual Exports are shown.

**Model Evaluation** (arrives with #20)
- [ ] The accuracy headline, Lift, Sector and region breakdowns, the Event Backtest and the known-limitations panel all render.
