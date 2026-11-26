# UK Exports Predictor

A model that learns from global HS6 goods trade to estimate the UK's **Export Potential** for each Product and Market. It ranks **Opportunities** and **Blocked Opportunities** and simulates trade-policy **Scenarios**. A FastAPI service serves the model, and Advisors use it through a Streamlit app. Terms follow [GLOSSARY.md](GLOSSARY.md).

Results are associations the model has learned, not causes.

## Run the app

No GPU and no Python needed. You need [podman](https://podman.io/) with `podman compose`, or Docker; about 5 GB of free RAM; and 2.5 GB of disk while fetching the trained artefact (a 0.9 GB download that unpacks to 1.6 GB).

```powershell
git clone https://github.com/noahlb1/exportspredictor.git
cd exportspredictor
podman compose run --rm fetch   # or: docker compose run --rm fetch
podman compose up               # or: docker compose up
```

Then open <http://localhost:8501>. Stop with `Ctrl+C` or `podman compose down`.

- **The fetch** downloads the artefact from the [`v1.1` release](https://github.com/noahlb1/exportspredictor/releases/tag/v1.1), checks its SHA-256 and unpacks it into `./artefacts`. It skips a version already there.
- **By hand**, if the fetch fails: download `20261126T204856Z.zip` and `20261126T204856Z.zip.sha256` from the release page, run `sha256sum -c 20261126T204856Z.zip.sha256` (in PowerShell, compare `Get-FileHash 20261126T204856Z.zip` with the hash in the file), then unzip into `artefacts/`, which gives `artefacts/20261126T204856Z/`.
- **Timings**: `compose up` is ready in about 20 s once the image is built. On two cores of a Ryzen 5800X the API loads the artefact in 3.5 s, a Trading Bloc preset takes 2.4–3.6 s and a custom Scenario about 1 s.
- **If the app says the API is unavailable**, check `podman compose logs api` and <http://localhost:8000/health>, which gives the reason. The API serves only the lexically newest folder in `./artefacts`, and only artefact format 4.
- On Windows, if `podman compose` cannot connect after a reboot, run `podman machine start`.

**Read the argument** in the notebook, [`notebooks/exports_predictor.ipynb`](notebooks/exports_predictor.ipynb), or on the app's [How the Model Works](http://localhost:8501/How_the_Model_Works) page.

## The served model

Export Potential blends an XGBoost model (weight 0.7) with the Naive Baseline (weight 0.3) on the log scale, then scales every UK pair in a Market by its **Market Reach**: how widely the UK already sells there (ADR 0010). Every choice was made on 2022, on the **Selection Measure**: how well the model ranks Markets for each Product on held-out UK pairs. A change was kept only if it beat the Win Threshold (ADR 0008, [research note](docs/research/opportunity-ranking-experiments.md)).

**Result on 2023, read once**: the model ranks UK Markets at 0.603 against the Naive Baseline's 0.514 (rank correlation with UK Actual Exports for a typical Product; 1 is a perfect ranking).

Scenario Effects and Blocked Opportunity values are **Illustrative**: upper-end associations shown beside a PPML gravity benchmark and the Event Backtests, never forecasts.

**Known limitations**, also in the model card and on the Model Evaluation page:

- **It ranks UK Markets better than it sizes UK Flows.** On held-out UK pairs in 2023, its Tweedie deviance is 2,611, against 2,095 for the Naive Baseline (lower is better). Trust rankings more than dollar values.
- **The Potential Range lies below Export Potential for small pairs** (under about $8k). Most comparable pairs of that size trade nothing, and Export Potential is an average that the few which do trade pull up.
- **A Tariff part is not a tariff elasticity.** The model's response to a tariff is more a step than a slope, so removing a tariff of 1% or less can be associated with a rise of tens of percent (turbo-jets to China: +37%).
- **Some Trade Agreements carry most-favoured-nation tariffs.** TRAINS records no preferential rates for some agreements in force, among them Canada's and Israel's, and none for UK goods under the UK–Türkiye agreement, so those pairs read as dutiable.

The last three come from a review of the served outputs by a reader who had not seen the code.

The model trains on the top 100 Exporters into every Market, leaves out Excluded Products, uses the Applied Tariff, Relatedness and the re-export ratio as features, averages Export Gaps over the 2022–2024 Gap Period and gives each pair a Potential Range. The notebook and the model card (`model_card.md` in the artefact folder) give the detail; the [Model History](notebooks/model_history.ipynb) is the full record of how the model was reached.

## Repo map

**Start reading here:** the notebook, [`notebooks/exports_predictor.ipynb`](notebooks/exports_predictor.ipynb), argues the project from the data to the results; then the package, [`src/exportspredictor/`](src/exportspredictor/), holds the logic it calls.

| Path | What it holds |
| --- | --- |
| `notebooks/` | The write-up, and the Model History. |
| `src/` | The package, `exportspredictor`: data loading, features, training, evaluation, Scenarios and prediction. Its experiment loop is in `loop/`. |
| `api/` | The FastAPI service over the saved artefacts. |
| `app/` | The Streamlit app Advisors use. |
| `tests/` | The pytest suite, on a small fixture world. |
| `scripts/` | `rebuild.py` rebuilds every derived file and artefact (below). `research/` holds the scripts behind the research notes, `gpu_check.py` measured the GPU row budget (ADR 0004), and `wt.ps1` runs a command in a git worktree against the main checkout's environment. |
| `docs/` | The decision records in [`adr/`](docs/adr/), the research notes in [`research/`](docs/research/README.md) and the Test-year Ledger, `test-year-ledger.csv`. |
| `.githooks/` | The pre-push hook: ruff, pyright and the suite. |
| [GLOSSARY.md](GLOSSARY.md) | The project's terms. |
| [CODING_STANDARDS.md](CODING_STANDARDS.md) | The modelling guardrails, the rules for each folder and the hardware budget. |
| `Containerfile`, `compose.yaml` | The containers. |
| `pyproject.toml`, `uv.lock` | The environment. |
| `UK_Exports_Predictor_presentation.pptx` | The presentation. |

`data/` is not tracked, and `artefacts/` holds only a placeholder: the fetch fills `artefacts/`, and the rebuild makes both.

## Development with uv

Python is managed with [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                                # install dependencies, including dev tools
uv run pytest                          # run the tests (includes a fixture-mode notebook run)
uv run uvicorn api.main:app --reload   # API on http://localhost:8000
uv run streamlit run app/Home.py       # app on http://localhost:8501
```

While the API runs, <http://localhost:8000/docs> lists every endpoint and lets you call each one from the browser.

Settings are environment variables:

| Variable | Default | Used by |
| --- | --- | --- |
| `EXPORTSPREDICTOR_DATA_DIR` | `<repo>/data` | data download and build steps, notebook |
| `EXPORTSPREDICTOR_ARTEFACTS_DIR` | `<repo>/artefacts` | pipeline (writes), API (reads), notebook (reads; writes only its fixture artefacts) |
| `EXPORTSPREDICTOR_API_URL` | `http://localhost:8000` | app |

- **Once per clone**, enable the pre-push hook: `git config core.hooksPath .githooks`. It runs ruff, pyright and the full suite before every push (6 to 10 minutes).
- **On Windows**, set `PYTHONUTF8=1` before running Python. The default cp1252 encoding crashes on printing an accented Market name and garbles UTF-8 files. Subprocesses such as the notebook kernel inherit the variable, which `-X utf8` does not reach.

## Rebuild the data and artefacts

The rebuild needs an NVIDIA GPU, about 31 GB of data and hours of compute. It is not needed to run the app: [Run the app](#run-the-app) fetches the trained artefact.

```powershell
uv run python scripts/rebuild.py --list            # show the steps
uv run python scripts/rebuild.py                   # run them all, in order
uv run python scripts/rebuild.py --only notebook   # refresh the committed notebook's outputs only
```

The script runs the steps below in order and stops at the first that fails. Long steps cache their results under `data/derived/`, so a re-run resumes. A full rebuild takes hours on the GPU; times are from the machine in [the hardware budget](CODING_STANDARDS.md#hardware-budget).

Every step runs as `python -m exportspredictor.<command>`. The steps are the list `STEPS` in [scripts/rebuild.py](scripts/rebuild.py).

**1. Download the sources and build the features**

| Step | Command | Notes |
| --- | --- | --- |
| BACI Flows and lookups (download, convert to Parquet) | `baci` | Idempotent. 1.3 GB download, 30 s convert, 2.5 GB peak memory. |
| Gravity and WDI features (download CEPII Gravity, WDI, IMF WEO; build Parquet) | `gravity` | Needs BACI's country codes. 210 MB download, 40 s build. |
| Trade Agreement and Trading Bloc features (WTO RTA-IS download, build) | `policy` | Skips the download if `raw/wto/AllRTAs.xlsx` exists. 2 s build. |
| Applied Tariffs (TRAINS download via WITS, HS17/HS07 -> HS12, EU schedule, gap fill) | `tariffs` | UNCTAD TRAINS for 2012–2021 (about 1,300 files, 62 MB, resumable); writes `derived/tariffs/` (ADR 0006). |
| Feature table (training rows with sampled zeros, UK grid; per year) | `features` | Needs the four steps above. Writes `derived/features_V202601/` (5.7 GB): top-100 Exporter scope, every UK Zero Cell, 5% of other Zero Cells. 15 minutes, 14 GB peak memory. |
| HMRC OTS 2023 exports for the BACI cross-check (download, HS 2022 -> 2012) | `hmrc` | 36 MB download. Used only by the notebook's cross-check, never by the model. |

**2. Make every modelling choice on 2022**

| Step | Command | Notes |
| --- | --- | --- |
| Experiment rows (Sector subset, 2012–2022) | `experiments build` | The rows the ablation and the learning curve share. |
| Feature ablation on 2022 (GPU) | `experiments ablation` | 2–4 minutes a model. Results in `derived/experiments_V202601/results.json`. |
| Learning curve on 2022 (GPU) | `experiments curve` | Nine models, about half an hour. |
| Hyperparameter search on 2022 (GPU, resumable, about 3 h) | `search` | Optuna TPE. The study is in `derived/search_V202601/study.sqlite3`. |
| Experiment loop on 2022: noise (GPU, resumable) | `loop noise` | The starting configuration with three seeds on fold 0: the Noise Floor and the Win Threshold every later Arm must beat. |
| Experiment loop on 2022: naive (GPU, resumable) | `loop naive` | The blend with the Naive Baseline (weight chosen from 0.1 to 0.9) and the starting-point model. |
| Experiment loop on 2022: select (GPU, resumable) | `loop select` | Five configurations, from the saved search and the project's first model, each with its own blend weight. |
| Experiment loop on 2022: two-part (GPU, resumable) | `loop two-part` | A binary model of any Flow times a size model on non-zero Flows. |
| Experiment loop on 2022: scope (GPU, resumable) | `loop scope` | UK Zero Cells sampled like other Exporters', then a 2% zero rate. |
| Experiment loop on 2022: product-detail (GPU, resumable) | `loop product-detail` | Three Product-level features, each added on its own. |
| Experiment loop on 2022: constraints (GPU, resumable) | `loop constraints` | The best Arm without the monotone constraints. |
| Five-fold confirmation of the loop's best form on 2022 (GPU, resumable, about 1 h) | `loop confirm` | The best Arm beside the model alone, on every held-out UK pair of five folds. |

Every Arm of the loop is judged on the Selection Measure. Its results are in `derived/loop_V202601/results.json` and its models are cached beside them in `models` (about 9.5 GB). Keep `results.json` and `confirmation/`: the notebook and the pipeline read them. The loop's package docstring says how to add an Arm.

**3. Train, evaluate on 2023 and write up**

| Step | Command | Notes |
| --- | --- | --- |
| Train the fold and test-year models, evaluate and save the artefacts (GPU, about 2.5 h) | `pipeline` | Trains the five fold models and the 2023 test-year models (about 75 minutes each), serves the Model Form the loop chose, and saves a new artefact version (format 4) with its `model_card.md`. |
| The Model History (reads the saved artefact, the Test-year Ledger and the loop's results) | | Runs `notebooks/model_history.ipynb`, the record of how the model was reached, on the real data and saves its outputs into the committed file. It trains nothing and reads no older artefact folder; under a minute on the CPU. |
| The notebook (reads the saved artefact) | | Runs `notebooks/exports_predictor.ipynb` on the real data and saves its outputs into the committed file. It reads saved results only and trains nothing; about 2 minutes on the CPU. |

### The pipeline step

- **It reads 2023.** Before evaluating, it records the read in the Test-year Ledger, which may refuse it (the test-year rule in the [modelling guardrails](CODING_STANDARDS.md#modelling-guardrails)). Commit the ledger with the run.
- **Three parts run only when asked for**: `--event-backtests` (Lift included), `--constraint-comparison` and `--ppml-benchmark`. The Model Evaluation page and the model card show a skipped part as not run.
- **History-only commands** are in the Model History's [Reproduce this record](notebooks/model_history.ipynb).

### Outside the script

- **The notebooks' outputs**: `uv run python scripts/rebuild.py --only notebook` (or `--only history` for the Model History) is the one way to refresh them. Rerun it after any edit to the notebook.
- **The PPML gravity benchmark**: `python -m exportspredictor.ppml` derives a new version from the newest one, with the benchmark saved beside it and the model card rewritten (about 25 minutes on the CPU; no training, and the 2023 Flows are not read).

### Change an artefact without retraining

Each command copies the newest version into a new one, from its saved fold models. An artefact missing a table that `artefacts.load` expects fails to load, so derive a new version after adding a table.

| Command | What changes | Cost |
| --- | --- | --- |
| `python -m exportspredictor.pipeline refresh` | Fresh metadata. | Seconds |
| `python -m exportspredictor.pipeline rerange` | The Potential Range, recomputed from the Baseline year's feature file. | A few minutes, CPU |
| `python -m exportspredictor.pipeline recaveat` | The Caveat Tiers on the Baseline. Run it after adding a tier or changing a tier's table. | Seconds |
| `python -m exportspredictor.pipeline rereach` | Format 3 becomes format 4, with Market Reach. | |

### Serve a new version in the containers

The API container mounts `./artefacts` read-only, so a new version needs only `podman compose restart api`. The API refuses a format-3 artefact: `python -m exportspredictor.pipeline rereach` upgrades it. After a code change, `podman compose up --build` rebuilds the image.

## Manual UI checklist

Run [docs/ui-checklist.md](docs/ui-checklist.md) before a release: it says what to click on every page of the app.
