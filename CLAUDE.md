# UK Exports Predictor

A final project for an ML/AI course. It is a model that learns from global HS6 goods trade to estimate the UK's **Export Potential** for each Product and Market. The model finds **Opportunities** and **Blocked Opportunities**, and simulates trade-policy **Scenarios**. The model is served by a FastAPI service and used by government export **Advisors** through a Streamlit app.

## Read first

- `GLOSSARY.md`: the project's language. Use its terms exactly in code names, docstrings, notebook prose and UI text. When a new domain term settles, add it there.
- `docs/adr/`: decisions that look wrong but are deliberate. Read the relevant ADR before changing how the model trains, what data it uses or how Scenarios are computed.
- The spec and the wayfinder map (label `wayfinder:map`) are GitHub issues: the spec says what is being built, and the map records open decisions. Cited research notes behind those decisions live in `docs/research/`.
- `ML_AI_Concept_Notes.html`: the course notes. They are kept out of git, so read them from the main checkout (`C:\CodingProjectsC\exportspredictor\ML_AI_Concept_Notes.html`), even from a worktree. The project should visibly apply them, then go beyond them. When explaining a choice in the notebook, link it to the course concept it uses or extends.

## Shape

- `notebooks/`: the write-up, an argument a marker reads top to bottom. In real mode it reads the newest artefact and saved results, and trains, fits and builds nothing; charts and tables come from `exportspredictor.reporting`. Notebooks import logic from the package and define no functions.
- `src/exportspredictor/`: the single source of truth for data loading, features, Scenario application and prediction. Both the notebook and the API import it, so training and serving compute features identically.
- `api/`: FastAPI. It loads saved model artefacts and precomputed tables, and holds no training code.
- `app/`: Streamlit. It talks to the model only through the API over HTTP.
- `data/`: raw downloads and derived Parquet, kept out of git. Every derived file is reproducible from a script.

## Modelling guardrails

These encode the project's core integrity. Each one exists because breaking it silently inflates results.

- **UK Flows are scored out-of-fold** (ADR 0001). Every UK Export Potential, Export Gap and Scenario Effect comes from the fold model that held that Product–Market pair out. Folds group a pair's years together.
- **Leave-one-out supply features**: any feature built from UK data excludes the target pair's own Market.
- **Time split**: fit on earlier years and evaluate on later ones. Fit scalers, encoders and aggregates on training years only.
- **The test year is read once per frozen version** (ADR 0009). Every choice is made on 2022 and fixed before 2023 is opened, and no 2023 result feeds back into that version. Every read goes through the Test-year Ledger (`docs/test-year-ledger.csv`, `exportspredictor.ledger`), which refuses a second read of a Configuration Fingerprint unless `--read-test-year-again` is passed.
- **Policy features are monotone-constrained**, and tariffs are clipped to the training range (ADR 0008 kept the constraints after testing them against the ranking; ADR 0006). Export Potential is the model blended with the naive baseline (ADR 0008). Scenarios change only policy features and re-predict through `prediction.export_potential`; the naive baseline has no policy features, so a Scenario leaves it unchanged.
- **Language about results stays associational**: "the model associates a Trade Agreement with +X%", not "the agreement causes +X%". Lift is a tendency, not a forecast.

## Environment

- Windows 11, PowerShell. Use XGBoost with `device="cuda"` and Polars with lazy Parquet scans, so that HS6 data never has to sit fully in memory.
- Hardware, the budget for every model and tuning choice:

  | Part | Spec | Limit to plan around |
  |---|---|---|
  | GPU | RTX 3080, **10 GB VRAM** (driver 610.88) | The binding limit for GPU training. Build training data as `QuantileDMatrix` and keep sampled training rows within VRAM, or use XGBoost external memory. |
  | CPU | Ryzen 7 5800X, 8 cores / 16 threads | Polars feature builds and CPU prediction. |
  | RAM | 32 GB | Peak for feature builds. Use compact types and never build the dense Exporter × Importer × Product grid. |
  | Disk | about 690 GB free | Not a constraint. |
- Python is managed with `uv`. Containers are run with `podman compose`.
- **Gotchas** of this machine (uv off PATH, worktrees and the shared `.venv`, shared data and artefacts, long GPU runs, `gh` bodies): `docs/agents/environment.md`.

## Agent skills

### Issue tracker

Issues, specs and the wayfinder map live in GitHub Issues on `noahlb1/exportspredictor`, managed through `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Unattended runs

`/implement-spec` runs unattended. It and its subagents take the recommended option instead of asking, and log each decision. See `docs/agents/unattended-runs.md`.
