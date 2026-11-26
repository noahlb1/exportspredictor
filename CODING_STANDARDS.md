# Coding standards

Read in review. "Shape" and "Modelling guardrails" are the project's rules; the sections after them are judgement calls. GLOSSARY.md applies too: use its terms exactly in code names, docstrings, notebook prose and UI text, and add a domain term there when it settles. The ADRs in `docs/adr/` record decisions that look wrong but are deliberate: read the relevant one before changing how the model trains, what data it uses or how Scenarios are computed.

## Shape

The README's [repo map](README.md#repo-map) says what each folder holds. The rules for them:

- `notebooks/`: an argument a marker reads top to bottom. In real mode the notebook reads the newest artefact and saved results, and trains, fits and builds nothing; charts and tables come from `exportspredictor.reporting`. Notebooks import logic from the package and define no functions.
- `src/exportspredictor/`: the single source of truth for data loading, features, Scenario application and prediction. Both the notebook and the API import it, so training and serving compute features identically.
- `api/`: loads saved model artefacts and precomputed tables, and holds no training code.
- `app/`: talks to the model only through the API over HTTP.
- `data/`: raw downloads and derived Parquet, kept out of git. Every derived file is reproducible from a script.

## Modelling guardrails

These encode the project's core integrity. Each one exists because breaking it silently inflates results.

- **UK Flows are scored out-of-fold** (ADR 0001). Every UK Export Potential, Export Gap and Scenario Effect comes from the fold model that held that Product–Market pair out. Folds group a pair's years together.
- **Leave-one-out supply features**: any feature built from UK data excludes the target pair's own Market. The one exception is Market Reach, which is post-processing in the Model Form, not a feature: it reads UK Flows into the pair's Market in other Products, only from other folds' pairs and the year before (ADR 0010).
- **Time split**: fit on earlier years and evaluate on later ones. Fit scalers, encoders and aggregates on training years only.
- **The test year is read once per frozen version** (ADR 0009). Every choice is made on 2022 and fixed before 2023 is opened, and no 2023 result feeds back into that version. Every read goes through the Test-year Ledger (`docs/test-year-ledger.csv`, `exportspredictor.ledger`), which refuses a second read of a Configuration Fingerprint unless `--read-test-year-again` is passed.
- **Policy features are monotone-constrained**, and tariffs are clipped to the training range (ADR 0008 kept the constraints after testing them against the ranking; ADR 0006). Export Potential is the model blended with the naive baseline (ADR 0008). Scenarios change only policy features and re-predict through `prediction.export_potential`; the naive baseline has no policy features, so a Scenario leaves it unchanged.
- **Language about results stays associational**: "the model associates a Trade Agreement with +X%", not "the agreement causes +X%". Lift is a tendency, not a forecast.

## Hardware budget

Every model and tuning choice is planned around this machine. Training uses XGBoost with `device="cuda"`, and feature builds use Polars with lazy Parquet scans, so that HS6 data never has to sit fully in memory.

| Part | Spec | Limit to plan around |
|---|---|---|
| GPU | RTX 3080, **10 GB VRAM** | The binding limit for GPU training. Build training data as `QuantileDMatrix` and keep sampled training rows within VRAM, or use XGBoost external memory. |
| CPU | Ryzen 7 5800X, 8 cores / 16 threads | Polars feature builds and CPU prediction. |
| RAM | 32 GB | Peak for feature builds. Use compact types and never build the dense Exporter × Importer × Product grid. |
| Disk | about 690 GB free | Not a constraint. |

## Where logic lives

- **`api/` only translates HTTP.** It validates, calls `src/exportspredictor`, and shapes the response. Ranking, sorting, filtering, Scenario wording, money formatting and currency conversion belong in the package, so the notebook gets the same behaviour.
- **One definition per constant or option.** Thresholds, ranking orders and preset names are defined once in the package. The API's schemas derive from that definition, and the app reads it from an endpoint (as `/opportunities/rank-by` does) rather than copying the literal.
- **A query has one entry point.** A new filter or order on rankings goes into `rankings.rank_opportunities`, not into a second sort in the API.

## Names

- **Money carries its unit**: `_usd` or `_gbp` on every parameter, column and field. A bare `min_gap` hides which.
- **Domain names come from GLOSSARY.md**, including for derived columns (a column holding the Blocked Opportunity value says so). Vocabulary from a source's own format (BACI's or HMRC's "partner") is fine where the code describes that source.

## Writing

- **Concise and tight, everywhere**: app text and help, model-evaluation text, the How the Model Works page, the notebook's prose, docs, docstrings and ticket comments. Say it once, in as few words as carry the meaning. Cut preamble, restated headings, stacked hedges and lists of everything considered. A rewrite is no longer than what it replaces unless it adds a missing unit, period or explanation. Agents tend to over-write: draft, then cut by about a third.
- **Product text names earlier configurations by role** ("the model alone", "the starting configuration"), never by version name (`v1`, `v2`). The history belongs to the Model History; `tests/test_product_surface.py` guards each enrolled surface (ADR 0013).

## Leakage

- **Check indirect paths.** Beyond the explicit leave-one-out features, ask whether an aggregate built from other rows (Market demand, income shares, RCA) still contains the held-out UK Flow. Note any such path in ADR 0001 or the notebook rather than leaving it implicit.

## Tests

- **Tests run on the fixture, on the CPU.** A test on real data or artefacts skips when they are absent (as `tests/test_policy.py` does) and stays fast; anything needing the GPU, the network or a retrain is a run evidenced in the ticket, not a test. The suite stays within about ten minutes; a test in pytest's `--durations` list that takes more than 10 s needs a reason.
- **A schema change ships with a contract test** in `tests/test_api.py`, and a new page with its entry in `docs/ui-checklist.md` (enforced by `tests/test_readme.py`).
