"""The experiment loop (spec #65, ticket #67): every modelling choice is kept or dropped on one selection measure.

ADR 0008 makes Opportunity ranking the one objective. This package fixes, in code, how a
choice is measured and when it counts as a win:

- **The selection measure** (`SELECTION_MEASURE`): the mean over Products of the Spearman
  correlation between Export Potential and Actual Exports across Markets, on the held-out
  fold's UK pairs in the validation year (2022), on every non-Excluded Product. It is the
  headline measure of `evaluation` moved to the validation year, and both call the one
  function `exportspredictor.selection_measure.selection_measure`: ties in Export Potential
  are broken by a stable hash of Product and Market, so tying the bottom of a ranking (an
  early tree, a floor) cannot raise it (spec #113, ADR 0008 amended). Nothing here selects on
  Tweedie deviance: it is reported with the other diagnostics (`measures`), never used.
- **An arm** (`Arm`) is a named set of settings that differ from the best arm so far.
  Running it trains one model on the training years (2012–2021) with one fold of UK pairs
  held out (ADR 0001), scores the validation year, and returns one result row. The round
  count is chosen by the selection measure itself, evaluated on the held-out UK grid every
  round, with early stopping after `Settings.patience` rounds without improvement.
- **The noise floor** (`Loop.noise_floor`) runs first: the current v2 configuration with
  `LoopPlan.noise_seeds`, each seed changing the non-UK hash sample and the XGBoost seed
  together. The win threshold is twice the standard deviation of the selection measure
  over those seeds, and at least `MIN_WIN_THRESHOLD`. Ticket #116 re-scored the seeds under the
  tie-proof measure (`recheck`); every decision since reads the threshold through
  `current_win_threshold`, which prefers the re-checked one.
- **The win rule** (`decide`): in an experiment, the arm with the highest selection
  measure is marked best; it is kept, and becomes the best so far, only if it beats the
  best so far by more than the win threshold. Every other arm is dropped.

Trained models are cached by the settings that decide them (`training_key`), so an arm
that changes only how a saved model is scored (`SCORING_FIELDS`) trains nothing. Results
are saved as each arm finishes, and a re-run skips finished arms, so a killed run resumes.

**The test year (2023) is never read**: the loop's data is `experiments.Data`, which
refuses it, and `real_data` refuses it before opening a file.

    uv run python -m exportspredictor.loop noise   # GPU, three arms of about 11 minutes each
    uv run python -m exportspredictor.loop naive   # #70: the blend (no training) and the starting point (one model)
    uv run python -m exportspredictor.loop select  # #71: v1, three confirmed v2 configurations, depth 6 (four models)
    uv run python -m exportspredictor.loop two-part  # #72: a binary "any Flow" part times a size part (two models)
    uv run python -m exportspredictor.loop scope   # #73: UK Zero Cells sampled; the zero rate at 2% (two models)
    uv run python -m exportspredictor.loop product-detail  # #74: three Product features, one experiment each (three models)
    uv run python -m exportspredictor.loop constraints  # #75: the monotone constraints on (cached) and off (one model)
    uv run python -m exportspredictor.loop confirm  # #76: the best arm beside v2 on all five folds (fold 0 cached)
    # #127, with EXPORTSPREDICTOR_FEATURE_SET=exporter_count: the best so far retrained on the rebuilt
    # files, then the Exporter count against it, both with Market Reach (two models)
    uv run python -m exportspredictor.loop exporter-count
    # #163, with EXPORTSPREDICTOR_FEATURE_SET set to the served files (`kept_feature_set`): the best so far as
    # served against it with Market Reach's upward multiplier capped (no training)
    uv run python -m exportspredictor.loop market-reach-cap
    # #164, with EXPORTSPREDICTOR_FEATURE_SET=sibling_demand (the served `exporter_count` files with Sibling
    # Demand stored): the best so far as served retrained on them, then Sibling Demand added (two models)
    uv run python -m exportspredictor.loop sibling-demand
    # #166, with EXPORTSPREDICTOR_FEATURE_SET set to the served files: the hyperparameter search, one stage per
    # command, in this order (each refuses to run before the ones above it are decided); one model per Arm
    uv run python -m exportspredictor.loop hyperparameters depth
    uv run python -m exportspredictor.loop hyperparameters learning-rate
    uv run python -m exportspredictor.loop hyperparameters min-child-weight
    uv run python -m exportspredictor.loop hyperparameters subsampling
    uv run python -m exportspredictor.loop hyperparameters regularisation
    # #167, with EXPORTSPREDICTOR_FEATURE_SET set to the served files: another model family against the best so
    # far as served, one model each, trained on the CPU (one at a time; each refuses other feature files)
    uv run python -m exportspredictor.loop model-family lightgbm
    uv run python -m exportspredictor.loop model-family catboost
    # #169, with EXPORTSPREDICTOR_FEATURE_SET=all_exporters (every Exporter's rows): the best so far as served
    # retrained on the top 100 Exporters of those files, then on every Exporter, both at one non-UK rate (two models)
    uv run python -m exportspredictor.loop wider-scope
    # #170, with EXPORTSPREDICTOR_FEATURE_SET set to the served files: the best so far as served (cached) against it
    # with every other Exporter's row (non-UK rate 1.0) through XGBoost external memory (one model, 8.2 h on this GPU)
    uv run python -m exportspredictor.loop external-memory
    # #171, with EXPORTSPREDICTOR_FEATURE_SET=excluded_products (`exporter_count` plus the Excluded Products' training
    # rows): the best so far as served with them dropped, then trained on them, both at one non-UK rate (two models)
    uv run python -m exportspredictor.loop excluded-products

The results are associations measured on one fold and one year: an arm that wins ranks
the held-out UK Markets better in 2022, which says nothing about why.

The package, one module per concern; everything public is importable from `exportspredictor.loop`:

- `selection`: the selection measure, its diagnostics and the win rule.
- `arms`: `Settings`, `Arm`, `LoopPlan`, the cache key and the seven experiments' arms.
- `data`: the fixture and real data, and the results file.
- `runner`: `Loop`, which runs and caches arms on one fold.
- `trainers`: the trainer seam, one `Trainer` per model family (`Settings.family`, XGBoost by default).
- `confirmation`: the five-fold confirmation.
- `served`: what the pipeline serves of the loop's choice.
- `recheck`: the noise floor, Win Threshold and best so far re-scored under the tie-proof measure (#116).
- `cli`: the commands above (`__main__` runs them).

Working on the loop
-------------------

**Adding an experiment.** An arm's `changes` replace fields of the best so far's `Settings`,
except `params`, which is merged into the best's params, so a full configuration names
every parameter. A new kind of change adds a `Settings` field whose default means "as
before": a training field, or a scoring field listed in `SCORING_FIELDS` if it only changes
how a saved model is scored. `training_key` hashes a later field only when it leaves its
default, so every cached model keeps its key; leave `KEYED_FIELDS` as it is. An arm that
re-chooses the blend weight carries `grid=BLEND_GRID` (scoring only, seconds). Each
experiment gets a command in `cli`, so it can be launched in the background, and trains at
most `MAX_TRAINED_MODELS`; a two-part arm counts as two.

**Adding a trainer** (another model family, #165). Training goes through `trainers.Trainer`: the
`Loop` builds the rows (`TrainingRows`: the training chunks, the per-chunk `prepare`, the held-out
UK grid, the features and a key that is the same whenever the rows are) and a `TrainingJob` (the
family's `params`, the seed, `monotone`, the round cap, `patience`, the Selection Measure of grid
predictions to stop on, the device and threads, and `binary` for a two-part model's "any Flow"
part). A trainer returns the model at its best round and its metadata (`rounds`, `hit_round_cap`,
`trained_rounds`, `stopping_measure`, `training_rows`, `matrix_seconds`, `train_seconds`), saves
and loads models under its own file `suffix`, and `release`s what it kept from the last rows. To
add one: write the class in `trainers`, name it in `TRAINERS`, and widen `prediction.Model` where
`prediction.export_potential` learns to predict with its models (Scenarios go through that
function). It must enforce the monotone policy constraints (`features.MONOTONE_CONSTRAINTS`) when
`monotone` is set, or refuse the job: a family that cannot is not run. An arm names it with
`{"family": ...}`, and its `params` are that family's own. `family` at its default (`XGBOOST`) is
left out of `Settings.to_dict` and so of `training_key`, so every model cached before it existed
is still found. A setting only one family reads (an XGBoost external-memory switch, say) is a
training field passed on in `TrainingJob`, keyed like any later field. An Arm that names another
family than its base's replaces the parameters with its own instead of merging them.

`LIGHTGBM` and `CATBOOST` (#167) train on the CPU from the same rows, written once to a float32 file
in a temporary folder so the raw rows are never all in memory: the LightGBM wheel has no GPU learner,
and CatBoost enforces monotone constraints only on the CPU and only for symmetric trees (it refuses
others). Neither trains a two-part model or from a starting point. Their pre-registered Arms are
`MODEL_FAMILY_ARMS`, each run by `run_model_family` against the best so far as served.

**Cache and results** live in `loop_dir()` (`data/derived/loop_V202601/`), shared by every
worktree through `scripts/wt.ps1`. Models are `models/<training_key>.ubj` (the XGBoost trainer's
suffix; `.lgbm` and `.cbm` for LightGBM and CatBoost) with a `.json` of
their rounds; `results.json` holds the naive baseline, the noise floor, every experiment's
rows and decision, `best` and `confirmation` (all measured with the old measure), and `recheck`
(#116: the same decisions under the tie-proof measure, the recomputed noise floor and Win
Threshold, and the re-scored best so far that `best` now holds). `load()` reads it. A re-run skips arms by
name, so to re-run an arm, give it a new name; the file is written only by the loop.

**Feature sets.** `real_data` reads the feature set `EXPORTSPREDICTOR_FEATURE_SET` names
(`features.FEATURE_SETS`; the original `features_V202601` when unset). A rebuilt set, such as
`exporter_count` (#126: the Exporter count stored as a candidate feature, and GDP filled for small
Markets), shares this folder and results file, but its models are cached under their own keys:
the set is part of the cache key, so a base arm on rebuilt files trains afresh instead of reusing
the original's model. `all_exporters` (#168) is `exporter_count` with every Exporter's rows, not
only ADR 0004's top 100; its rows are unsampled, so an arm on it needs a non-UK rate that fits the
VRAM budget (`non_uk_rate=None` derives the largest, `search.final_non_uk_rate`). The Exporter count is not in `features.FEATURES`; an arm tries it by naming
it in `features`. An experiment on rebuilt files passes `base_arm` to `Loop.run_experiment`: the best
so far is retrained on those files and the arms are decided against it (a paired comparison), and the
section records the feature set, which `served_configuration` hands the pipeline (it refuses to build
from other files).

**Market Reach** (#124) is the scoring field `market_reach_beta`: an arm with it multiplies each
held-out UK pair's Export Potential by exp(beta · offset) of its Market, learned on this fold's
complement in the year before (`LoopData.reach_source`, the same feature set's UK grid), and records
its measures without Reach under `without_market_reach`. The rescaling constant is computed per
scored table, so pooling several folds' Reach scores ranks with a slightly different constant per
fold; a five-fold figure should re-pool the folds' plain scores with one constant, as #124 did.

**The final round** (spec #155) stacks its ideas on the best so far *as served*. Market Reach was adopted
outside the loop (#124), so `results["best"]` carries it only once an Arm of the round is kept. Each
experiment of the round therefore passes `base_arm=served_base_arm(best)` (Market Reach at `CHOSEN_BETA` added
where the best so far has none) and runs on the files `served_configuration` serves (`kept_feature_set`).
Its first, #163, caps the upward Market Reach multiplier (the scoring field `market_reach_cap`, None: no
cap); a kept Arm becomes `results["best"]` with Market Reach and any cap in its settings. An idea on its own
rebuilt files (#164's `sibling_demand`: Sibling Demand stored beside the served columns) passes
`files_kept_if_dropped=False` to `Loop.run_experiment`, so a dropped Arm leaves `kept_feature_set` on the served
files. The hyperparameter search (#166, `run_hyperparameter_stage`) is a sequence of experiments, one per stage
of `HYPERPARAMETER_STAGES`, each against `served_base_arm` of the best so far after the stages before it; Arms
whose best round is below `STOPPING_RULE_ROUNDS` are listed by `decided_by_stopping_rule`. The wider Exporter scope (#169,
`run_wider_scope`) trains on `all_exporters` with the training field `exporter_scope` (the top n Exporters by mean GDP,
`scope.exporter_scope`; None: every Exporter the files hold): its base keeps ADR 0004's top 100 of the same files, its Arm
every Exporter, both at `WIDER_SCOPE_NON_UK_RATE`. Every experiment now refuses, before training, an arm whose non-UK rate
might not fit the VRAM row budget (`LoopPlan.row_budget`, by default `training.row_budget`), unless the arm trains
through external memory (#170, `run_external_memory`): the training field `external_memory` (XGBoost only, False by
default and then left out of the key) builds the rows as an `ExtMemQuantileDMatrix`, one page per year cached in host
memory and streamed to the GPU each round, so a non-UK rate of 1.0 fits. Training on the Excluded Products (#171,
`run_excluded_products`) runs on `excluded_products` (`exporter_count` plus their training rows) with the training field
`train_excluded_products` (False by default, left out of the key: the rows are dropped, as ADR 0005 and the loop always
have); its base drops them and its Arm keeps them, both at the largest non-UK rate those files allow. The round's note is
`docs/research/final-round-2022.md`.

**Pitfalls.**

- Rounds stop on the selection measure of the *model alone*, while an arm that blends is
  selected on the blend. A model that is weak at first stops early: a best round under
  about 50 (`rounds` in the row) means the stopping rule decided the result, not the
  configuration. In spec #65 this decided v1, depth 6, the starting point and the two-part
  model's binary part (`docs/research/opportunity-ranking-experiments.md`).
- GPU training is not bit-reproducible: a retrained arm is a fresh draw within the noise
  floor. Re-scoring a cached model is exact.
- One trained arm on every Product takes about 11 minutes on a quiet machine and about
  twice that while other jobs load the CPU, since the per-round measure runs on the CPU.
  Setup holds the whole 2022 validation year (about 13.5 M rows) in RAM.
- `features.subsample_zeros` is correct only with the seed the rows were built with; the
  loop passes the build seed (`LoopData.zero_seed`) whatever the arm's seed. A zero rate
  below the built 0.05 subsamples for free; above it needs a feature rebuild.
- On the fixture the seeds give identical models, so the noise floor's SD is about 0, and
  its grid is too small for the measure to mean anything. A test that needs a sure win sets
  `min_win_threshold=0.0` on the plan.
"""

from exportspredictor.experiments import TestYearRead, log_progress, read_results
from exportspredictor.features import EXCLUDED_PRODUCTS_SET, SIBLING_DEMAND_SET
from exportspredictor.loop.arms import (
    ANY_FLOW,
    BLEND,
    BLEND_GRID,
    BLEND_WEIGHTS,
    CONSTRAINTS_EXPERIMENT,
    EXCLUDED_PRODUCTS,
    EXCLUDED_PRODUCTS_BASE_SET,
    EXCLUDED_PRODUCTS_EXPERIMENT,
    EXPORTER_COUNT_EXPERIMENT,
    EXTERNAL_MEMORY,
    EXTERNAL_MEMORY_EXPERIMENT,
    FIXTURE_LOOP,
    HYPERPARAMETER_STAGES,
    KEYED_FIELDS,
    LOWER_ZERO_RATE,
    MARKET_REACH_CAP,
    MARKET_REACH_CAP_EXPERIMENT,
    MARKET_REACH_CAP_GRID,
    MAX_ROUNDS,
    MAX_TRAINED_MODELS,
    MODEL_FAMILY_ARMS,
    NAIVE_EXPERIMENT,
    NOISE_FLOOR,
    NOISE_SEEDS,
    PATIENCE,
    REBUILT_BASE,
    SCOPE_EXPERIMENTS,
    SCORING_FIELDS,
    SELECT_EXPERIMENT,
    SERVED_BASE,
    SHALLOW_DEPTH,
    SIBLING_DEMAND_BASE_SET,
    SIBLING_DEMAND_EXPERIMENT,
    SIZE,
    STARTING_POINT,
    STOPPING_RULE_ROUNDS,
    TRAINING_FIELDS,
    TWO_PART,
    TWO_PART_EXPERIMENT,
    UK_ZEROS_EXPERIMENT,
    UK_ZEROS_SAMPLED,
    WIDER_SCOPE,
    WIDER_SCOPE_BASE_SET,
    WIDER_SCOPE_EXPERIMENT,
    WIDER_SCOPE_NON_UK_RATE,
    WIDER_SCOPE_SET,
    ZERO_RATE_EXPERIMENT,
    Arm,
    LoopPlan,
    Settings,
    blend_arm,
    constraint_arms,
    count_experiments,
    decided_by_stopping_rule,
    excluded_products_arms,
    experiment_label,
    exporter_count_arms,
    external_memory_arm,
    hyperparameter_experiment,
    market_reach_cap_arm,
    model_family_experiment,
    models_trained_by,
    product_detail_arm,
    product_detail_experiment,
    select_arms,
    served_base_arm,
    sibling_demand_arm,
    starting_point_arm,
    training_key,
    two_part_arm,
    wider_scope_arms,
)
from exportspredictor.loop.cli import confirm_best, main, time_one_arm
from exportspredictor.loop.confirmation import CONFIRMATION, V2, WINNER, confirm
from exportspredictor.loop.data import (
    MODELS_DIR,
    RESULTS_FILE,
    LoopData,
    fixture_data,
    load,
    loop_dir,
    naive_from_features,
    real_data,
    save_results,
    world_products,
)
from exportspredictor.loop.recheck import (
    RECHECK,
    current_win_threshold,
    recheck_results,
    save_recheck,
)
from exportspredictor.loop.runner import (
    NO_MEASURE,
    Loop,
    TwoPart,
    naive_margin,
    run_excluded_products,
    run_exporter_count,
    run_external_memory,
    run_hyperparameter_stage,
    run_market_reach_cap,
    run_model_family,
    run_product_detail,
    run_sibling_demand,
    run_wider_scope,
)
from exportspredictor.loop.selection import (
    DIAGNOSTICS,
    MIN_WIN_THRESHOLD,
    ROUNDING,
    SELECTION_MEASURE,
    SELECTION_MEASURE_TEXT,
    TOP_MARKETS,
    decide,
    measures,
    rank_average_measures,
    win_threshold,
)
from exportspredictor.loop.served import (
    CONFIRMED_NAIVE_WEIGHT,
    RECORDED,
    ServedConfiguration,
    chosen_form,
    fixture_ranking,
    ranking_record,
    served_configuration,
    v2_plan,
)
from exportspredictor.loop.trainers import (
    ANY_FLOW_PARAMS,
    CATBOOST,
    CATBOOST_DEFAULT_PARAMS,
    LIGHTGBM,
    LIGHTGBM_DEFAULT_PARAMS,
    TRAINERS,
    TWEEDIE_ONLY,
    XGBOOST,
    CatBoostTrainer,
    LightGBMTrainer,
    Model,
    Trainer,
    TrainingJob,
    TrainingRows,
    XGBoostTrainer,
    trainer_for,
)

__all__ = [
    "ANY_FLOW",
    "ANY_FLOW_PARAMS",
    "BLEND",
    "BLEND_GRID",
    "BLEND_WEIGHTS",
    "CATBOOST",
    "CATBOOST_DEFAULT_PARAMS",
    "CONFIRMATION",
    "CONFIRMED_NAIVE_WEIGHT",
    "CONSTRAINTS_EXPERIMENT",
    "DIAGNOSTICS",
    "EXCLUDED_PRODUCTS",
    "EXCLUDED_PRODUCTS_BASE_SET",
    "EXCLUDED_PRODUCTS_EXPERIMENT",
    "EXCLUDED_PRODUCTS_SET",
    "EXPORTER_COUNT_EXPERIMENT",
    "EXTERNAL_MEMORY",
    "EXTERNAL_MEMORY_EXPERIMENT",
    "FIXTURE_LOOP",
    "HYPERPARAMETER_STAGES",
    "KEYED_FIELDS",
    "LIGHTGBM",
    "LIGHTGBM_DEFAULT_PARAMS",
    "LOWER_ZERO_RATE",
    "MARKET_REACH_CAP",
    "MARKET_REACH_CAP_EXPERIMENT",
    "MARKET_REACH_CAP_GRID",
    "MAX_ROUNDS",
    "MAX_TRAINED_MODELS",
    "MIN_WIN_THRESHOLD",
    "MODELS_DIR",
    "MODEL_FAMILY_ARMS",
    "NAIVE_EXPERIMENT",
    "NOISE_FLOOR",
    "NOISE_SEEDS",
    "NO_MEASURE",
    "PATIENCE",
    "REBUILT_BASE",
    "RECHECK",
    "RECORDED",
    "RESULTS_FILE",
    "ROUNDING",
    "SCOPE_EXPERIMENTS",
    "SCORING_FIELDS",
    "SELECTION_MEASURE",
    "SELECTION_MEASURE_TEXT",
    "SELECT_EXPERIMENT",
    "SERVED_BASE",
    "SHALLOW_DEPTH",
    "SIBLING_DEMAND_BASE_SET",
    "SIBLING_DEMAND_EXPERIMENT",
    "SIBLING_DEMAND_SET",
    "SIZE",
    "STARTING_POINT",
    "STOPPING_RULE_ROUNDS",
    "TOP_MARKETS",
    "TRAINERS",
    "TRAINING_FIELDS",
    "TWEEDIE_ONLY",
    "TWO_PART",
    "TWO_PART_EXPERIMENT",
    "UK_ZEROS_EXPERIMENT",
    "UK_ZEROS_SAMPLED",
    "V2",
    "WIDER_SCOPE",
    "WIDER_SCOPE_BASE_SET",
    "WIDER_SCOPE_EXPERIMENT",
    "WIDER_SCOPE_NON_UK_RATE",
    "WIDER_SCOPE_SET",
    "WINNER",
    "XGBOOST",
    "ZERO_RATE_EXPERIMENT",
    "Arm",
    "CatBoostTrainer",
    "LightGBMTrainer",
    "Loop",
    "LoopData",
    "LoopPlan",
    "Model",
    "ServedConfiguration",
    "Settings",
    "TestYearRead",
    "Trainer",
    "TrainingJob",
    "TrainingRows",
    "TwoPart",
    "XGBoostTrainer",
    "blend_arm",
    "chosen_form",
    "confirm",
    "confirm_best",
    "constraint_arms",
    "count_experiments",
    "current_win_threshold",
    "decide",
    "decided_by_stopping_rule",
    "excluded_products_arms",
    "experiment_label",
    "exporter_count_arms",
    "external_memory_arm",
    "fixture_data",
    "fixture_ranking",
    "hyperparameter_experiment",
    "load",
    "log_progress",
    "loop_dir",
    "main",
    "market_reach_cap_arm",
    "measures",
    "model_family_experiment",
    "models_trained_by",
    "naive_from_features",
    "naive_margin",
    "product_detail_arm",
    "product_detail_experiment",
    "rank_average_measures",
    "ranking_record",
    "read_results",
    "real_data",
    "recheck_results",
    "run_excluded_products",
    "run_exporter_count",
    "run_external_memory",
    "run_hyperparameter_stage",
    "run_market_reach_cap",
    "run_model_family",
    "run_product_detail",
    "run_sibling_demand",
    "run_wider_scope",
    "save_recheck",
    "save_results",
    "select_arms",
    "served_base_arm",
    "served_configuration",
    "sibling_demand_arm",
    "starting_point_arm",
    "time_one_arm",
    "trainer_for",
    "training_key",
    "two_part_arm",
    "v2_plan",
    "wider_scope_arms",
    "win_threshold",
    "world_products",
]
