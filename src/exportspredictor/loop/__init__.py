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
  over those seeds, and at least `MIN_WIN_THRESHOLD`.
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

The results are associations measured on one fold and one year: an arm that wins ranks
the held-out UK Markets better in 2022, which says nothing about why.

The package, one module per concern; everything public is importable from `exportspredictor.loop`:

- `selection`: the selection measure, its diagnostics and the win rule.
- `arms`: `Settings`, `Arm`, `LoopPlan`, the cache key and the seven experiments' arms.
- `data`: the fixture and real data, and the results file.
- `runner`: `Loop`, which runs and caches arms on one fold.
- `confirmation`: the five-fold confirmation.
- `served`: what the pipeline serves of the loop's choice.
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

**Cache and results** live in `loop_dir()` (`data/derived/loop_V202601/`), shared by every
worktree through `scripts/wt.ps1`. Models are `models/<training_key>.ubj` with a `.json` of
their rounds; `results.json` holds the naive baseline, the noise floor, every experiment's
rows and decision, `best` and `confirmation`. `load()` reads it. A re-run skips arms by
name, so to re-run an arm, give it a new name; the file is written only by the loop.

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

from exportspredictor.experiments import TestYearRead
from exportspredictor.loop.arms import (
    ANY_FLOW,
    ANY_FLOW_PARAMS,
    BLEND,
    BLEND_GRID,
    BLEND_WEIGHTS,
    CONSTRAINTS_EXPERIMENT,
    FIXTURE_LOOP,
    KEYED_FIELDS,
    LOWER_ZERO_RATE,
    MAX_ROUNDS,
    MAX_TRAINED_MODELS,
    NAIVE_EXPERIMENT,
    NOISE_FLOOR,
    NOISE_SEEDS,
    PATIENCE,
    SCOPE_EXPERIMENTS,
    SCORING_FIELDS,
    SELECT_EXPERIMENT,
    SHALLOW_DEPTH,
    SIZE,
    STARTING_POINT,
    TRAINING_FIELDS,
    TWEEDIE_ONLY,
    TWO_PART,
    TWO_PART_EXPERIMENT,
    UK_ZEROS_EXPERIMENT,
    UK_ZEROS_SAMPLED,
    ZERO_RATE_EXPERIMENT,
    Arm,
    LoopPlan,
    Settings,
    blend_arm,
    constraint_arms,
    count_experiments,
    experiment_label,
    models_trained_by,
    product_detail_arm,
    product_detail_experiment,
    select_arms,
    starting_point_arm,
    training_key,
    two_part_arm,
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
    read_results,
    real_data,
    save_results,
    world_products,
)
from exportspredictor.loop.runner import (
    NO_MEASURE,
    Loop,
    TwoPart,
    log_progress,
    naive_margin,
    run_product_detail,
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

__all__ = [
    "ANY_FLOW",
    "ANY_FLOW_PARAMS",
    "BLEND",
    "BLEND_GRID",
    "BLEND_WEIGHTS",
    "CONFIRMATION",
    "CONFIRMED_NAIVE_WEIGHT",
    "CONSTRAINTS_EXPERIMENT",
    "DIAGNOSTICS",
    "FIXTURE_LOOP",
    "KEYED_FIELDS",
    "LOWER_ZERO_RATE",
    "MAX_ROUNDS",
    "MAX_TRAINED_MODELS",
    "MIN_WIN_THRESHOLD",
    "MODELS_DIR",
    "NAIVE_EXPERIMENT",
    "NOISE_FLOOR",
    "NOISE_SEEDS",
    "NO_MEASURE",
    "PATIENCE",
    "RECORDED",
    "RESULTS_FILE",
    "ROUNDING",
    "SCOPE_EXPERIMENTS",
    "SCORING_FIELDS",
    "SELECTION_MEASURE",
    "SELECTION_MEASURE_TEXT",
    "SELECT_EXPERIMENT",
    "SHALLOW_DEPTH",
    "SIZE",
    "STARTING_POINT",
    "TOP_MARKETS",
    "TRAINING_FIELDS",
    "TWEEDIE_ONLY",
    "TWO_PART",
    "TWO_PART_EXPERIMENT",
    "UK_ZEROS_EXPERIMENT",
    "UK_ZEROS_SAMPLED",
    "V2",
    "WINNER",
    "ZERO_RATE_EXPERIMENT",
    "Arm",
    "Loop",
    "LoopData",
    "LoopPlan",
    "ServedConfiguration",
    "Settings",
    "TestYearRead",
    "TwoPart",
    "blend_arm",
    "chosen_form",
    "confirm",
    "confirm_best",
    "constraint_arms",
    "count_experiments",
    "decide",
    "experiment_label",
    "fixture_data",
    "fixture_ranking",
    "load",
    "log_progress",
    "loop_dir",
    "main",
    "measures",
    "models_trained_by",
    "naive_from_features",
    "naive_margin",
    "product_detail_arm",
    "product_detail_experiment",
    "rank_average_measures",
    "ranking_record",
    "read_results",
    "real_data",
    "run_product_detail",
    "save_results",
    "select_arms",
    "served_configuration",
    "starting_point_arm",
    "time_one_arm",
    "training_key",
    "two_part_arm",
    "v2_plan",
    "win_threshold",
    "world_products",
]
