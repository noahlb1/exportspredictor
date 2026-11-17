"""Accuracy of Export Potential on the test year, the Baseline's stability, and the known limitations.

The time split is train ≤ 2021, validate 2022, test 2023 (Baseline year 2024). Tuning
reads only the training and validation years (`pipeline`). The **test-year models** here
are five fold models refitted on every year before the test year with the tuned
hyperparameters, so the test year is never seen by tuning or training.

Each metric is computed for the model and for a **naive baseline**, the Exporter's share
of the Product's world exports times the Importer's imports of that Product (both with
the row's own Flow left out, as the model's supply and demand features are):

- **Tweedie deviance** (variance power 1.5, weighted, so sampled zero cells count by
  their weight): the training loss, on every row. Lower is better.
- **RMSE of log(1 + v)** on non-zero Flows: the typical error in orders of magnitude.
  Lower is better.
- **Spearman rank correlation within each Product**, averaged over Products: how well
  the ranking of Flows within a Product matches reality. Higher is better. This is how
  Advisors use the model (Markets ranked for one Product), so it is the headline.

Two scopes are evaluated:

- `all_exporters`: every test-year training row (every non-zero Flow plus the sampled
  zero cells). UK rows are scored out-of-fold; other rows by the mean of the fold models.
- `uk_out_of_fold`: the UK's full Product × Market grid of the test year, each pair scored
  by the fold model that held it out (ADR 0001).

Each is broken down by Sector (HS2 chapter) and by the Importer's region, weakest first.

The **stability** check compares the Baseline's rankings with the same models' rankings of
the test year's UK grid: how much the rankings move from one year of data to the next.

The result is a JSON-ready dict stored in the artefact as `evaluation.json`; Lift and the
Event Backtest (#21) add their own keys to it. `uv run python -m exportspredictor.evaluation`
adds an evaluation to the newest real artefact without retraining its Baseline models.
"""

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor import artefacts as store
from exportspredictor import baci
from exportspredictor.features import FEATURES, FeatureTransform, features_dir
from exportspredictor.folds import N_FOLDS, score_out_of_fold, train_fold_models
from exportspredictor.gap_period import baseline_year_scores
from exportspredictor.paths import artefacts_dir, data_dir
from exportspredictor.training import Chunk, scan_chunks

TWEEDIE_POWER = 1.5
# Predictions are floored at $1 before the deviance, so that a zero naive prediction of a
# non-zero Flow is penalised heavily but finitely.
MIN_PREDICTION_USD = 1.0
# A Product (or Product within a group) needs this many rows for its rank correlation.
MIN_RANKED_ROWS = 3
STABILITY_TOP_N = 1000
TEST_MODELS_FILE = "test_models.json"
METRICS = ["tweedie_deviance", "rmse_log", "spearman_within_product"]
METRIC_INFO = {
    "tweedie_deviance": {"label": "Tweedie deviance", "better": "lower"},
    "rmse_log": {"label": "RMSE of log(1 + value), non-zero Flows", "better": "lower"},
    "spearman_within_product": {"label": "Spearman rank correlation within each Product", "better": "higher"},
}
HEADLINE_METRIC = "spearman_within_product"
HEADLINE_SCOPE = "uk_out_of_fold"
# A headline win smaller than this is reported as "only slightly better".
NARROW_MARGIN = 0.05
_PREDICTORS = {"model": "export_potential_usd", "naive": "naive_usd"}

LIMITATIONS = [
    {
        "title": "Goods only",
        "text": "The model covers goods trade only. Services, about half of UK exports, are not in the data.",
    },
    {
        "title": "No tariffs",
        "text": "Tariffs are not a model input in this version. The only Barrier the model sees is the "
        "absence of a Trade Agreement or Trading Bloc membership, so it cannot say how a tariff change would "
        "shift Export Potential.",
    },
    {
        "title": "Associations, not causes",
        "text": "Export Potential and Scenario Effects are associations the model has learned from global trade. "
        "\"The model associates a Trade Agreement with +X%\" does not mean the agreement causes +X%, and an "
        "Opportunity is a tendency, not a forecast.",
    },
    {
        "title": "Provisional 2024 data",
        "text": "The Baseline year, 2024, is provisional in BACI: some countries had not reported when it was "
        "built, and it will be revised in the next release. 2024 has about 5% fewer Flows than 2023.",
    },
    {
        "title": "The 2012 product edition",
        "text": "Products use HS 2012 codes (BACI's HS12 revision, chosen for its longer history). A few HS6 codes "
        "Advisors know from current tariff schedules (HS 2017 or 2022) differ or are split.",
    },
    {
        "title": "Reporting quality differs between countries",
        "text": "BACI reconciles what exporters and importers declare, and countries report with different care "
        "and delay. A missing Flow is treated as zero, so the model may be weaker for low-income and poorly "
        "reporting Markets; the region breakdown shows where.",
    },
    {
        "title": "Sanctions and other non-tariff restrictions are invisible",
        "text": "Sanctions, embargoes, quotas, licensing and product standards are not model inputs. A Market under "
        "sanctions can still show a large Export Gap: Russia ranks high for UK whisky because the model sees "
        "its fundamentals, not the sanctions in force since 2022. Check such Markets before acting on them.",
    },
]


# --- Metrics ---------------------------------------------------------------------------


def tweedie_deviance(prediction: str, *, power: float = TWEEDIE_POWER) -> pl.Expr:
    """Weighted mean Tweedie unit deviance of `prediction` against `value_usd` (1 < power < 2)."""
    y = pl.col("value_usd").cast(pl.Float64)
    mu = pl.col(prediction).cast(pl.Float64).clip(lower_bound=MIN_PREDICTION_USD)
    unit = 2 * (
        y.pow(2 - power) / ((1 - power) * (2 - power))
        - y * mu.pow(1 - power) / (1 - power)
        + mu.pow(2 - power) / (2 - power)
    )
    weight = pl.col("weight").cast(pl.Float64)
    return (unit * weight).sum() / weight.sum()


def _rmse_log(prediction: str) -> pl.Expr:
    error = pl.col(prediction).cast(pl.Float64).log1p() - pl.col("value_usd").cast(pl.Float64).log1p()
    return error.filter(pl.col("value_usd") > 0).pow(2).mean().sqrt()


def naive_potential(rows: pl.DataFrame, flows: pl.DataFrame) -> pl.DataFrame:
    """Add `naive_usd`: the Exporter's share of the Product's world exports times the Importer's imports of it.

    `flows` holds every non-zero Flow of the same year (exporter, importer, hs6, value_usd).
    Each row's own Flow is left out of all three totals, as the model's features leave it out.
    """
    value = pl.col("value_usd")
    active = flows.filter(value > 0).select(
        pl.col("exporter", "importer", "hs6").cast(pl.String), value.cast(pl.Float64)
    )
    supply = active.group_by("exporter", "hs6").agg(_supply=value.sum())
    demand = active.group_by("importer", "hs6").agg(_demand=value.sum())
    world = active.group_by("hs6").agg(_world=value.sum())
    keys = rows.select(
        _e=pl.col("exporter").cast(pl.String),
        _i=pl.col("importer").cast(pl.String),
        _p=pl.col("hs6").cast(pl.String),
        _own=pl.col("value_usd").cast(pl.Float64),
    )
    joined = (
        keys.join(supply.rename({"exporter": "_e", "hs6": "_p"}), on=["_e", "_p"], how="left", maintain_order="left")
        .join(demand.rename({"importer": "_i", "hs6": "_p"}), on=["_i", "_p"], how="left", maintain_order="left")
        .join(world.rename({"hs6": "_p"}), on="_p", how="left", maintain_order="left")
        .fill_null(0.0)
    )
    own = pl.col("_own")
    other_world = pl.col("_world") - own
    naive = joined.select(
        naive_usd=pl.when(other_world > 0)
        .then((pl.col("_supply") - own).clip(lower_bound=0) * (pl.col("_demand") - own).clip(lower_bound=0) / other_world)
        .otherwise(0.0)
    )
    return rows.with_columns(naive["naive_usd"])


def accuracy(scored: pl.DataFrame, markets: pl.DataFrame) -> dict:
    """Every metric for the model and the naive baseline: overall, by Sector, by region and by income level.

    `scored` has exporter, importer, hs6, value_usd, weight, `export_potential_usd` and
    `naive_usd`; `markets` maps `iso3` to `region` and, optionally, World Bank
    `income_level` (the bias review's stand-in for reporting quality; `by_income` is
    added only when it is there). Breakdowns are sorted weakest first (lowest
    within-Product rank correlation of the model).
    """
    has_income = "income_level" in markets.columns
    regions = markets.select(
        *(["income_level"] if has_income else []), _i=pl.col("iso3").cast(pl.String), region=pl.col("region")
    )
    frame = (
        scored.with_columns(
            hs6=pl.col("hs6").cast(pl.String),
            _i=pl.col("importer").cast(pl.String),
        )
        .with_columns(sector=pl.col("hs6").str.slice(0, 2))
        .join(regions, on="_i", how="left")
        .with_columns(pl.col("region", *(["income_level"] if has_income else [])).fill_null("Unclassified"))
    )
    overall = _metrics(frame.with_columns(_all=pl.lit("all")), "_all").drop("group").row(0, named=True)
    groups = {"by_sector": "sector", "by_region": "region", **({"by_income": "income_level"} if has_income else {})}
    return {
        "overall": _clean(overall),
        **{
            name: [_clean(row) for row in _weakest_first(_metrics(frame, column)).iter_rows(named=True)]
            for name, column in groups.items()
        },
    }


def _metrics(frame: pl.DataFrame, group: str) -> pl.DataFrame:
    sums = frame.group_by(group).agg(
        rows=pl.len(),
        non_zero_flows=(pl.col("value_usd") > 0).sum(),
        products=pl.col("hs6").n_unique(),
        actual_usd=pl.col("value_usd").cast(pl.Float64).dot(pl.col("weight")),
        **{f"{name}_tweedie_deviance": tweedie_deviance(col) for name, col in _PREDICTORS.items()},
        **{f"{name}_rmse_log": _rmse_log(col) for name, col in _PREDICTORS.items()},
    )
    within = (
        frame.group_by(group, "hs6")
        .agg(
            _n=pl.len(),
            **{name: pl.corr(col, "value_usd", method="spearman") for name, col in _PREDICTORS.items()},
        )
        .filter(pl.col("_n") >= MIN_RANKED_ROWS)
        .group_by(group)
        .agg(
            ranked_products=pl.len(),
            **{
                f"{name}_spearman_within_product": pl.col(name).filter(pl.col(name).is_not_nan()).mean()
                for name in _PREDICTORS
            },
        )
    )
    return (
        sums.join(within, on=group, how="left")
        .with_columns(pl.col("ranked_products").fill_null(0))
        .rename({group: "group"})
        .with_columns(pl.col("group").cast(pl.String))
    )


def _weakest_first(table: pl.DataFrame) -> pl.DataFrame:
    return table.sort("model_spearman_within_product", "group", nulls_last=True)


def _clean(row: dict) -> dict:
    """JSON-safe: NaN becomes None, numpy scalars become Python numbers."""
    out = {}
    for key, value in row.items():
        if isinstance(value, float) and math.isnan(value):
            value = None
        out[key] = value
    return out


def headline(overall: dict, test_year: int) -> dict:
    """The plain-English accuracy headline: the model's within-Product ranking against the naive baseline."""
    model = overall[f"model_{HEADLINE_METRIC}"]
    naive = overall[f"naive_{HEADLINE_METRIC}"]
    beats = model is not None and naive is not None and model > naive
    scale = "(1 would be a perfect ranking, 0 no better than chance)"
    if beats:
        degree = "only slightly better" if model - naive < NARROW_MARGIN else "better"
        verdict = (
            f"That is {degree} than the naive baseline (the UK's world market share times each Market's imports), "
            f"which scores {naive:.2f}."
        )
    else:
        verdict = (
            f"The model does not beat the naive baseline on this measure: the UK's world market share times each "
            f"Market's imports scores {naive:.2f}. Treat its rankings with extra caution."
        )
    text = (
        f"For a typical Product, the model's ranking of Markets for UK exports in {test_year}, a year it never "
        f"saw, has a rank correlation of {model:.2f} with what the UK actually exported {scale}. {verdict}"
    )
    return {
        "scope": HEADLINE_SCOPE,
        "metric": HEADLINE_METRIC,
        "model": model,
        "naive": naive,
        "beats_naive": beats,
        "text": text,
    }


def stability(current: pl.DataFrame, previous: pl.DataFrame, *, top_n: int = STABILITY_TOP_N) -> dict:
    """How much UK rankings move between two years scored by the same models.

    Both tables hold hs6, iso3, export_potential_usd and export_gap_usd. Reports the mean
    within-Product Spearman correlation of Export Potential and of Export Gap between the
    years, and the share of the top `top_n` Opportunities (by Export Gap) shared by both.
    """
    columns = ["hs6", "iso3", "export_potential_usd", "export_gap_usd"]
    joined = current.select(columns).join(previous.select(columns), on=["hs6", "iso3"], suffix="_prev")
    within = (
        joined.group_by("hs6")
        .agg(
            _n=pl.len(),
            potential=pl.corr("export_potential_usd", "export_potential_usd_prev", method="spearman"),
            gap=pl.corr("export_gap_usd", "export_gap_usd_prev", method="spearman"),
        )
        .filter(pl.col("_n") >= MIN_RANKED_ROWS)
    )
    n = max(1, min(top_n, joined.height))
    top_now = set(joined.top_k(n, by="export_gap_usd").select("hs6", "iso3").iter_rows())
    top_before = set(joined.top_k(n, by="export_gap_usd_prev").select("hs6", "iso3").iter_rows())
    return {
        "pairs": joined.height,
        "spearman_potential": _mean_finite(within["potential"]),
        "spearman_gap": _mean_finite(within["gap"]),
        "top_n": n,
        "top_overlap": len(top_now & top_before) / n,
    }


def _mean_finite(values: pl.Series) -> float | None:
    finite = values.drop_nulls().filter(values.drop_nulls().is_not_nan())
    return float(finite.mean()) if finite.len() else None


# --- Test-year models and scoring -------------------------------------------------------


@dataclass(frozen=True)
class TestModels:
    """Fold models refitted on every year before the test year, and their feature transform."""

    test_year: int
    training_years: list[int]
    models: list[xgb.Booster]
    transform: FeatureTransform

    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for k, model in enumerate(self.models):
            model.save_model(folder / f"fold_{k}.ubj")
        info = {
            "test_year": self.test_year,
            "training_years": self.training_years,
            "n_folds": len(self.models),
            "transform": self.transform.to_dict(),
        }
        # Written last: its presence marks the folder as complete.
        (folder / TEST_MODELS_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> "TestModels":
        folder = Path(folder)
        info = json.loads((folder / TEST_MODELS_FILE).read_text(encoding="utf-8"))
        models = []
        for k in range(info["n_folds"]):
            model = xgb.Booster()
            model.load_model(folder / f"fold_{k}.ubj")
            models.append(model)
        return cls(info["test_year"], info["training_years"], models, FeatureTransform.from_dict(info["transform"]))


def train_test_models(
    training: Mapping[int, Chunk],
    *,
    test_year: int,
    uk: str,
    params: dict,
    rounds: int,
    n_folds: int = N_FOLDS,
    device: str = "cuda",
    non_uk_rate: float = 1.0,
    seed: int = 0,
) -> TestModels:
    """Fit the fold models on every year before `test_year`; the test year itself is never read."""
    years = [y for y in sorted(training) if y < test_year]
    if not years:
        raise ValueError(f"No training years before the test year {test_year}")
    chunks = [training[y] for y in years]
    transform = FeatureTransform.fit(scan_chunks(chunks), years)
    models = train_fold_models(
        chunks, uk=uk, params=params, rounds=rounds, n_folds=n_folds, device=device, seed=seed,
        transform=transform, non_uk_rate=non_uk_rate,
    )
    return TestModels(test_year=test_year, training_years=years, models=models, transform=transform)


def score_rows(table: pl.DataFrame, models: Sequence[xgb.Booster], transform: FeatureTransform, uk: str) -> pl.DataFrame:
    """Add `export_potential_usd`: UK rows out-of-fold, other rows by the mean of the fold models."""
    features = transform.apply(table)
    is_uk = pl.col("exporter").cast(pl.String) == uk
    uk_rows = score_out_of_fold(features, list(models), uk=uk).drop("fold")
    others = features.filter(~is_uk)
    if others.height:
        data = others.select(FEATURES).to_numpy().astype(np.float32)
        mean = np.mean([model.inplace_predict(data) for model in models], axis=0)
        others = others.with_columns(export_potential_usd=pl.Series(mean, dtype=pl.Float64))
    else:
        others = others.with_columns(export_potential_usd=pl.lit(None, pl.Float64))
    return pl.concat([uk_rows.select(others.columns), others])


def _uk_table(scored_grid: pl.DataFrame) -> pl.DataFrame:
    return scored_grid.select(
        pl.col("hs6").cast(pl.String),
        iso3=pl.col("importer").cast(pl.String),
        export_potential_usd="export_potential_usd",
        export_gap_usd=pl.col("export_potential_usd") - pl.col("value_usd"),
    )


def evaluate(
    test_rows: pl.DataFrame,
    test_grid: pl.DataFrame,
    *,
    test_models: TestModels,
    baseline: pl.DataFrame,
    baseline_models: Sequence[xgb.Booster],
    baseline_transform: FeatureTransform,
    markets: pl.DataFrame,
    uk: str,
    settings: dict,
) -> dict:
    """The evaluation report for the test year.

    `test_rows` are the test year's training rows (every non-zero Flow plus sampled zeros),
    `test_grid` its full UK grid, both with features untransformed. `baseline` holds the
    Baseline year's UK scores (`gap_period.baseline_year_scores`) and `baseline_models` /
    `baseline_transform` the models that scored them, used for the stability check. `settings` (validation year, params, sampling rates, ...)
    is recorded in the report.
    """
    year = test_models.test_year
    if set(test_rows["year"].unique().to_list()) != {year} or set(test_grid["year"].unique().to_list()) != {year}:
        raise ValueError(f"Test rows and grid must hold the test year {year} only")
    if year in test_models.training_years:
        raise ValueError("The test-year models were trained on the test year")

    flows = test_rows.filter(pl.col("value_usd") > 0).select("exporter", "importer", "hs6", "value_usd")
    scored_rows = naive_potential(score_rows(test_rows, test_models.models, test_models.transform, uk), flows)
    scored_grid = naive_potential(score_rows(test_grid, test_models.models, test_models.transform, uk), flows)
    accuracy_by_scope = {
        "all_exporters": accuracy(scored_rows, markets),
        "uk_out_of_fold": accuracy(scored_grid, markets),
    }

    previous = _uk_table(score_out_of_fold(baseline_transform.apply(test_grid), list(baseline_models), uk=uk))
    baseline_year = year + 1
    stable = stability(baseline, previous)
    stable["years"] = [year, baseline_year]
    stable["text"] = _stability_text(stable, year, baseline_year)

    return {
        "test_year": year,
        "test_training_years": test_models.training_years,
        **settings,
        "tweedie_power": TWEEDIE_POWER,
        "min_prediction_usd": MIN_PREDICTION_USD,
        "metrics": {name: METRIC_INFO[name] for name in METRICS},
        "naive_baseline": "The Exporter's share of the Product's world exports times the Importer's imports "
        "of that Product, each leaving the Flow itself out.",
        "accuracy": accuracy_by_scope,
        "headline": headline(accuracy_by_scope[HEADLINE_SCOPE]["overall"], year),
        "stability": stable,
    }


def _stability_text(result: dict, year: int, baseline_year: int) -> str:
    potential = result["spearman_potential"]
    agreement = "n/a" if potential is None else f"{potential:.2f}"
    return (
        f"Scored by the same models, each Product's ranking of Markets by Export Potential in {baseline_year} "
        f"has a rank correlation of {agreement} with its {year} ranking, and {result['top_overlap']:.0%} of the "
        f"top {result['top_n']:,} Opportunities by Export Gap are the same in both years."
    )


# --- Real data ----------------------------------------------------------------------------

# Metadata keys of the trained artefacts that the evaluation report records.
SETTINGS_KEYS = (
    "validation_year",
    "tuning_years",
    "params",
    "rounds",
    "n_folds",
    "non_uk_sampling_rate",
    "zero_sampling_rate",
)


def report_settings(metadata: dict) -> dict:
    """The training settings an evaluation report records, from the artefact metadata."""
    return {key: metadata[key] for key in SETTINGS_KEYS}


def test_models_dir(root: Path, metadata: dict) -> Path:
    """Where real test-year models are cached under the data root, keyed by their training settings."""
    from exportspredictor.features import features_dir

    settings = {**report_settings(metadata), "model_params": metadata["model_params"], "test_year": metadata["baseline_year"] - 1}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return features_dir(root).parent / "test_models" / f"{settings['test_year']}_{key}"


def real_test_models(artefacts, root: Path | None = None, *, device: str = "cuda") -> TestModels:
    """The real test-year fold models for an artefact: from the cache, or trained (about 20 min on GPU).

    They use the artefact's tuned parameters, round count, sampling rate and seed, so they
    differ from its Baseline models only in leaving out the test year.
    """
    root = Path(root or data_dir())
    metadata = artefacts.metadata
    test_year = metadata["baseline_year"] - 1
    cache = test_models_dir(root, metadata)
    if (cache / TEST_MODELS_FILE).exists():
        return TestModels.load(cache)
    folder = features_dir(root)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    training = {y: folder / f"training_{y}.parquet" for y in settings["years"] if y < test_year}
    models = train_test_models(
        training,
        test_year=test_year,
        uk=baci.UK,
        params=metadata["params"],
        rounds=metadata["rounds"],
        n_folds=metadata["n_folds"],
        device=device,
        non_uk_rate=metadata["non_uk_sampling_rate"],
        seed=metadata["model_params"].get("seed", 0),
    )
    models.save(cache)
    return models


def evaluate_real(artefacts, test_models: TestModels, root: Path | None = None) -> dict:
    """The evaluation report of a real artefact, from the per-year feature files."""
    folder = features_dir(root)
    year = test_models.test_year
    return evaluate(
        pl.read_parquet(folder / f"training_{year}.parquet"),
        pl.read_parquet(folder / f"uk_grid_{year}.parquet"),
        test_models=test_models,
        baseline=baseline_year_scores(artefacts.history, artefacts.baseline_year),
        baseline_models=artefacts.fold_models,
        baseline_transform=FeatureTransform.from_dict(artefacts.metadata["feature_transform"]),
        markets=markets_with_income(artefacts.markets, root),
        uk=baci.UK,
        settings=report_settings(artefacts.metadata),
    )


def markets_with_income(markets: pl.DataFrame, root: Path | None = None) -> pl.DataFrame:
    """The artefact's Market lookup plus World Bank `income_level` from the BACI lookups (keyed by standard ISO3)."""
    if "income_level" in markets.columns:
        return markets
    country_map = pl.read_parquet((root or data_dir()) / "derived" / "country_map.parquet")
    income = (
        baci.markets(root)
        .select(baci_iso3="iso3", income_level="income_level")
        .join(country_map.select("baci_iso3", "iso3").unique("baci_iso3"), on="baci_iso3", how="left")
        .select(iso3=pl.col("iso3").fill_null(pl.col("baci_iso3")), income_level="income_level")
        .unique("iso3")
    )
    return markets.join(income, on="iso3", how="left")


def add_to_latest(root: Path | None = None, artefacts_root: Path | None = None, *, device: str = "cuda") -> Path:
    """Evaluate the newest artefact and save it again, with the report, as a new version.

    The Baseline models and tables are copied unchanged; only the test-year models are
    trained (or read from the cache). Returns the new folder.
    """
    artefacts_root = Path(artefacts_root or artefacts_dir())
    source = store.latest_folder(artefacts_root)
    bundle = store.load(source)
    started = time.perf_counter()
    test_models = real_test_models(bundle, root, device=device)
    report = evaluate_real(bundle, test_models, root)
    report["evaluation_seconds"] = round(time.perf_counter() - started, 2)
    metadata = {**bundle.metadata, "version": store.new_version(), "copied_from": bundle.version}
    return store.copy_version(source, artefacts_root, metadata=metadata, evaluation=report)


if __name__ == "__main__":
    print(add_to_latest(), flush=True)
