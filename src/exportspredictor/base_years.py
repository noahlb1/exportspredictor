"""The base-year fold models and the scored UK panel that ITC's Realisation Test runs on (spec #156).

For each base year t in `BASE_YEARS`, the frozen Configuration's fold models (ADR 0001) are trained again on the
years up to t only, with the same hyperparameters, round count, non-UK sampling and features, through the
test-year fold-model trainer (`evaluation.train_test_models`). Each base year's UK grid is then scored out-of-fold in
the served Model Form: the Blend with the Naive Baseline, then Market Reach learned from the UK grid of t − 1
(ADR 0008, ADR 0010), through `prediction.export_potential`. So no base-year score has seen a later year: the
artefact's own fold models were trained on 2012–2023 and could not stand in for them.

The **panel** (`build_panel`) holds every UK Product–Market pair of the base years' grids (Excluded Products left
out, as in the served version) in every year of `PANEL_YEARS`, 2012–2022, with:

- `actual_exports`: the UK's Actual Exports, 0 where there is no Flow;
- `export_potential` and `naive_baseline`: filled in base-year rows only (null elsewhere, and for a pair not in
  that year's UK grid).

Product and Market keep the names of the codebase's UK tables, `hs6` and `iso3` (`PANEL_SCHEMA`).

**2023 is never read**: every year the panel needs (the panel years, the base years and Market Reach's reference
years) must be before the test year, or `build_panel` raises `LeakageError` before reading anything. Nothing goes
through the Test-year Ledger (ADR 0009).

Models are cached under the data root (`models_dir`), keyed by their settings, so a re-run costs minutes. The real
run, about 25 models on the GPU, is `python -m exportspredictor.base_years`; it reads the Configuration from the
newest artefact, writes the panel (`save_panel`) and prints a record of each base year. `load_panel` reads it back.
"""

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from exportspredictor import baci, downloads, market_reach
from exportspredictor.backtests import LeakageError
from exportspredictor.evaluation import (
    TEST_MODELS_FILE,
    TestModels,
    naive_potential,
    train_test_models,
)
from exportspredictor.exclusions import is_excluded
from exportspredictor.features import (
    FEATURES,
    build_features,
    feature_set,
    features_dir,
    load_sources,
    load_year_flows,
    uk_grid,
)
from exportspredictor.fixture import UK as FIXTURE_UK
from exportspredictor.fixture import World
from exportspredictor.folds import score_out_of_fold
from exportspredictor.paths import data_dir
from exportspredictor.prediction import ModelForm
from exportspredictor.training import Chunk, read_chunk

BASE_YEARS = (2015, 2016, 2017, 2018, 2019)
# Past growth needs t − 3 ≥ 2012 and the outcomes t + 1..t + 3 end in 2022.
PANEL_YEARS = tuple(range(2012, 2023))
TEST_YEAR = 2023  # never read (ADR 0009)

PANEL_SCHEMA = {
    "hs6": pl.String,
    "iso3": pl.String,
    "year": pl.Int16,
    "export_potential": pl.Float64,
    "naive_baseline": pl.Float64,
    "actual_exports": pl.Float64,
}
PANEL_FILE = "panel.parquet"
PANEL_INFO_FILE = "panel.json"
_TRAINING_FILE = "training.json"


@dataclass(frozen=True)
class Configuration:
    """What the base-year models share with the frozen version: its settings, read from the artefact metadata."""

    params: dict
    rounds: int
    n_folds: int
    non_uk_rate: float
    seed: int
    features: tuple[str, ...]
    feature_set: str  # the feature files the version was trained on ("" is the original build)
    model_form: dict = field(default_factory=dict)  # `ModelForm.to_dict()` of the served form

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "Configuration":
        return cls(
            params=dict(metadata["params"]),
            rounds=int(metadata["rounds"]),
            n_folds=int(metadata["n_folds"]),
            non_uk_rate=float(metadata["non_uk_sampling_rate"]),
            seed=int(metadata["model_params"].get("seed", 0)),
            features=tuple(metadata["features"]),
            feature_set=metadata.get("feature_set") or "",
            model_form=dict(metadata["model_form"]),
        )

    @property
    def form(self) -> ModelForm:
        """The served Model Form; its Market Reach is fitted again for each base year (`market_reach.fit`)."""
        return ModelForm.from_dict(self.model_form)

    def settings(self, training_years: Sequence[int]) -> dict:
        """The settings that decide a base-year model: its cache key."""
        return {
            "training_years": list(training_years), "params": self.params, "rounds": self.rounds,
            "n_folds": self.n_folds, "non_uk_rate": self.non_uk_rate, "seed": self.seed,
            "features": list(self.features), "feature_set": self.feature_set,
        }


@dataclass(frozen=True)
class Inputs:
    """Per-year feature tables (training rows and UK grids, Parquet paths or frames) and every Exporter's Flows of a
    year (for the Naive Baseline). Only the years the panel needs are ever read."""

    training: Mapping[int, Chunk]
    grids: Mapping[int, Chunk]
    year_flows: Callable[[int], pl.DataFrame]


def models_dir(cache: Path, *, base_year: int, configuration: Configuration, training_years: Sequence[int]) -> Path:
    """Where the base-year models of these settings are cached under `cache`: `<base year>_<hash of the settings>`."""
    settings = configuration.settings(training_years)
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return Path(cache) / f"{base_year}_{key}"


def train_base_year_models(
    training: Mapping[int, Chunk],
    *,
    base_year: int,
    configuration: Configuration,
    uk: str = baci.UK,
    device: str = "cuda",
    cache: Path | None = None,
) -> TestModels:
    """The Configuration's fold models trained on every year of `training`, all of which must be ≤ `base_year`.

    Raises `LeakageError` if `training` holds any later year, as the backtest trainer does. With `cache`, the models
    are read from `models_dir` when saved there (no chunk is read), or saved there after training. They are
    `TestModels` whose `test_year` is base year + 1, the first year they did not see.
    """
    years = sorted(training)
    if not years:
        raise ValueError(f"No training years for the base year {base_year}")
    if max(years) > base_year:
        raise LeakageError(f"Base-year models for {base_year} would train on {years}, not only on years up to it")
    folder = None
    if cache is not None:
        folder = models_dir(cache, base_year=base_year, configuration=configuration, training_years=years)
        if (folder / TEST_MODELS_FILE).exists():
            return TestModels.load(folder)
    started = time.perf_counter()
    models = train_test_models(
        {year: training[year] for year in years}, test_year=base_year + 1, uk=uk, params=configuration.params,
        rounds=configuration.rounds, n_folds=configuration.n_folds, device=device,
        non_uk_rate=configuration.non_uk_rate, seed=configuration.seed, features=configuration.features,
    )
    if max(models.training_years) > base_year:
        raise LeakageError(f"Base-year models for {base_year} trained on {models.training_years}")
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        seconds = round(time.perf_counter() - started, 2)
        (folder / _TRAINING_FILE).write_text(json.dumps({"training_seconds": seconds}), encoding="utf-8")
        models.save(folder)
    return models


def score_base_year(
    inputs: Inputs, models: TestModels, configuration: Configuration, *, base_year: int, uk: str = baci.UK
) -> pl.DataFrame:
    """The UK grid of `base_year` scored out-of-fold in the served Model Form: hs6, iso3, export_potential_usd,
    naive_usd. Market Reach is learned from the UK grid of the year before (ADR 0010)."""
    grid = naive_potential(read_chunk(inputs.grids[base_year]), inputs.year_flows(base_year))
    transformed = models.transform.apply(grid)
    form = configuration.form
    if form.market_reach is not None:
        reference = base_year - market_reach.LAG_YEARS
        source = read_chunk(inputs.grids[reference]).select("year", "hs6", "importer", "value_usd", market_reach.DEMAND)
        form = market_reach.fit(
            form, models.models, transformed, source, scored_year=base_year, uk=uk, beta=form.market_reach.beta,
            features=models.features,
        )
    scored = score_out_of_fold(transformed, models.models, uk, inputs=models.features, form=form).select(
        pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String),
        export_potential_usd="export_potential_usd", naive_usd="naive_usd",
    )
    if scored.select("hs6", "iso3").is_duplicated().any() or scored.height != grid.height:
        raise ValueError("A scored grid must hold exactly one row per UK Product–Market pair")
    if scored.select(is_excluded().any()).item():
        raise ValueError("The UK grid holds Excluded Products; they are never scored (ADR 0005)")
    return scored


def build_panel(
    inputs: Inputs,
    configuration: Configuration,
    *,
    base_years: Sequence[int] = BASE_YEARS,
    panel_years: Sequence[int] = PANEL_YEARS,
    test_year: int = TEST_YEAR,
    uk: str = baci.UK,
    device: str = "cuda",
    cache: Path | None = None,
    records: list[dict] | None = None,
) -> pl.DataFrame:
    """The scored UK panel (module docstring): `PANEL_SCHEMA`, one row per pair and panel year.

    Raises `LeakageError`, before reading anything, if any year it needs is the test year or later. Each base year's
    models train on the years of `inputs.training` up to it (`train_base_year_models`, cached under `cache`).
    `records`, if given, gains one dict per base year: its cache folder, training years, seconds and rows.
    """
    if not set(base_years) <= set(panel_years):
        raise ValueError(f"The base years {list(base_years)} must be panel years")
    needed = {*panel_years, *(t - market_reach.LAG_YEARS for t in base_years)}
    if max(needed) >= test_year:
        raise LeakageError(f"The panel would read {sorted(needed)}, not only years before the test year {test_year}")
    scored = []
    for t in base_years:
        started = time.perf_counter()
        years = sorted(y for y in inputs.training if y <= t)
        folder = None if cache is None else models_dir(cache, base_year=t, configuration=configuration, training_years=years)
        cached = folder is not None and (folder / TEST_MODELS_FILE).exists()
        models = train_base_year_models(
            {y: inputs.training[y] for y in years}, base_year=t, configuration=configuration, uk=uk, device=device,
            cache=cache,
        )
        trained = time.perf_counter()
        table = score_base_year(inputs, models, configuration, base_year=t, uk=uk)
        done = time.perf_counter()
        scored.append(table.with_columns(year=pl.lit(t, pl.Int16)))
        record = {
            "base_year": t, "training_years": models.training_years, "cache": None if folder is None else folder.name,
            "cached": cached, "train_seconds": round(trained - started, 1), "score_seconds": round(done - trained, 1),
            "rows": table.height,
        }
        print(json.dumps(record), flush=True)
        if records is not None:
            records.append(record)
    scores = pl.concat(scored)
    pairs = scores.select("hs6", "iso3").unique()
    actual = pl.concat(
        [
            _grid_columns(inputs.grids[y]).filter(pl.col("value_usd") > 0).select(
                pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String), year=pl.lit(y, pl.Int16),
                actual_exports=pl.col("value_usd").cast(pl.Float64),
            )
            for y in panel_years
        ]
    )
    return (
        pairs.join(pl.DataFrame({"year": list(panel_years)}, schema={"year": pl.Int16}), how="cross")
        .join(actual, on=["hs6", "iso3", "year"], how="left")
        .join(
            scores.select("hs6", "iso3", "year", export_potential="export_potential_usd", naive_baseline="naive_usd"),
            on=["hs6", "iso3", "year"], how="left",
        )
        .with_columns(pl.col("actual_exports").fill_null(0.0))
        .select(list(PANEL_SCHEMA))
        .cast(PANEL_SCHEMA)
        .sort("hs6", "iso3", "year")
    )


def _grid_columns(chunk: Chunk) -> pl.DataFrame:
    """A UK grid's pairs and Actual Exports only, without loading its features."""
    columns = ("hs6", "importer", "value_usd")
    if isinstance(chunk, pl.DataFrame):
        return chunk.select(columns)
    if isinstance(chunk, pl.LazyFrame):
        return chunk.select(columns).collect()
    return pl.scan_parquet(chunk).select(columns).collect()


# --- Inputs -------------------------------------------------------------------------------------


def fixture_inputs(world: World) -> Inputs:
    """The fixture world's per-year training rows and UK grids (each year's own grid), and its Flows."""
    from exportspredictor.pipeline import fixture_training

    flows, sources = world.flows, world.sources
    years = sorted(flows["year"].unique().to_list())

    def flows_of_year(year: int) -> pl.DataFrame:
        return flows.filter(pl.col("year") == year)

    return Inputs(
        training=fixture_training(world, baseline_year=max(years) + 1),
        grids={y: build_features(uk_grid(flows_of_year(y), FIXTURE_UK), flows, sources) for y in years},
        year_flows=flows_of_year,
    )


def real_inputs(root: Path | None = None, *, feature_set_name: str | None = None) -> Inputs:
    """The real per-year feature files of a feature set (paths: a year is read only when used) and BACI Flows."""
    root = Path(root or data_dir())
    folder = features_dir(root, feature_set_name=feature_set_name)
    years = json.loads((folder / "features.json").read_text(encoding="utf-8"))["years"]
    to_iso3 = load_sources(root, feature_set_name=feature_set_name)[1]
    return Inputs(
        training={y: folder / f"training_{y}.parquet" for y in years},
        grids={y: folder / f"uk_grid_{y}.parquet" for y in years},
        year_flows=lambda year: load_year_flows(year, to_iso3, root),
    )


# --- The saved panel ----------------------------------------------------------------------------


def panel_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / "base_years"


def panel_path(root: Path | None = None) -> Path:
    """The saved panel: `<data root>/derived/base_years/panel.parquet`."""
    return panel_dir(root) / PANEL_FILE


def models_cache(root: Path | None = None) -> Path:
    """Where the real base-year models are cached under the data root."""
    return Path(root or data_dir()) / "derived" / "base_year_models"


def save_panel(panel: pl.DataFrame, root: Path | None = None, *, info: Mapping) -> Path:
    """Write the panel and, beside it, `info` (the run's record) as JSON; return the panel's path."""
    target = panel_path(root)
    downloads.write_parquet(panel.select(list(PANEL_SCHEMA)).cast(PANEL_SCHEMA), target)
    with downloads.writing(target.with_name(PANEL_INFO_FILE)) as part:
        part.write_text(json.dumps(dict(info), indent=2), encoding="utf-8")
    return target


def load_panel(root: Path | None = None) -> pl.DataFrame:
    """The saved scored UK panel (`PANEL_SCHEMA`): hs6, iso3, year, export_potential, naive_baseline, actual_exports."""
    path = panel_path(root)
    if not path.exists():
        raise FileNotFoundError(f"No base-year panel at {path}; run `python -m exportspredictor.base_years`")
    return pl.read_parquet(path).select(list(PANEL_SCHEMA))


def run(
    root: Path | None = None,
    artefacts_root: Path | None = None,
    *,
    base_years: Sequence[int] = BASE_YEARS,
    panel_years: Sequence[int] = PANEL_YEARS,
    device: str = "cuda",
) -> Path:
    """Build the panel from the newest artefact's Configuration on the real feature files, and save it."""
    from exportspredictor import artefacts
    from exportspredictor.paths import artefacts_dir

    root = Path(root or data_dir())
    folder = artefacts.latest_folder(Path(artefacts_root or artefacts_dir()))
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    configuration = Configuration.from_metadata(metadata)
    if list(configuration.features) != list(FEATURES):
        raise ValueError(f"The newest artefact was trained on {configuration.features}, not the current features")
    if configuration.feature_set != feature_set():
        raise ValueError(
            f"The newest artefact was trained on the feature set {configuration.feature_set!r}, but this run reads "
            f"{feature_set()!r}; set EXPORTSPREDICTOR_FEATURE_SET={configuration.feature_set}"
        )
    started = time.perf_counter()
    records: list[dict] = []
    panel = build_panel(
        real_inputs(root, feature_set_name=configuration.feature_set or None), configuration,
        base_years=base_years, panel_years=panel_years, device=device, cache=models_cache(root), records=records,
    )
    info = {
        "artefact_version": metadata["version"],
        "feature_set": configuration.feature_set,
        "base_years": list(base_years),
        "panel_years": list(panel_years),
        "test_year_read": False,
        "configuration": configuration.settings([]) | {"model_form": configuration.model_form},
        "base_year_models": records,
        "rows": panel.height,
        "pairs": panel.select("hs6", "iso3").n_unique(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    path = save_panel(panel, root, info=info)
    print(json.dumps({k: v for k, v in info.items() if k not in ("configuration",)}), flush=True)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m exportspredictor.base_years [--device cpu]`: train or read the 25 base-year models and save the panel."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(run(device=args.device), flush=True)


if __name__ == "__main__":
    main()
