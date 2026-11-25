"""The base-year fold models and the scored UK panel behind ITC's realisation test (spec #156)."""

from dataclasses import replace

import polars as pl
import pytest

from exportspredictor import base_years, ledger, pipeline
from exportspredictor.backtests import LeakageError
from exportspredictor.fixture import UK

# The fixture world holds 2021-2024; its test year is 2023, so its one base year is 2022 (Market Reach reads 2021).
FIXTURE_BASE_YEARS = (2022,)
FIXTURE_PANEL_YEARS = (2021, 2022)


@pytest.fixture(scope="module")
def configuration(fixture_artefacts):
    return base_years.Configuration.from_metadata(fixture_artefacts.metadata)


@pytest.fixture(scope="module")
def fixture_inputs(fixture_world):
    return base_years.fixture_inputs(fixture_world)


def _panel(inputs, configuration, cache=None):
    return base_years.build_panel(
        inputs, configuration, base_years=FIXTURE_BASE_YEARS, panel_years=FIXTURE_PANEL_YEARS, test_year=2023,
        uk=UK, device="cpu", cache=cache,
    )


def test_training_base_year_models_on_a_year_after_the_base_year_raises_leakage_error(fixture_world, configuration):
    chunks = pipeline.fixture_training(fixture_world, baseline_year=2024)  # 2021-2023

    with pytest.raises(LeakageError):
        base_years.train_base_year_models(chunks, base_year=2022, configuration=configuration, uk=UK, device="cpu")

    kept = {year: chunk for year, chunk in chunks.items() if year <= 2022}
    models = base_years.train_base_year_models(kept, base_year=2022, configuration=configuration, uk=UK, device="cpu")
    assert models.training_years == [2021, 2022]
    assert len(models.models) == configuration.n_folds


def test_base_year_models_are_cached_by_their_settings_and_a_second_run_reads_no_training_data(
    fixture_world, configuration, tmp_path
):
    chunks = {y: c for y, c in pipeline.fixture_training(fixture_world, baseline_year=2023).items() if y <= 2022}
    first = base_years.train_base_year_models(
        chunks, base_year=2022, configuration=configuration, uk=UK, device="cpu", cache=tmp_path
    )

    class Unreadable(dict):
        def __getitem__(self, year):
            raise AssertionError(f"read training year {year}")

    again = base_years.train_base_year_models(
        Unreadable(chunks), base_year=2022, configuration=configuration, uk=UK, device="cpu", cache=tmp_path
    )
    assert [bytes(m.save_raw()) for m in again.models] == [bytes(m.save_raw()) for m in first.models]

    # Other settings are other models: a different round count misses the cache.
    other = replace(configuration, rounds=configuration.rounds + 1)
    folders = {
        base_years.models_dir(tmp_path, base_year=2022, configuration=c, training_years=[2021, 2022])
        for c in (configuration, other)
    }
    assert len(folders) == 2


def test_the_panel_holds_every_uk_pair_in_every_year_scored_only_in_base_years(
    fixture_world, fixture_inputs, configuration
):
    panel = _panel(fixture_inputs, configuration)

    assert panel.schema == pl.Schema(base_years.PANEL_SCHEMA)
    assert list(panel.columns) == ["hs6", "iso3", "year", "export_potential", "naive_baseline", "actual_exports"]
    pairs = panel.select("hs6", "iso3").unique()
    assert panel.height == pairs.height * len(FIXTURE_PANEL_YEARS)
    assert not panel.select("hs6", "iso3", "year").is_duplicated().any()
    # The fixture's UK grid: 5 Products (its Excluded Product left out) x 4 Markets.
    assert pairs.height == 5 * 4
    scored = panel.filter(pl.col("year") == 2022)
    assert scored["export_potential"].is_not_null().all() and (scored["export_potential"] > 0).all()
    assert scored["naive_baseline"].is_not_null().all()
    unscored = panel.filter(pl.col("year") == 2021)
    assert unscored["export_potential"].is_null().all() and unscored["naive_baseline"].is_null().all()

    flows = fixture_world.flows.filter((pl.col("exporter") == UK) & pl.col("year").is_in(FIXTURE_PANEL_YEARS))
    expected = flows.select(
        pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String), year=pl.col("year").cast(pl.Int16),
        expected="value_usd",
    )
    joined = panel.join(expected, on=["hs6", "iso3", "year"], how="left")
    assert joined.select((pl.col("actual_exports") == pl.col("expected").fill_null(0.0)).all()).item()


def test_the_panel_never_reads_the_test_year_or_later_and_leaves_the_ledger_untouched(
    fixture_world, fixture_inputs, configuration, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("the panel touched the Test-year Ledger")

    for name in ("record_read", "record_rescore", "complete_read", "complete_rescores"):
        monkeypatch.setattr(ledger, name, refuse)

    later = pl.col("year") >= 2023
    poisoned_world = replace(
        fixture_world,
        flows=fixture_world.flows.with_columns(
            value_usd=pl.when(later).then(pl.col("value_usd") * 1000 + 1).otherwise("value_usd")
        ),
    )
    poisoned = base_years.fixture_inputs(poisoned_world)

    read = []

    class Recording(dict):
        def __getitem__(self, year):
            read.append(year)
            return super().__getitem__(year)

    def flows(year):
        read.append(year)
        return poisoned.year_flows(year)

    recording = base_years.Inputs(training=Recording(poisoned.training), grids=Recording(poisoned.grids), year_flows=flows)

    assert _panel(recording, configuration).equals(_panel(fixture_inputs, configuration))
    assert read and max(read) < 2023

    with pytest.raises(LeakageError):
        base_years.build_panel(
            fixture_inputs, configuration, base_years=(2022,), panel_years=(2021, 2022, 2023), test_year=2023,
            uk=UK, device="cpu",
        )


def test_the_panel_is_saved_under_the_data_root_and_loaded_back(fixture_inputs, configuration, tmp_path):
    panel = _panel(fixture_inputs, configuration)

    path = base_years.save_panel(panel, tmp_path, info={"artefact_version": "fixture-test"})

    assert path == base_years.panel_path(tmp_path)
    assert path.is_relative_to(tmp_path / "derived")
    assert base_years.load_panel(tmp_path).equals(panel)
