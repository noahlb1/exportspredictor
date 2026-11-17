"""Training scope and sampling (ADR 0004): which rows train the model, at what weight, and
whether they fit the row budget.

Leakage guard: UK rows and UK Zero Cells are never sampled and carry weight 1, so every
UK pair a fold model trains on is the whole pair, and each fold model trains on 4/5 of
UK pairs, zeros included. Rows are built through the package's public functions on the
fixture world.
"""

import polars as pl
import pytest
import xgboost as xgb
from hypothesis import given, settings
from hypothesis import strategies as st

from exportspredictor.features import FEATURES, KEYS, build_features, training_rows
from exportspredictor.fixture import UK, build_world
from exportspredictor.folds import N_FOLDS, assign_folds, outside_fold
from exportspredictor.pipeline import build_artefacts
from exportspredictor.scope import TOP_EXPORTERS, exporter_scope
from exportspredictor.training import (
    GPU_CHECK,
    RATE_STEP,
    RowBudgetExceeded,
    RowCounts,
    check_row_budget,
    count_rows,
    fit_non_uk_rate,
    preparer,
    row_budget,
)

YEAR = 2023


@pytest.fixture(scope="module")
def world():
    return build_world()


def _is_uk() -> pl.Expr:
    return pl.col("exporter") == UK


def test_every_uk_zero_cell_is_kept_at_weight_one_whatever_the_zero_rate(world):
    every_zero = training_rows(world.flows, rate=1.0).filter(pl.col("value_usd") == 0)
    uk_zeros = every_zero.filter(_is_uk())
    assert uk_zeros.height > 0  # the fixture UK has Zero Cells to lose

    for seed in range(5):
        sampled = training_rows(world.flows, rate=0.05, seed=seed).filter(pl.col("value_usd") == 0)
        kept = sampled.filter(_is_uk())
        assert kept.select(KEYS).sort(KEYS).equals(uk_zeros.select(KEYS).sort(KEYS))
        assert (kept["weight"] == 1.0).all()
        others = sampled.filter(~_is_uk())
        assert others.height < every_zero.filter(~_is_uk()).height  # other Exporters' zeros are sampled
        assert (others["weight"] == 20.0).all()


# --- The Exporter scope -------------------------------------------------------------------


def test_the_scope_is_the_top_exporters_by_mean_gdp_over_2012_to_2023_with_the_uk_always_in():
    gdp = {  # mean GDP over 2012–2023 decides; 2024 and missing years do not
        "AAA": {2012: 10.0, 2023: 10.0},
        "BBB": {2012: 1.0, 2023: 1.0, 2024: 1000.0},
        "CCC": {2012: 2.0, 2023: None},
        "DDD": {2012: 3.0, 2023: 3.0},
        UK: {2012: 0.5, 2023: 0.5},
    }
    country_year = pl.DataFrame(
        [(iso3, year, value) for iso3, years in gdp.items() for year, value in years.items()],
        schema={"iso3": pl.String, "year": pl.Int32, "gdp_usd": pl.Float64},
        orient="row",
    )

    assert exporter_scope(country_year, uk=UK, n=2) == ["AAA", "DDD", UK]
    assert exporter_scope(country_year, uk=UK, n=3) == ["AAA", "CCC", "DDD", UK]
    assert len(exporter_scope(country_year, uk=UK, n=TOP_EXPORTERS)) == 5


def test_the_scope_keeps_every_importer_and_drops_only_rows_not_features(world):
    flows = world.flows.filter(pl.col("year") == YEAR)
    scope = ["DEU", "FRA"]  # the UK is always in; the USA is out of scope

    everyone = build_features(training_rows(flows, rate=1.0), flows, world.sources)
    scoped = build_features(training_rows(flows, rate=1.0, exporters=scope), flows, world.sources)

    assert set(scoped["exporter"]) == {UK, "DEU", "FRA"}
    assert set(scoped["importer"]) == set(everyone["importer"])
    in_scope = everyone.filter(pl.col("exporter") != "USA").sort(KEYS)
    assert scoped.sort(KEYS).select(*KEYS, "value_usd", "weight", *FEATURES).equals(
        in_scope.select(*KEYS, "value_usd", "weight", *FEATURES)
    )


# --- The row budget -----------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(
    uk=st.integers(0, 50_000_000),
    non_uk=st.integers(1, 1_000_000_000),
    budget=st.integers(1_000_000, 200_000_000),
)
def test_the_derived_non_uk_rate_never_exceeds_the_row_budget(uk, non_uk, budget):
    counts = RowCounts(uk=uk, non_uk=non_uk)
    try:
        rate = fit_non_uk_rate(counts, budget)
    except RowBudgetExceeded:  # only when not even the smallest rate fits
        assert counts.high(RATE_STEP) > budget
        return

    assert 0 < rate <= 1
    assert counts.high(rate) <= budget  # even a sample four standard deviations large fits
    check_row_budget(counts, rate, budget)  # and the derived rate passes the pre-training check
    if rate < 1:  # it is the largest such rate, to the rate's precision
        assert counts.high(min(1.0, rate + 2 * RATE_STEP)) > budget


def test_a_worked_budget():
    # 10 M UK rows and 200 M other rows in a 110 M budget leave 100 M rows for a ~50% sample.
    rate = fit_non_uk_rate(RowCounts(uk=10_000_000, non_uk=200_000_000), 110_000_000)
    assert 0.499 <= rate < 0.5
    assert fit_non_uk_rate(RowCounts(uk=10, non_uk=20), 110_000_000) == 1.0


def test_an_over_budget_rate_fails_the_check():
    counts = RowCounts(uk=10_000_000, non_uk=200_000_000)
    check_row_budget(counts, 0.4, 110_000_000)
    with pytest.raises(RowBudgetExceeded, match="110,000,000"):
        check_row_budget(counts, 0.65, 110_000_000)


def test_the_row_budget_shrinks_as_features_are_added():
    assert row_budget(len(FEATURES) + 4) < row_budget(len(FEATURES))
    assert 90_000_000 < row_budget(len(FEATURES)) < 130_000_000  # about 110 M rows (ADR 0004)


def test_rows_are_counted_by_uk_and_other_exporters(world, tmp_path):
    rows = training_rows(world.flows, rate=1.0)
    path = tmp_path / "rows.parquet"
    rows.write_parquet(path)
    uk = rows.filter(_is_uk()).height

    assert count_rows([rows.lazy(), path], uk=UK) == RowCounts(uk=2 * uk, non_uk=2 * (rows.height - uk))


# --- Through the pipeline -----------------------------------------------------------------


def test_the_artefact_records_the_scope_both_rates_and_the_row_budget(fixture_artefacts):
    metadata = fixture_artefacts.metadata

    # The fixture world has seven countries, so all are top-100 economies.
    assert metadata["exporter_scope"] == ["CAN", "DEU", "FRA", UK, "IND", "JPN", "USA"]
    assert metadata["uk_zero_sampling_rate"] == 1.0
    assert metadata["zero_sampling_rate"] == pytest.approx(0.5)
    assert metadata["non_uk_sampling_rate"] == 1.0
    assert metadata["row_budget"] == row_budget(len(FEATURES))
    rows = metadata["training_rows"]
    assert 0 < rows["uk"] < rows["non_uk"]
    assert metadata["gpu_check"] == GPU_CHECK


def test_an_over_budget_configuration_fails_before_training(fixture_world, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(xgb, "train", never)

    with pytest.raises(RowBudgetExceeded, match="budget of 100 rows"):
        build_artefacts(fixture_world, version="over-budget", row_budget=100)


# --- The training matrix's rows -------------------------------------------------------------


def test_sampling_other_exporters_rows_never_touches_uk_rows_or_uk_zero_cells(world):
    rows = training_rows(world.flows, rate=0.05)
    uk_rows = rows.filter(_is_uk()).sort(KEYS)

    prepared = preparer(uk=UK, non_uk_rate=0.3, seed=1)(rows)

    assert prepared.filter(_is_uk()).sort(KEYS).equals(uk_rows)
    assert (uk_rows["weight"] == 1.0).all()
    others = prepared.filter(~_is_uk())
    assert 0 < others.height < rows.filter(~_is_uk()).height
    flows, zeros = others.filter(pl.col("value_usd") > 0), others.filter(pl.col("value_usd") == 0)
    assert flows["weight"].to_list() == pytest.approx([1 / 0.3] * flows.height)
    assert zeros["weight"].to_list() == pytest.approx([20 / 0.3] * zeros.height)


def test_each_fold_model_trains_on_the_uk_pairs_outside_its_fold_zero_cells_included(world):
    rows = training_rows(world.flows, rate=0.05)
    uk_rows = rows.filter(_is_uk())
    folds = assign_folds(uk_rows.rename({"importer": "iso3"}))

    for k in range(N_FOLDS):
        prepared = preparer(uk=UK, non_uk_rate=0.3, keep=outside_fold(UK, N_FOLDS, k))(rows)
        expected = folds.filter(pl.col("fold") != k).drop("fold").rename({"iso3": "importer"})
        assert prepared.filter(_is_uk()).sort(KEYS).equals(expected.sort(KEYS))
    assert uk_rows.filter(pl.col("value_usd") == 0).height > 0
