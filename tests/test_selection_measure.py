"""The tie-proof Selection Measure (spec #113, ticket #115): ties in Export Potential cannot raise it."""

import polars as pl
import pytest

from exportspredictor.selection_measure import (
    legacy_selection_measure,
    selection_measure,
)


def _rows(potential: dict[str, list[float]], actual: dict[str, list[float]]) -> pl.DataFrame:
    """One row per Product and Market: Markets named M0, M1, ... in each Product's list order."""
    return pl.DataFrame(
        [
            {"hs6": hs6, "importer": f"M{i}", "prediction": p, "value_usd": a}
            for hs6 in potential
            for i, (p, a) in enumerate(zip(potential[hs6], actual[hs6], strict=True))
        ]
    )


def test_with_no_ties_it_is_the_mean_over_products_of_the_ordinary_spearman_correlation():
    # 010101: ranks (1 2 3 4) against (1 2 4 3): rho = 1 - 6*2/(4*15) = 0.8.
    # 020202: ranks (1 2 3 4) against (4 3 2 1): rho = -1.
    rows = _rows(
        {"010101": [1.0, 2.0, 3.0, 4.0], "020202": [10.0, 20.0, 30.0, 40.0]},
        {"010101": [5.0, 6.0, 8.0, 7.0], "020202": [4.0, 3.0, 2.0, 1.0]},
    )

    assert selection_measure(rows, "prediction") == pytest.approx((0.8 - 1.0) / 2)


# Two Products, each ten Markets: the UK sells to four, so six Actual Exports are tied zeros.
# Export Potential's lowest seven Markets are in the order of Actual Exports; its top three are not.
POTENTIAL = {
    "010101": [900.0, 800.0, 700.0, 300.0, 400.0, 50.0, 200.0, 20.0, 100.0, 10.0],
    "020202": [5e6, 2e6, 3e6, 1e4, 3e4, 4e5, 2e4, 2e3, 7e3, 1e3],
}
ACTUAL = {
    "010101": [50.0, 30.0, 40.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "020202": [9e6, 1e6, 5e5, 0.0, 0.0, 2e5, 0.0, 0.0, 0.0, 0.0],
}


def _floored(rows: pl.DataFrame, tied: int) -> pl.DataFrame:
    """Each Product's Export Potential floored at its `tied`-th lowest value: its lowest `tied` Markets tie."""
    floors = rows.group_by("hs6").agg(_floor=pl.col("prediction").sort().get(tied - 1))
    return rows.join(floors, on="hs6").with_columns(pl.col("prediction").clip(lower_bound=pl.col("_floor"))).drop("_floor")


@pytest.mark.parametrize("tied", range(2, 8))
def test_tying_any_lower_part_of_a_ranking_never_raises_the_measure(tied):
    # A tie broken at random orders the tied Markets no better, on average, than a random order; a part already in
    # the order of Actual Exports is the best order there is, so tying it can only lose (rearrangement inequality).
    rows = _rows(POTENTIAL, ACTUAL)

    assert selection_measure(_floored(rows, tied), "prediction") <= selection_measure(rows, "prediction") + 1e-12


def test_the_legacy_measure_is_raised_by_the_same_floor_which_is_why_it_was_replaced():
    rows = _rows(POTENTIAL, ACTUAL)

    assert legacy_selection_measure(_floored(rows, 7), "prediction") > legacy_selection_measure(rows, "prediction")


def test_a_perfect_ranking_scores_one():
    rows = _rows({"010101": [1.0, 2.0, 3.0, 4.0, 5.0]}, {"010101": [10.0, 20.0, 30.0, 40.0, 50.0]})

    assert selection_measure(rows, "prediction") == pytest.approx(1.0)


def test_ties_in_actual_exports_keep_average_ranks():
    # Ranks (1 2 3 4) against (1.5 1.5 3 4): covariance 4.5, variances 5 and 4.5, so rho = sqrt(0.9).
    rows = _rows({"010101": [1.0, 2.0, 3.0, 4.0]}, {"010101": [0.0, 0.0, 1.0, 2.0]})

    assert selection_measure(rows, "prediction") == pytest.approx(0.948683, abs=1e-6)


def test_it_is_the_same_across_calls_and_row_orders_even_when_every_prediction_is_tied():
    rows = _rows(
        {"010101": [5.0] * 10, "020202": [1.0] * 10},
        {"010101": [float(i % 4) for i in range(10)], "020202": [float(i) for i in range(10)]},
    )
    first = selection_measure(rows, "prediction")

    assert first is not None
    assert selection_measure(rows, "prediction") == first
    for seed in range(5):
        assert selection_measure(rows.sample(fraction=1.0, shuffle=True, seed=seed), "prediction") == pytest.approx(
            first, abs=1e-12
        )


# --- Real data: the saved 2022 confirmation grid (skipped where the loop's cache is absent) -----------------


@pytest.fixture(scope="module")
def confirmation_grid():
    from exportspredictor.loop import CONFIRMATION, loop_dir

    folds = sorted((loop_dir() / CONFIRMATION).glob("fold_*.parquet"))
    if len(folds) != 5:
        pytest.skip(f"the five-fold confirmation is not saved in {loop_dir()}")
    return pl.concat(pl.read_parquet(fold) for fold in folds)


@pytest.mark.parametrize("floor_usd", [100, 1_000, 3_000, 10_000])
def test_flooring_the_blend_on_the_2022_confirmation_grid_never_raises_the_measure(confirmation_grid, floor_usd):
    blend = "winner"  # the Blend, the served form, scored out-of-fold on every UK pair of 2022
    floored = confirmation_grid.with_columns(pl.col(blend).clip(lower_bound=floor_usd))

    assert selection_measure(floored, blend) <= selection_measure(confirmation_grid, blend)
