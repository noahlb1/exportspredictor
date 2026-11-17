"""Lift and Event Backtests: Opportunities and Scenario Effects checked against what actually happened."""

import polars as pl
import pytest

from exportspredictor import backtests


def _pairs(gaps, grew, exporters=None, potential=None):
    n = len(gaps)
    return pl.DataFrame(
        {
            "exporter": exporters or ["A", "B"] * (n // 2),
            "export_gap_usd": [float(g) for g in gaps],
            "export_potential_usd": [float(p) for p in (potential or [100.0] * n)],
            "grew": grew,
            "weight": [1.0] * n,
        }
    )


# --- Lift ------------------------------------------------------------------------------


def test_lift_compares_the_top_decile_by_export_gap_with_all_pairs():
    # 20 pairs; the two largest gaps both grew, 5 of 20 grew overall.
    gaps = list(range(20, 0, -1))
    grew = [True, True, True, False, False, True, True] + [False] * 13

    result = backtests.lift(_pairs(gaps, grew), resamples=200)

    assert result["top_rate"] == pytest.approx(1.0)
    assert result["base_rate"] == pytest.approx(5 / 20)
    assert result["lift"] == pytest.approx(4.0)
    assert result["by_decile"][0] == {"decile": 1, "rate": pytest.approx(1.0), "pairs": pytest.approx(2.0)}


def test_lift_weights_sampled_zero_cells_by_their_weight():
    pairs = _pairs([10, 9, 1, 0], [True, False, False, True]).with_columns(weight=pl.Series([1.0, 1.0, 1.0, 7.0]))

    result = backtests.lift(pairs, resamples=50)

    # 10 weighted pairs: the top tenth is the largest gap alone; 8 of 10 grew.
    assert result["top_rate"] == pytest.approx(1.0)
    assert result["base_rate"] == pytest.approx(8 / 10)


def test_lift_has_a_bootstrap_confidence_interval_around_it():
    gaps = list(range(200, 0, -1))
    grew = [i % 3 == 0 or i < 10 for i in range(200)]
    exporters = [f"E{i % 10}" for i in range(200)]

    result = backtests.lift(_pairs(gaps, grew, exporters), resamples=300, seed=1)

    low, high = result["ci"]
    assert low <= result["lift"] <= high
    assert low < high
    assert result["ci_level"] == 0.95
    assert result["bootstrap_unit"] == "Exporter"


def test_size_matched_lift_compares_with_pairs_of_similar_export_potential():
    # The largest gaps are also the largest pairs, and every large pair grew: the
    # size-matched Lift is 1 (no better than other pairs of the same size), the plain Lift is not.
    n = 100
    gaps = list(range(n, 0, -1))
    potential = list(range(n, 0, -1))
    grew = [i < 10 or i % 4 == 0 for i in range(n)]

    result = backtests.lift(_pairs(gaps, grew, potential=potential), resamples=50)

    assert result["lift"] > 1.5
    assert result["size_matched"]["lift"] == pytest.approx(1.0)


def test_grows_materially_needs_both_the_share_and_the_dollar_threshold():
    frame = pl.DataFrame({"before": [100.0, 0.0, 1e6, 1e6], "after": [1e6, 2e5, 1.2e6, 1.6e6]})

    grew = frame.select(backtests.grows_materially("before", "after", min_share=0.5, min_usd=1e5)).to_series()

    assert grew.to_list() == [True, True, False, True]


def test_held_out_exporters_are_a_deterministic_fifth():
    exporters = [f"C{i:02d}" for i in range(50)]

    held = backtests.held_out_exporters(exporters, share=0.2)

    assert len(held) == 10
    assert held == backtests.held_out_exporters(list(reversed(exporters)), share=0.2)
    assert backtests.held_out_exporters(["GBR", "DEU", "FRA", "USA"], share=0.2) != []


# --- Event Backtest: the observed difference-in-differences ------------------------------


def _flows(rows):
    return pl.DataFrame(rows, schema=["exporter", "importer", "hs6", "year", "value_usd"], orient="row")


def test_observed_effect_is_the_treated_change_relative_to_the_control_change():
    # EU -> JPN rises 50% from the pre-period, EU -> USA rises 20%: the observed effect is 1.5 / 1.2 - 1 = +25%.
    flows = _flows(
        [
            ("DEU", "JPN", "1", 2018, 60.0), ("FRA", "JPN", "2", 2018, 40.0),
            ("DEU", "JPN", "1", 2019, 90.0), ("FRA", "JPN", "2", 2019, 60.0),
            ("DEU", "USA", "1", 2018, 50.0), ("FRA", "USA", "2", 2018, 50.0),
            ("DEU", "USA", "1", 2019, 60.0), ("FRA", "USA", "2", 2019, 60.0),
            ("DEU", "CHN", "1", 2019, 999.0),  # not a control Market
        ]
    )
    predicted = pl.DataFrame({"hs6": ["1", "2"], "year": [2019, 2019], "on_usd": [75.0, 50.0], "off_usd": [60.0, 40.0], "weight": [1.0, 1.0]})

    result = backtests.compare(
        flows, predicted, exporters=["DEU", "FRA"], treated=["JPN"], controls=["USA"],
        pre_years=[2018], post_years=[2019], resamples=100,
    )

    year = result["by_year"][0]
    assert year["year"] == 2019
    assert year["observed"] == pytest.approx(0.25)
    assert year["predicted"] == pytest.approx(0.25)
    assert result["pooled"]["observed"] == pytest.approx(0.25)
    low, high = year["observed_ci"]
    assert low <= 0.25 <= high


def test_predicted_effect_weights_sampled_zero_cells():
    flows = _flows([("A", "J", "1", 2018, 1.0), ("A", "J", "1", 2019, 1.0), ("A", "U", "1", 2018, 1.0), ("A", "U", "1", 2019, 1.0)])
    predicted = pl.DataFrame({"hs6": ["1", "1"], "year": [2019, 2019], "on_usd": [12.0, 3.0], "off_usd": [10.0, 1.0], "weight": [1.0, 10.0]})

    result = backtests.compare(
        flows, predicted, exporters=["A"], treated=["J"], controls=["U"], pre_years=[2018], post_years=[2019], resamples=10
    )

    # (12 - 10 + 10 * (3 - 1)) / (10 + 10 * 1) = 22 / 20
    assert result["by_year"][0]["predicted"] == pytest.approx(1.1)


def test_control_markets_are_similar_markets_without_an_agreement_largest_first():
    rows = pl.DataFrame(
        {
            "exporter": ["DEU"] * 5,
            "importer": ["USA", "CAN", "AUS", "IND", "JPN"],
            "year": [2018] * 5,
            "value_usd": [500.0, 400.0, 100.0, 900.0, 800.0],
            "trade_agreement": [0, 1, 0, 0, 0],
        }
    )

    controls = backtests.control_markets(
        rows, exporters=["DEU"], treated=["JPN"], pre_years=[2018], high_income={"USA", "CAN", "AUS", "JPN"}, n=5
    )

    # CAN has an agreement, IND is not high income, JPN is treated.
    assert controls == ["USA", "AUS"]
