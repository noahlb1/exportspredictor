"""The Selection Measure on the largest importing Markets, reported beside the headline (#123)."""

import polars as pl
import pytest

from exportspredictor.large_markets import largest_importers, restricted_measures
from exportspredictor.selection_measure import (
    legacy_selection_measure,
    selection_measure,
)


def _flows() -> pl.DataFrame:
    """Every Exporter's Flows of one year. By total imports: DEU 1,000, FRA 600, JPN 300, KEN 50.

    KEN has the most Flows and the UK's largest Flow, and DEU imports nothing from the UK, so a choice by
    Flow count or by UK Flows alone would differ from the choice by total imports.
    """
    rows = [
        ("USA", "DEU", "010101", 900.0), ("CHN", "DEU", "020202", 100.0),
        ("USA", "FRA", "010101", 500.0), ("GBR", "FRA", "010101", 100.0),
        ("CHN", "JPN", "020202", 300.0),
        ("GBR", "KEN", "010101", 40.0), ("CHN", "KEN", "010101", 5.0), ("USA", "KEN", "020202", 3.0),
        ("IND", "KEN", "020202", 2.0),
        ("FRA", "GBR", "010101", 5_000.0),  # the UK is no Market for UK exports, however much it imports
    ]
    return pl.DataFrame(rows, schema=["exporter", "importer", "hs6", "value_usd"], orient="row")


def _uk_rows() -> pl.DataFrame:
    """The UK's grid: two Products in four Markets, with a prediction and Actual Exports."""
    markets = ["DEU", "FRA", "JPN", "KEN"]
    return pl.DataFrame({
        "exporter": ["GBR"] * 8,
        "importer": markets * 2,
        "hs6": ["010101"] * 4 + ["020202"] * 4,
        # Within each Product: right on the three largest Markets, KEN out of place
        "prediction": [3.0, 2.0, 1.0, 9.0, 30.0, 20.0, 10.0, 90.0],
        "value_usd": [30.0, 20.0, 10.0, 0.0, 3.0, 2.0, 1.0, 0.0],
    })


def test_the_largest_importers_are_chosen_by_total_imports_from_every_exporter():
    assert largest_importers(_flows(), ["DEU", "FRA", "JPN", "KEN"], 2) == ["DEU", "FRA"]
    assert largest_importers(_flows(), ["DEU", "FRA", "JPN", "KEN"], 3) == ["DEU", "FRA", "JPN"]


def test_a_market_with_no_flows_imports_nothing_and_ranks_last():
    assert largest_importers(_flows(), ["NRU", "KEN"], 2) == ["KEN", "NRU"]


@pytest.mark.parametrize("measure", [selection_measure, legacy_selection_measure])
def test_the_measure_is_restricted_to_the_largest_importers_among_the_rows_markets(measure):
    # Worked by hand: on the three largest Markets each Product is ranked exactly (1.0); adding KEN, predicted
    # first but with no exports, gives each Product 1 - 6 * 12 / (4 * 15) = -0.2. Two Markets rank no Product.
    measured = restricted_measures(_uk_rows(), _flows(), "prediction", counts=(2, 3, 4), measure=measure)

    assert measured == {2: None, 3: pytest.approx(1.0), 4: pytest.approx(-0.2)}
