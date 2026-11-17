"""Potential Range guards (ADR 0007): calibrated on other Exporters' Baseline-year rows, never the UK's.

Calibration rows are the fixture world's 2024 training rows; fold models are trained
through the package on the years before, as the pipeline does.
"""

import numpy as np
import polars as pl
import pytest

from exportspredictor import conformal
from exportspredictor.features import (
    FEATURES,
    FeatureTransform,
    build_features,
    training_rows,
)
from exportspredictor.fixture import UK
from exportspredictor.folds import train_fold_models

PARAMS = {"max_depth": 3, "eta": 0.3}
ROUNDS = 20


@pytest.fixture(scope="module")
def transform_and_models(fixture_world):
    flows = fixture_world.flows.filter(pl.col("year") < 2024)
    training = build_features(training_rows(flows, rate=1.0), flows, fixture_world.sources)
    transform = FeatureTransform.fit(training.lazy(), [2021, 2022, 2023])
    return transform, train_fold_models([training], uk=UK, params=PARAMS, rounds=ROUNDS, device="cpu", transform=transform)


@pytest.fixture(scope="module")
def baseline_year_rows(fixture_world):
    flows = fixture_world.flows.filter(pl.col("year") == 2024)
    return build_features(training_rows(flows, rate=1.0), fixture_world.flows, fixture_world.sources)


@pytest.fixture(scope="module")
def scored(baseline_year_rows, transform_and_models):
    transform, models = transform_and_models
    return conformal.score_calibration(baseline_year_rows, models, transform, uk=UK)


def test_no_uk_row_is_ever_a_calibration_row(baseline_year_rows, scored):
    assert baseline_year_rows.filter(pl.col("exporter") == UK).height > 0
    assert scored.height == baseline_year_rows.filter(pl.col("exporter") != UK).height
    assert UK not in scored["exporter"].to_list()


def test_each_calibration_row_is_scored_by_its_hash_assigned_fold_model(
    baseline_year_rows, transform_and_models, scored
):
    transform, models = transform_and_models
    features = transform.apply(baseline_year_rows.filter(pl.col("exporter") != UK)).with_columns(
        pl.col("exporter").cast(pl.String), hs6=pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String)
    )
    rows = scored.join(features, on=["exporter", "hs6", "iso3"])
    assert rows.height == scored.height
    for k, model in enumerate(models):
        own = rows.filter(pl.col("fold") == k)
        assert own.height > 0, f"no calibration row in fold {k}"
        expected = model.inplace_predict(own.select(FEATURES).to_numpy().astype(np.float32))
        np.testing.assert_allclose(own["export_potential_usd"].to_numpy(), expected, rtol=1e-6)


def test_the_fold_hash_includes_the_exporter_and_ignores_row_order(baseline_year_rows, transform_and_models, scored):
    transform, models = transform_and_models
    again = conformal.score_calibration(baseline_year_rows.reverse(), models, transform, uk=UK)
    keys = ["exporter", "hs6", "iso3"]
    assert again.sort(keys).select(*keys, "fold").equals(scored.sort(keys).select(*keys, "fold"))
    # The same Product–Market pair lands in different folds for different Exporters.
    folds_per_pair = scored.group_by("hs6", "iso3").agg(pl.col("fold").n_unique())["fold"]
    assert folds_per_pair.max() > 1


def _calibration(potentials, residuals, weights=None, exporter="DEU") -> pl.DataFrame:
    """Scored calibration rows built by hand, shaped like `conformal.score_calibration`'s output."""
    potentials = [float(p) for p in potentials]
    residuals = [float(r) for r in residuals]
    return pl.DataFrame(
        {
            "exporter": exporter,
            "hs6": [f"{i:06d}" for i in range(len(potentials))],
            "iso3": "JPN",
            "weight": weights or [1.0] * len(potentials),
            "fold": 0,
            "actual_exports_usd": [p + r for p, r in zip(potentials, residuals, strict=True)],
            "export_potential_usd": potentials,
            "residual_usd": residuals,
        }
    )


def test_each_band_takes_the_10th_and_90th_percentiles_of_its_residuals():
    # Four equal-count bands of 25 rows; in each, residuals run -12..12 times the band's scale.
    potentials = range(1, 101)
    residuals = [(i % 25 - 12) * (1 + i // 25) for i in range(100)]

    bands = conformal.fit_bands(_calibration(potentials, residuals), n_bands=4, min_rows=10)

    assert bands["rows"].to_list() == [25, 25, 25, 25]
    # The 10th percentile of 25 sorted values is the 3rd (-10), the 90th the 23rd (+10).
    assert bands["low_offset_usd"].to_list() == [-10.0, -20.0, -30.0, -40.0]
    assert bands["high_offset_usd"].to_list() == [10.0, 20.0, 30.0, 40.0]
    assert bands["min_export_potential_usd"].to_list() == [0.0, 26.0, 51.0, 76.0]


def test_a_band_below_the_minimum_rows_merges_with_a_neighbour():
    bands = conformal.fit_bands(_calibration(range(100), [0] * 100), n_bands=20, min_rows=30)

    assert bands["rows"].sum() == 100
    assert bands["rows"].min() >= 30
    assert 1 < bands.height <= 3
    assert bands["min_export_potential_usd"].is_sorted()


def test_sampled_zero_cells_count_by_their_weight():
    # Nine rows of +1 and one row of -100 weighted 9: half the weight sits at -100.
    bands = conformal.fit_bands(
        _calibration([5] * 10, [1] * 9 + [-100], weights=[1.0] * 9 + [9.0]), n_bands=1, min_rows=1
    )

    assert bands["low_offset_usd"].item() == -100.0
    assert bands["high_offset_usd"].item() == 1.0


def test_the_potential_range_is_the_band_offsets_around_export_potential_clipped_at_zero():
    bands = conformal.fit_bands(
        _calibration(range(1, 101), [(i % 25 - 12) * (1 + i // 25) for i in range(100)]), n_bands=4, min_rows=10
    )
    pairs = pl.DataFrame({"export_potential_usd": [5.0, 30.0, 1000.0, 0.0]})

    ranged = conformal.potential_range(pairs, bands)

    assert ranged["potential_range_low_usd"].to_list() == [0.0, 10.0, 960.0, 0.0]
    assert ranged["potential_range_high_usd"].to_list() == [15.0, 50.0, 1040.0, 10.0]


def test_an_uncertain_opportunity_is_an_opportunity_whose_actual_exports_lie_inside_its_range():
    bands = conformal.fit_bands(_calibration([100.0] * 10, range(-40, 60, 10)), n_bands=1, min_rows=1)
    # Residuals -40..50: offsets -40 (1st of 10) and +40 (9th), so Export Potential 100 gives 60 to 140.
    baseline = pl.DataFrame(
        {
            "export_potential_usd": [100.0, 100.0, 100.0, 100.0, 100.0],
            "actual_exports_usd": [80.0, 60.0, 59.0, 120.0, 100.0],
            "export_gap_usd": [20.0, 40.0, 41.0, -20.0, 0.0],
        }
    )

    ranged = conformal.with_potential_range(baseline, bands)

    assert ranged["potential_range_low_usd"].to_list() == [60.0] * 5
    assert ranged["potential_range_high_usd"].to_list() == [140.0] * 5
    # Inside and an Opportunity; on the edge; below the range; inside but no Opportunity (twice).
    assert ranged["uncertain"].to_list() == [True, True, False, False, False]


def test_coverage_is_measured_on_a_held_out_half_overall_per_band_and_per_exporter(scored):
    report = conformal.coverage_report(scored, n_bands=4, min_rows=5)

    assert report["target"] == pytest.approx(0.8)
    assert report["calibration_rows"] + report["held_out_rows"] == scored.height
    assert min(report["calibration_rows"], report["held_out_rows"]) > scored.height / 4
    assert {row["exporter"] for row in report["by_exporter"]} == set(scored["exporter"])
    assert sum(row["rows"] for row in report["by_exporter"]) == report["held_out_rows"]
    assert sum(row["rows"] for row in report["by_band"]) == report["held_out_rows"]
    for row in [report, *report["by_exporter"], *report["by_band"]]:
        assert 0.0 <= row["coverage"] <= 1.0
    # The fixture is small, so the held-out half lands only roughly on the 80% target.
    assert report["coverage"] == pytest.approx(0.8, abs=0.15)


def test_the_coverage_text_states_the_held_out_coverage_and_the_uncertain_share():
    report = {
        "target": 0.8, "held_out_rows": 27, "coverage": 0.7142, "opportunities": 11, "uncertain_opportunities": 10,
        "uncertain_opportunity_share": 10 / 11,
    }

    assert conformal.coverage_text(report) == (
        "On a held-out half of the calibration rows (27 other Exporters' rows), 71% of their Actual Exports fell "
        "inside their Potential Range, against a target of 80%. 10 of the 11 Opportunities (91%) are "
        "Uncertain Opportunities."
    )
