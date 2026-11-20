"""Potential Range guards (ADR 0007): calibrated on other Exporters' Baseline-year rows, never the UK's.

Calibration rows are the fixture world's 2024 training rows; fold models are trained
through the package on the years before, as the pipeline does.
"""

import math

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
    """Scored calibration rows built by hand, shaped like `conformal.score_calibration`'s output.

    `residuals` are log residuals: each row's Actual Exports are exp(log(1 + potential) + residual) − 1.
    """
    potentials = [float(p) for p in potentials]
    residuals = [float(r) for r in residuals]
    return pl.DataFrame(
        {
            "exporter": exporter,
            "hs6": [f"{i:06d}" for i in range(len(potentials))],
            "iso3": "JPN",
            "weight": weights or [1.0] * len(potentials),
            "fold": 0,
            "actual_exports_usd": [math.expm1(math.log1p(p) + r) for p, r in zip(potentials, residuals, strict=True)],
            "export_potential_usd": potentials,
            "residual_log": residuals,
        }
    )


def _bands(low_ratio: float, high_ratio: float) -> pl.DataFrame:
    """One band whose offsets multiply 1 + Export Potential by `low_ratio` and `high_ratio`."""
    return pl.DataFrame({
        "band": pl.Series([0], dtype=pl.Int16), "min_export_potential_usd": [0.0], "rows": [1000],
        "low_offset_log": [math.log(low_ratio)], "high_offset_log": [math.log(high_ratio)],
    })


def test_a_calibration_residual_is_on_the_log_scale(scored):
    expected = np.log1p(scored["actual_exports_usd"].to_numpy()) - np.log1p(scored["export_potential_usd"].to_numpy())
    np.testing.assert_allclose(scored["residual_log"].to_numpy(), expected)


def test_each_band_takes_the_10th_and_90th_percentiles_of_its_residuals():
    # Four equal-count bands of 25 rows; in each, residuals run -12..12 times the band's scale.
    potentials = range(1, 101)
    residuals = [(i % 25 - 12) * 0.1 * (1 + i // 25) for i in range(100)]

    bands = conformal.fit_bands(_calibration(potentials, residuals), n_bands=4, min_rows=10)

    assert bands["rows"].to_list() == [25, 25, 25, 25]
    # The 10th percentile of 25 sorted values is the 3rd (-1.0 × scale), the 90th the 23rd (+1.0 × scale).
    assert bands["low_offset_log"].to_list() == pytest.approx([-1.0, -2.0, -3.0, -4.0])
    assert bands["high_offset_log"].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert bands["min_export_potential_usd"].to_list() == [0.0, 26.0, 51.0, 76.0]


def test_a_band_below_the_minimum_rows_merges_with_a_neighbour():
    bands = conformal.fit_bands(_calibration(range(100), [0] * 100), n_bands=20, min_rows=30)

    assert bands["rows"].sum() == 100
    assert bands["rows"].min() >= 30
    assert 1 < bands.height <= 3
    assert bands["min_export_potential_usd"].is_sorted()


def test_sampled_zero_cells_count_by_their_weight():
    # Nine rows of +0.1 and one row of -2 weighted 9: half the weight sits at -2.
    bands = conformal.fit_bands(
        _calibration([5] * 10, [0.1] * 9 + [-2.0], weights=[1.0] * 9 + [9.0]), n_bands=1, min_rows=1
    )

    assert bands["low_offset_log"].item() == pytest.approx(-2.0)
    assert bands["high_offset_log"].item() == pytest.approx(0.1)


def test_the_potential_range_is_the_band_offsets_applied_to_one_plus_export_potential_clipped_at_zero():
    pairs = pl.DataFrame({"export_potential_usd": [99.0, 999.0, 0.0]})

    ranged = conformal.potential_range(pairs, _bands(0.5, 2.0))

    # (1 + 99) × 0.5 − 1 = 49 and (1 + 99) × 2 − 1 = 199; at $0, (1 × 0.5) − 1 is clipped to $0.
    assert ranged["potential_range_low_usd"].to_list() == pytest.approx([49.0, 499.0, 0.0])
    assert ranged["potential_range_high_usd"].to_list() == pytest.approx([199.0, 1999.0, 1.0])


def test_the_range_scales_with_export_potential_within_one_band():
    # A $2m pair and a $1bn pair share the top band: the range is the same ratio, not the same dollars.
    pairs = pl.DataFrame({"export_potential_usd": [2e6, 1e9]})

    ranged = conformal.potential_range(pairs, _bands(0.25, 3.0))

    width = (ranged["potential_range_high_usd"] - ranged["potential_range_low_usd"]).to_list()
    assert width[1] / width[0] == pytest.approx(500.0, rel=1e-3)
    assert ranged["potential_range_low_usd"][1] == pytest.approx(0.25e9, rel=1e-6)


def test_an_uncertain_opportunity_is_an_opportunity_whose_actual_exports_lie_inside_its_range():
    # Export Potential 99 with ratios 0.6 and 1.4 gives a range of 59 to 139.
    baseline = pl.DataFrame(
        {
            "export_potential_usd": [99.0, 99.0, 99.0, 99.0, 99.0],
            "actual_exports_usd": [80.0, 59.5, 58.5, 120.0, 99.0],
            "export_gap_usd": [19.0, 39.5, 40.5, -21.0, 0.0],
        }
    )

    ranged = conformal.with_potential_range(baseline, _bands(0.6, 1.4))

    assert ranged["potential_range_low_usd"].to_list() == pytest.approx([59.0] * 5)
    assert ranged["potential_range_high_usd"].to_list() == pytest.approx([139.0] * 5)
    # Inside and an Opportunity; just inside the edge; just below it; inside but no Opportunity (twice).
    assert ranged["uncertain_opportunity"].to_list() == [True, True, False, False, False]


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

    text = conformal.coverage_text(report)

    for figure in ("27", "71%", "80%", "10 of the 11 Opportunities (91%)"):
        assert figure in text, figure
    assert "Potential Range" in text.split("(91%)")[1]
