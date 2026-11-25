"""The real feature ablation and learning curve: their saved results and the package settings agree."""

import pytest

from exportspredictor import experiments, features


@pytest.fixture(scope="module")
def results():
    if not experiments.results_path().exists():
        pytest.skip("No saved experiment results; run `python -m exportspredictor.experiments build|ablation|curve`")
    saved = experiments.load()
    if "learning_curve" not in saved:
        pytest.skip("The learning curve has not been run yet")
    return saved


def test_every_real_run_was_measured_on_2022_only(results):
    rows = results["ablation"] + results["learning_curve"]
    assert {row["validation_year"] for row in rows} == {2022}
    assert results["plan"]["validation_year"] == 2022 < experiments.TEST_YEAR


def test_the_kept_features_are_the_package_features(results):
    assert results["kept_features"] == experiments.kept_features(results["ablation"])
    assert results["kept_features"] == features.FEATURES
    assert results["curve_features"] == features.FEATURES


def test_the_zero_rate_setting_is_a_measured_point_that_fits_the_budget(results):
    measured = [row for row in results["learning_curve"] if row["zero_rate"] == features.ZERO_SAMPLING_RATE]
    assert measured
    assert any(row["fits_budget"] for row in measured)
