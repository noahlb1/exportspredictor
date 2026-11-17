"""The monotone-constraint comparison: Scenario Effects of a constrained and an unconstrained model."""

import pytest

from exportspredictor import constraints, pipeline
from exportspredictor.features import FeatureTransform
from exportspredictor.fixture import UK


@pytest.fixture(scope="module")
def report(fixture_world, fixture_artefacts):
    metadata = fixture_artefacts.metadata
    return constraints.compare(
        pipeline.fixture_training(fixture_world),
        fixture_artefacts.baseline_features,
        plan=constraints.FIXTURE_PLAN,
        uk=UK,
        params=metadata["params"],
        rounds=metadata["rounds"],
        transform=FeatureTransform.from_dict(metadata["feature_transform"]),
        n_folds=metadata["n_folds"],
        device="cpu",
    )


def test_both_models_are_compared_on_the_same_pairs(report):
    constrained, unconstrained = report["models"]

    assert (constrained["model"], unconstrained["model"]) == ("constrained", "unconstrained")
    for change in ("add", "remove"):
        assert constrained[change]["pairs"] == unconstrained[change]["pairs"] > 0


def test_the_constrained_model_never_moves_against_the_policy_change(report):
    constrained = report["models"][0]

    # Adding a Trade Agreement never lowers Export Potential; removing one never raises it.
    assert constrained["add"]["wrong_sign_share"] == 0
    assert constrained["remove"]["wrong_sign_share"] == 0
    assert constrained["add"]["min_effect_share"] >= 0
    assert constrained["remove"]["max_effect_share"] <= 0


def test_the_report_is_worded_as_an_association(report):
    assert "associates" in report["text"]
    assert "cause" not in report["text"].replace("because", "")


def test_the_fixture_artefacts_carry_the_comparison(fixture_artefacts):
    assert [m["model"] for m in fixture_artefacts.evaluation["monotone_constraints"]["models"]] == [
        "constrained",
        "unconstrained",
    ]
