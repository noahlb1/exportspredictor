"""The one prediction function: every Export Potential the package serves comes from `prediction`."""

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from exportspredictor.features import FEATURES
from exportspredictor.prediction import export_potential, log_contributions


def test_export_potential_reproduces_the_out_of_fold_baseline(fixture_artefacts):
    features = fixture_artefacts.baseline_features
    baseline = fixture_artefacts.baseline.join(features.select("hs6", "iso3"), on=["hs6", "iso3"], how="right")

    predicted = export_potential(features, fixture_artefacts.fold_models)

    assert predicted.to_list() == pytest.approx(baseline["baseline_export_potential_usd"].to_list(), rel=1e-6)


def test_a_row_is_never_scored_by_a_model_that_did_not_hold_its_fold_out(fixture_artefacts):
    features = fixture_artefacts.baseline_features
    models = fixture_artefacts.fold_models
    swapped = [models[1], models[0], *models[2:]]

    with pytest.raises(ValueError, match="held out fold"):
        export_potential(features, swapped)
    with pytest.raises(ValueError, match="no fold model"):
        export_potential(features.with_columns(fold=pl.lit(len(models), features.schema["fold"])), models)


def test_log_contributions_add_up_to_the_same_export_potential(fixture_artefacts):
    features = fixture_artefacts.baseline_features

    contributions, log_potential = log_contributions(features, fixture_artefacts.fold_models)

    assert contributions.shape == (features.height, len(FEATURES) + 1)
    assert contributions.sum(axis=1) == pytest.approx(log_potential, abs=1e-4)
    assert np.exp(log_potential.astype(np.float64)) == pytest.approx(
        export_potential(features, fixture_artefacts.fold_models).to_numpy(), rel=1e-5
    )


ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "src" / "exportspredictor" / "prediction.py"
SOURCES = sorted(
    p
    for p in [*(ROOT / "src" / "exportspredictor").rglob("*.py"), *(ROOT / "api").rglob("*.py"), *(ROOT / "app").rglob("*.py")]
    if p != PREDICTION
)
# How a Booster predicts: in place, or from a DMatrix (`.predict`, SHAP values, margins).
BOOSTER_PREDICTION = re.compile(r"inplace_predict|\bDMatrix\(|pred_contribs|output_margin|pred_interactions")


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: str(path.relative_to(ROOT)))
def test_only_the_prediction_module_predicts_from_a_booster(path):
    found = [
        f"line {n}: {line.strip()}"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if BOOSTER_PREDICTION.search(line)
    ]
    assert found == [], f"{path.name} predicts from a Booster; call exportspredictor.prediction instead"
