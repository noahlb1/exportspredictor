"""The one prediction function: every Export Potential the package serves comes from `prediction`."""

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from exportspredictor.features import FEATURES
from exportspredictor.prediction import ModelForm, export_potential, log_contributions


def test_export_potential_in_the_served_form_reproduces_the_out_of_fold_baseline(fixture_artefacts):
    features = fixture_artefacts.baseline_features
    baseline = fixture_artefacts.baseline.join(features.select("hs6", "iso3"), on=["hs6", "iso3"], how="right")

    predicted = export_potential(features, fixture_artefacts.fold_models, form=fixture_artefacts.model_form)

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


def test_a_blend_takes_the_weighted_geometric_mean_of_the_model_and_the_naive_baseline(fixture_artefacts):
    rows = fixture_artefacts.baseline_features.head(2)
    model_alone = export_potential(rows, fixture_artefacts.fold_models).to_numpy()
    # Worked example: naive = model × 10^2 gives exp(0.7 · ln m + 0.3 · ln(100 m)) = m × 10^0.6;
    # a naive baseline below $1 counts as $1.
    rows = rows.with_columns(naive_usd=pl.Series([model_alone[0] * 100, 0.0]))

    blended = export_potential(rows, fixture_artefacts.fold_models, form=ModelForm(naive_weight=0.3)).to_numpy()

    assert blended[0] == pytest.approx(model_alone[0] * 10**0.6, rel=1e-6)
    assert blended[1] == pytest.approx(max(model_alone[1], 1.0) ** 0.7, rel=1e-6)


def test_a_blend_needs_the_naive_baseline_on_every_row(fixture_artefacts):
    rows = fixture_artefacts.baseline_features.drop("naive_usd", strict=False)

    with pytest.raises(ValueError, match="naive_usd"):
        export_potential(rows, fixture_artefacts.fold_models, form=ModelForm(naive_weight=0.3))


def test_a_blends_contributions_add_up_to_its_log_export_potential_with_the_naive_share_on_supply_and_demand(
    fixture_artefacts,
):
    rows = fixture_artefacts.baseline_features
    models = fixture_artefacts.fold_models
    model_alone, _ = log_contributions(rows, models)
    rows = rows.with_columns(naive_usd=pl.Series(np.geomspace(0.01, 1e9, rows.height)))
    form = ModelForm(naive_weight=0.3)

    contributions, log_potential = log_contributions(rows, models, form=form)

    assert contributions.sum(axis=1) == pytest.approx(log_potential, abs=1e-4)
    assert np.exp(log_potential) == pytest.approx(export_potential(rows, models, form=form).to_numpy(), rel=1e-4)
    others = [i for i, f in enumerate(FEATURES) if f not in ("log_supply_loo", "log_demand_loo")]
    assert contributions[:, others] == pytest.approx(0.7 * model_alone[:, others], abs=1e-5)
    naive_share = 0.3 * np.log(np.maximum(rows["naive_usd"].to_numpy(), 1.0))
    on_supply_and_demand = contributions[:, [FEATURES.index("log_supply_loo"), FEATURES.index("log_demand_loo")]].sum(1)
    expected = 0.7 * model_alone[:, [FEATURES.index("log_supply_loo"), FEATURES.index("log_demand_loo")]].sum(1)
    assert on_supply_and_demand == pytest.approx(expected + naive_share, abs=1e-4)
