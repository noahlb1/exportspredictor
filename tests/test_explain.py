"""Package seam: "Why?" explanations of one UK Product–Market pair from its own fold model."""

import math
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from exportspredictor.explain import UnknownPair, explain_pair
from exportspredictor.features import FEATURES


def _pairs(artefacts):
    return artefacts.baseline.select("hs6", "iso3").iter_rows()


def test_contributions_plus_base_value_reconstruct_the_baseline_export_potential(fixture_artefacts):
    for hs6, iso3 in _pairs(fixture_artefacts):
        explanation = explain_pair(fixture_artefacts, hs6, iso3)
        baseline = fixture_artefacts.baseline.filter((pl.col("hs6") == hs6) & (pl.col("iso3") == iso3))

        log_potential = explanation.base_value + sum(f.contribution for f in explanation.factors)
        assert log_potential == pytest.approx(explanation.log_export_potential, abs=1e-4)
        assert math.exp(log_potential) == pytest.approx(baseline["baseline_export_potential_usd"].item(), rel=1e-3)


def test_explanation_uses_the_fold_model_that_scored_the_pair(fixture_artefacts):
    for hs6, iso3 in _pairs(fixture_artefacts):
        explanation = explain_pair(fixture_artefacts, hs6, iso3)
        row = fixture_artefacts.baseline_features.filter((pl.col("hs6") == hs6) & (pl.col("iso3") == iso3))

        assert explanation.fold == row["fold"].item()
        assert fixture_artefacts.fold_models[explanation.fold].attr("held_out_fold") == str(explanation.fold)


def test_another_fold_model_would_not_reconstruct_the_baseline(fixture_artefacts):
    # Guards the test above: fold models disagree, so matching the Baseline pins the model.
    disagreements = 0
    for hs6, iso3 in _pairs(fixture_artefacts):
        explanation = explain_pair(fixture_artefacts, hs6, iso3)
        row = fixture_artefacts.baseline_features.filter((pl.col("hs6") == hs6) & (pl.col("iso3") == iso3))
        data = row.select(FEATURES).to_numpy().astype(np.float32)
        other = fixture_artefacts.fold_models[(explanation.fold + 1) % len(fixture_artefacts.fold_models)]
        disagreements += not math.isclose(float(other.inplace_predict(data)[0]), math.exp(explanation.log_export_potential), rel_tol=1e-3)
    assert disagreements > 0


def test_every_feature_appears_once_with_a_plain_english_label(fixture_artefacts):
    explanation = explain_pair(fixture_artefacts, "220830", "JPN")

    assert sorted(f.feature for f in explanation.factors) == sorted(FEATURES)
    for factor in explanation.factors:
        assert factor.label and factor.feature not in factor.label
        assert "_" not in factor.label and "loo" not in factor.label.lower()
        assert factor.value_text


def test_labels_name_the_market(fixture_artefacts):
    explanation = explain_pair(fixture_artefacts, "220830", "JPN")
    labels = {f.feature: f.label for f in explanation.factors}

    assert "Japan" in labels["log_demand_loo"]
    assert "Japan" in labels["log_distance"]
    assert "Japan" in labels["trade_agreement"]


def test_factors_are_ordered_by_size_of_contribution(fixture_artefacts):
    sizes = [abs(f.contribution) for f in explain_pair(fixture_artefacts, "220830", "JPN").factors]

    assert sizes == sorted(sizes, reverse=True)


def test_history_holds_the_pairs_actual_exports_by_year(fixture_artefacts):
    explanation = explain_pair(fixture_artefacts, "220830", "JPN")
    expected = fixture_artefacts.history.filter((pl.col("hs6") == "220830") & (pl.col("iso3") == "JPN"))

    assert [(h.year, h.actual_exports_usd, h.export_gap_usd) for h in explanation.history] == list(
        expected.select("year", "actual_exports_usd", "export_gap_usd").iter_rows()
    )
    assert expected.height > 0


@pytest.mark.parametrize(("hs6", "iso3"), [("999999", "JPN"), ("220830", "ZZZ")])
def test_unknown_pair_raises(fixture_artefacts, hs6, iso3):
    with pytest.raises(UnknownPair):
        explain_pair(fixture_artefacts, hs6, iso3)


def test_global_importance_is_the_mean_absolute_contribution_over_pairs(fixture_artefacts):
    from exportspredictor.explain import global_importance

    importance = global_importance(fixture_artefacts, sample=None)

    pairs = list(_pairs(fixture_artefacts))
    expected = {feature: 0.0 for feature in FEATURES}
    for hs6, iso3 in pairs:
        for factor in explain_pair(fixture_artefacts, hs6, iso3).factors:
            expected[factor.feature] += abs(factor.contribution) / len(pairs)
    got = dict(importance.select("feature", "mean_abs_contribution").iter_rows())
    assert got == pytest.approx(expected, abs=1e-5)
    assert importance["mean_abs_contribution"].to_list() == sorted(got.values(), reverse=True)
    assert importance["pairs"].unique().to_list() == [len(pairs)]


def test_the_tariff_caveat_appears_only_for_a_known_carried_forward_tariff(fixture_artefacts):
    medicaments_india = ("300490", "IND")
    observed = replace(fixture_artefacts, baseline=fixture_artefacts.baseline.with_columns(tariff_imputed=pl.lit(False)))
    unknown = replace(
        fixture_artefacts,
        baseline_features=fixture_artefacts.baseline_features.with_columns(
            log_applied_tariff=pl.lit(None, pl.Float32)
        ),
    )

    assert explain_pair(fixture_artefacts, *medicaments_india).tariff_caveat
    assert explain_pair(observed, *medicaments_india).tariff_caveat is None
    assert explain_pair(unknown, *medicaments_india).tariff_caveat is None
