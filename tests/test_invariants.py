"""What the spec promises of every artefact, checked on the fixture artefact and, when present, the real one.

A rule here holds for any artefact a build writes, so a build that breaks one fails here before
it is served. The real-artefact cases skip without data (point `EXPORTSPREDICTOR_ARTEFACTS_DIR`
at the real artefact root, as `scripts/wt.ps1` does).
"""

import polars as pl
import pytest

from exportspredictor import artefacts as store
from exportspredictor import rankings
from exportspredictor.features import MONOTONE_CONSTRAINTS
from exportspredictor.paths import artefacts_dir

FLOAT_NOISE_USD = 1e-3


def _real_artefacts() -> store.Artefacts:
    try:
        bundle = store.load_latest(artefacts_dir())
    except store.ArtefactsNotFound:
        pytest.skip("No real artefacts under EXPORTSPREDICTOR_ARTEFACTS_DIR")
    except store.UnreadableArtefacts as error:
        pytest.skip(f"The newest real artefacts are not artefact format {store.ARTEFACT_FORMAT}: {error}")
    if bundle.metadata.get("source", "fixture") == "fixture":
        pytest.skip("The artefacts are fixture artefacts")
    return bundle


@pytest.fixture(scope="module")
def real_artefacts():
    return _real_artefacts()


@pytest.fixture(scope="module", params=["fixture", "real"])
def artefacts(request):
    if request.param == "fixture":
        return request.getfixturevalue("fixture_artefacts")
    return request.getfixturevalue("real_artefacts")


def test_adding_a_trade_agreement_never_lowers_export_potential(artefacts):
    # Spec #39: "a Trade Agreement Scenario yields non-negative Scenario Effects". The monotone
    # constraints guarantee it only on known values, so an unknown tariff stays unknown (ADR 0006).
    values = artefacts.baseline["blocked_opportunity_value_usd"]
    negative = artefacts.baseline.filter(pl.col("blocked_opportunity_value_usd") < -FLOAT_NOISE_USD)

    assert values.null_count() == 0
    assert negative.height == 0, f"{negative.height} pairs lose Export Potential when an agreement is added:\n{negative.head()}"


def test_the_blocked_summary_totals_the_blocked_opportunities_only(artefacts):
    ranked = rankings.ranking(artefacts, rank_by=rankings.BLOCKED.name)
    rows = rankings.rank_opportunities(artefacts, rank_by=rankings.BLOCKED.name)
    blocked = rows.filter("blocked_opportunity")

    assert ranked.summary["blocked_opportunities"] == blocked.height
    assert ranked.summary["blocked_opportunity_value_usd"] == pytest.approx(blocked["blocked_opportunity_value_usd"].sum())


def test_every_potential_range_is_an_ordered_non_negative_interval(artefacts):
    baseline = artefacts.baseline
    disordered = baseline.filter(
        (pl.col("potential_range_low_usd") < 0) | (pl.col("potential_range_low_usd") > pl.col("potential_range_high_usd"))
    )

    assert baseline["potential_range_low_usd"].null_count() == 0
    assert disordered.height == 0, disordered.head()


def test_the_fixture_takes_every_missing_value_branch_the_real_data_takes(fixture_artefacts, real_artefacts):
    # XGBoost's monotone constraints do not bind the missing-value branch, so a constrained
    # feature that is ever missing in real data must be missing somewhere in the fixture too,
    # or no fixture test exercises that branch.
    real = real_artefacts.baseline_features
    fixture = fixture_artefacts.baseline_features
    untested = [
        feature
        for feature in MONOTONE_CONSTRAINTS
        if real[feature].null_count() > 0 and fixture[feature].null_count() == 0
    ]

    assert untested == [], f"Missing in real data but never in the fixture: {untested}"
