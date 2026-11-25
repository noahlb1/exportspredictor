"""Scenarios on the real artefacts, when they are present (skipped otherwise).

Point `EXPORTSPREDICTOR_ARTEFACTS_DIR` at the real artefact root to run these.
"""

import time

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.paths import artefacts_dir


@pytest.fixture(scope="module")
def client():
    try:
        bundle = store.load_latest(artefacts_dir())
    except store.ArtefactsNotFound:
        pytest.skip("No real artefacts under EXPORTSPREDICTOR_ARTEFACTS_DIR")
    except store.UnreadableArtefacts as error:
        pytest.skip(f"The newest real artefacts are not artefact format {store.ARTEFACT_FORMAT}: {error}")
    if bundle.metadata.get("source", "fixture") == "fixture":
        pytest.skip("The artefacts are fixture artefacts")
    with TestClient(create_app(artefacts_dir())) as client:
        yield client


def test_the_cptpp_preset_has_a_non_zero_effect_because_the_2024_baseline_has_no_uk_membership(client):
    body = client.post("/scenarios", json={"preset": "cptpp"}).json()

    assert body["baseline_year"] == 2024
    assert body["changes_nothing"] is False
    assert body["total"]["effect_usd"] > 0


def test_a_bloc_scenario_returns_within_a_few_seconds_on_cpu(client):
    started = time.perf_counter()
    response = client.post("/scenarios", json={"preset": "eu_rejoin"})
    seconds = time.perf_counter() - started

    assert response.status_code == 200
    assert sum(market["pairs"] for market in response.json()["markets"]) > 100_000
    assert seconds < 5


def test_no_pair_has_a_negative_blocked_opportunity_value_and_the_summary_totals_blocked_ones(client):
    # Adding a Trade Agreement never lowers Export Potential: an unknown tariff stays unknown (ADR 0006).
    values = client.app.state.artefacts.baseline["blocked_opportunity_value_usd"]
    summary = client.get("/opportunities/blocked", params={"limit": 1}).json()["summary"]

    assert values.min() >= -1e-3  # float noise only
    assert 0 < summary["blocked_opportunities"] < values.len()
    # The total is over the Blocked Opportunities only, well under the sum over every pair.
    assert 0 < summary["blocked_opportunity_value_usd"] < values.sum()


# The artefact the write-up reports: 20261122T133316Z rebuilt with `rereach` after Market
# Reach's sent was nested in imported (ADR 0010, Amendment), which took the Falkland Islands off the list. A newer
# version moves these figures, so they are pinned to this one.
REPORTED_VERSION = "20261122T223312Z"


def test_the_reported_artefact_has_its_blocked_opportunity_figures_with_room_to_grow(client):
    if client.app.state.artefacts.version != REPORTED_VERSION:
        pytest.skip(f"The served artefacts are not the reported version {REPORTED_VERSION}")
    body = client.get("/opportunities/blocked", params={"limit": 6000}).json()

    assert body["summary"]["blocked_opportunities"] == 2_128
    assert body["summary"]["blocked_opportunity_value_usd"] == pytest.approx(18.52e9, rel=0.001)
    assert len(body["markets"]) == 16
    assert "FLK" not in {market["iso3"] for market in body["markets"]}
    rows = body["opportunities"]
    # Room to grow: every listed pair's Baseline-year Actual Exports are below Export Potential with an agreement.
    assert all(row["baseline_actual_exports_usd"] < row["potential_with_agreement_usd"] for row in rows)
    tariff_part = sum(row["tariff_part_usd"] for row in rows)
    assert tariff_part == pytest.approx(5.8e9, rel=0.01)
    for row in rows:
        assert row["tariff_part_usd"] + row["agreement_part_usd"] == pytest.approx(row["blocked_opportunity_value_usd"])
        if row["barrier"] in ("No agreement, tariff 0%", "No agreement, tariff unknown"):
            assert row["tariff_part_usd"] == 0
