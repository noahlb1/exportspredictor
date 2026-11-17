"""API contract for Scenarios and Blocked Opportunity rankings, on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with
India; Japan and Canada are CPTPP members, and the UK is in no Trading Bloc.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app

FIXTURE_HS6 = "220830"


@pytest.fixture(scope="module")
def client(fixture_artefacts_dir):
    with TestClient(create_app(fixture_artefacts_dir)) as client:
        yield client


def test_presets_lists_the_four_trading_bloc_presets(client):
    body = client.get("/scenarios/presets").json()

    assert [preset["name"] for preset in body] == ["eu_rejoin", "eu_customs_union", "cptpp", "leave"]
    assert {"name", "title", "description"} == set(body[0])


def test_a_custom_scenario_adding_an_agreement_returns_non_negative_effects(client):
    response = client.post("/scenarios", json={"add_agreements": ["IND"], "min_effect_share": 0, "min_effect_usd": 0})

    assert response.status_code == 200
    body = response.json()
    assert body["changes_nothing"] is False
    assert {row["iso3"] for row in body["markets"]} == {"IND"}
    assert len(body["top_pairs"]) == 5
    assert all(row["effect_usd"] >= 0 for row in body["top_pairs"])
    assert body["total"]["effect_usd"] == pytest.approx(sum(row["effect_usd"] for row in body["top_pairs"]))
    assert body["total"]["effect_usd"] > 0
    assert "associates" in body["summary"]


def test_top_pairs_are_sorted_by_scenario_effect(client):
    body = client.post("/scenarios", json={"add_agreements": ["IND"]}).json()

    effects = [row["effect_usd"] for row in body["top_pairs"]]
    assert effects == sorted(effects, reverse=True)
    assert set(body["top_pairs"][0]) == {
        "hs6",
        "description",
        "iso3",
        "market",
        "baseline_potential_usd",
        "scenario_potential_usd",
        "effect_usd",
        "effect_share",
        "blocked_opportunity",
    }


def test_a_preset_scenario_succeeds(client):
    response = client.post("/scenarios", json={"preset": "cptpp"})

    assert response.status_code == 200
    body = response.json()
    assert body["scenario"]["preset"] == "cptpp"
    assert body["changes_nothing"] is False
    assert {row["iso3"] for row in body["markets"]} == {"JPN", "CAN"}


def test_blocked_opportunity_thresholds_are_respected(client):
    loose = client.post("/scenarios", json={"add_agreements": ["IND"], "min_effect_share": 0, "min_effect_usd": 0}).json()
    strict = client.post(
        "/scenarios", json={"add_agreements": ["IND"], "min_effect_share": 0.2, "min_effect_usd": 1e15}
    ).json()
    share = 0.3
    middle = client.post("/scenarios", json={"add_agreements": ["IND"], "min_effect_share": share, "min_effect_usd": 0}).json()

    assert len(loose["blocked_opportunities"]) == 5
    assert strict["blocked_opportunities"] == []
    assert all(row["effect_share"] >= share for row in middle["blocked_opportunities"])
    expected = {(r["hs6"], r["iso3"]) for r in loose["top_pairs"] if r["effect_share"] >= share}
    assert {(r["hs6"], r["iso3"]) for r in middle["blocked_opportunities"]} == expected
    assert middle["thresholds"] == {"min_effect_share": share, "min_effect_usd": 0}


def test_default_thresholds_are_20_percent_and_one_million_dollars(client):
    body = client.post("/scenarios", json={"add_agreements": ["IND"]}).json()

    assert body["thresholds"] == {"min_effect_share": 0.2, "min_effect_usd": 1_000_000}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"preset": "cptpp", "add_agreements": ["IND"]},
        {"preset": "not-a-preset"},
        {"add_agreements": ["XXX"]},
        {"add_agreements": ["IND"], "remove_agreements": ["IND"]},
        {"add_agreements": ["IND"], "min_effect_share": -0.1},
    ],
)
def test_an_invalid_scenario_returns_422(client, payload):
    response = client.post("/scenarios", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("payload", [{"add_agreements": ["JPN"]}, {"preset": "leave"}])
def test_a_scenario_that_changes_nothing_is_reported_as_such(client, payload):
    body = client.post("/scenarios", json=payload).json()

    assert body["changes_nothing"] is True
    assert body["total"]["effect_usd"] == 0
    assert body["markets"] == body["top_pairs"] == body["blocked_opportunities"] == []
    assert "changes nothing" in body["summary"]


def test_by_product_ranks_by_export_gap_by_default(client):
    rows = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()["opportunities"]

    gaps = [row["export_gap_usd"] for row in rows]
    assert gaps == sorted(gaps, reverse=True)


def test_by_product_can_rank_by_blocked_opportunity_value(client):
    body = client.get(f"/opportunities/by-product/{FIXTURE_HS6}", params={"rank_by": "blocked"}).json()

    rows = body["opportunities"]
    assert body["rank_by"] == "blocked"
    assert [row["iso3"] for row in rows][0] == "IND"  # the only Market without a Trade Agreement
    values = [row["blocked_effect_usd"] for row in rows]
    assert values == sorted(values, reverse=True)
    assert {row["iso3"]: row["blocked_effect_usd"] for row in rows}["JPN"] == 0


def test_blocked_value_matches_a_scenario_adding_that_agreement(client):
    ranked = client.get(f"/opportunities/by-product/{FIXTURE_HS6}", params={"rank_by": "blocked"}).json()
    scenario = client.post("/scenarios", json={"add_agreements": ["IND"]}).json()

    india = next(row for row in ranked["opportunities"] if row["iso3"] == "IND")
    pair = next(row for row in scenario["top_pairs"] if row["hs6"] == FIXTURE_HS6)
    assert india["blocked_effect_usd"] == pytest.approx(pair["effect_usd"])


def test_an_unknown_rank_by_returns_422(client):
    response = client.get(f"/opportunities/by-product/{FIXTURE_HS6}", params={"rank_by": "size"})

    assert response.status_code == 422
