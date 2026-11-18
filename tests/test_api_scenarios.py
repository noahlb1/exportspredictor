"""API contract for Scenarios and Blocked Opportunity rankings, on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with
India; Japan and Canada are CPTPP members, and the UK is in no Trading Bloc.
"""

import re

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
        "baseline_tariff_pct",
        "scenario_tariff_pct",
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


def _blocked(client, **params) -> dict:
    response = client.get("/opportunities/blocked", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _blocked_summary(rows: list[dict]) -> dict:
    return {
        "pairs": len(rows),
        "positive_export_gap_usd": pytest.approx(sum(max(r["export_gap_usd"], 0) for r in rows)),
        "blocked_opportunities": sum(r["blocked_opportunity"] for r in rows),
        "blocked_effect_usd": pytest.approx(sum(r["blocked_effect_usd"] for r in rows if r["blocked_opportunity"])),
    }


def test_blocked_ranks_every_uk_pair_by_blocked_opportunity_value_with_a_summary(client):
    body = _blocked(client)

    rows = body["opportunities"]
    values = [row["blocked_effect_usd"] for row in rows]
    assert len(rows) == 20
    assert values == sorted(values, reverse=True)
    assert values[0] > 0
    assert rows[0]["iso3"] in {"IND", "HKG"}  # the fixture Markets without a UK Trade Agreement
    assert body["rank_by"] == "blocked"
    assert body["gap_years"] == [2022, 2024]
    assert body["summary"] == _blocked_summary(rows)
    assert body["order"]["title"] == "Blocked Opportunity value"
    assert "associates" in body["order"]["description"]


def test_blocked_rows_have_the_leaderboard_row_fields(client):
    blocked = _blocked(client)["opportunities"][0]
    top = client.get("/opportunities/top").json()["opportunities"][0]

    assert set(blocked) == set(top)


def test_blocked_value_is_zero_where_a_trade_agreement_is_in_force(client):
    rows = _blocked(client)["opportunities"]

    with_agreement = [row for row in rows if row["trade_agreement"]]
    assert {row["iso3"] for row in with_agreement} == {"JPN", "CAN"}
    assert {row["blocked_effect_usd"] for row in with_agreement} == {0}
    assert not any(row["blocked_opportunity"] for row in with_agreement)
    assert all(row["blocked_effect_usd"] >= 0 for row in rows)  # monotone: adding an agreement never lowers


@pytest.mark.parametrize(
    ("params", "keep"),
    [
        ({"sector": "22"}, lambda row: row["sector"] == "22"),
        ({"region": "South Asia"}, lambda row: row["region"] == "South Asia"),
        ({"market": "IND"}, lambda row: row["iso3"] == "IND"),
        ({"opportunity_type": "Expand"}, lambda row: row["opportunity_type"] == "Expand"),
        ({"min_gap_usd": 5000}, lambda row: row["export_gap_usd"] >= 5000),
        ({"trade_agreement": False}, lambda row: not row["trade_agreement"]),
        ({"hide_flagged": True}, lambda row: not row["flagged_product"]),
        ({"hide_hubs": True}, lambda row: not row["reexport_hub"]),
    ],
    ids=lambda value: "-".join(f"{k}={v}" for k, v in value.items()) if isinstance(value, dict) else "",
)
def test_every_blocked_filter_keeps_exactly_the_matching_rows_and_the_summary_follows(client, params, keep):
    everything = _blocked(client)["opportunities"]
    body = _blocked(client, **params)

    expected = [row for row in everything if keep(row)]
    assert body["opportunities"] == expected
    assert body["summary"] == _blocked_summary(expected)


def test_blocked_limit_keeps_the_top_but_not_the_summary(client):
    everything = _blocked(client)["opportunities"]
    top = _blocked(client, limit=2)

    assert top["opportunities"] == everything[:2]
    assert top["summary"] == _blocked_summary(everything)


def test_blocked_rejects_another_order_an_unknown_market_or_filter(client):
    assert client.get("/opportunities/blocked", params={"rank_by": "gap"}).status_code == 422
    assert client.get("/opportunities/blocked", params={"market": "XYZ"}).status_code == 404
    assert client.get("/opportunities/blocked", params={"colour": "red"}).status_code == 422


def test_blocked_value_matches_a_scenario_adding_that_agreement(client):
    ranked = _blocked(client, market="IND", sector="22")["opportunities"]
    scenario = client.post("/scenarios", json={"add_agreements": ["IND"]}).json()

    [india] = ranked
    pair = next(row for row in scenario["top_pairs"] if row["hs6"] == FIXTURE_HS6)
    assert india["blocked_effect_usd"] == pytest.approx(pair["effect_usd"])


@pytest.mark.parametrize("path", [f"/opportunities/by-product/{FIXTURE_HS6}", "/opportunities/by-market/IND"])
def test_ranking_views_no_longer_rank_by_blocked_opportunity_value(client, path):
    assert client.get(path, params={"rank_by": "blocked"}).status_code == 422
    assert client.get("/opportunities/top", params={"rank_by": "blocked"}).status_code == 422


def test_an_unknown_rank_by_returns_422(client):
    response = client.get(f"/opportunities/by-product/{FIXTURE_HS6}", params={"rank_by": "size"})

    assert response.status_code == 422


def test_by_market_unknown_rank_by_returns_422(client):
    assert client.get("/opportunities/by-market/JPN", params={"rank_by": "size"}).status_code == 422


def test_scenario_summary_states_the_effect_in_usd_only(client):
    body = client.post("/scenarios", json={"add_agreements": ["IND"], "min_effect_share": 0, "min_effect_usd": 0}).json()

    assert re.search(r"change of [+-]\$[0-9,.]+(bn|m|k)? ", body["summary"])
    assert "£" not in body["summary"]


def test_scenario_top_pairs_can_be_asked_for_one_market(client):
    everywhere = client.post("/scenarios", json={"preset": "cptpp", "limit": 1}).json()
    canada = client.post("/scenarios", json={"preset": "cptpp", "limit": 1, "market": "CAN"}).json()
    japan = client.post("/scenarios", json={"preset": "cptpp", "limit": 1, "market": "JPN"}).json()

    # With limit 1 the global top pair is in one Market only; each Market still gets its own.
    assert len(everywhere["top_pairs"]) == 1
    assert [row["iso3"] for row in canada["top_pairs"]] == ["CAN"]
    assert [row["iso3"] for row in japan["top_pairs"]] == ["JPN"]
    assert canada["markets"] == everywhere["markets"]  # the per-Market totals are unchanged
    assert canada["market"] == "CAN"


def test_scenario_drill_down_market_must_be_known(client):
    assert client.post("/scenarios", json={"preset": "cptpp", "market": "ZZZ"}).status_code == 422


def test_default_thresholds_are_served(client):
    assert client.get("/scenarios/thresholds").json() == {"min_effect_share": 0.2, "min_effect_usd": 1_000_000}


def test_rank_orders_are_served_with_titles(client):
    body = client.get("/opportunities/rank-by").json()

    # Blocked Opportunity value is the Blocked Opportunities page's order, not a ranking toggle.
    assert [order["name"] for order in body] == ["gap", "relative"]
    assert [order["title"] for order in body] == ["Export Gap", "Relative Export Gap"]


# --- Tariffs in Scenarios (ADR 0006) ------------------------------------------------------


def _scenario(client, payload: dict) -> dict:
    response = client.post("/scenarios", json={**payload, "min_effect_share": 0, "min_effect_usd": 0, "limit": 1000})
    assert response.status_code == 200, response.text
    return response.json()


def test_adding_a_trade_agreement_removes_the_tariff_and_never_lowers_export_potential(client):
    body = _scenario(client, {"add_agreements": ["IND"]})

    assert len(body["top_pairs"]) == 5
    assert all(row["effect_usd"] >= 0 for row in body["top_pairs"])
    assert all(row["scenario_tariff_pct"] == 0 for row in body["top_pairs"])
    assert any(row["baseline_tariff_pct"] > 0 for row in body["top_pairs"])
    assert body["clipped_pairs"] == 0


def test_removing_a_trade_agreement_restores_mfn_and_never_raises_export_potential(client):
    body = _scenario(client, {"remove_agreements": ["JPN", "CAN"]})

    assert {row["iso3"] for row in body["top_pairs"]} == {"JPN", "CAN"}
    assert all(row["effect_usd"] <= 0 for row in body["top_pairs"])
    assert all(row["scenario_tariff_pct"] >= row["baseline_tariff_pct"] for row in body["top_pairs"])
    assert any(row["scenario_tariff_pct"] > 0 for row in body["top_pairs"])


def test_a_tariff_hike_never_raises_export_potential(client):
    body = _scenario(client, {"tariff": {"markets": ["IND", "JPN", "HKG"], "rate_pct": 150}})

    assert body["scenario"]["tariff"] == {"markets": ["IND", "JPN", "HKG"], "rate_pct": 150, "sectors": None}
    assert body["top_pairs"]
    assert all(row["scenario_tariff_pct"] == pytest.approx(150) for row in body["top_pairs"])
    assert all(row["effect_usd"] <= 0 for row in body["top_pairs"])
    assert body["clipped_pairs"] == 0


def test_a_tariff_scenario_can_be_limited_to_sectors(client):
    body = _scenario(client, {"tariff": {"markets": ["JPN"], "rate_pct": 25, "sectors": ["22", "87"]}})

    assert {(row["hs6"], row["iso3"]) for row in body["top_pairs"]} == {("220830", "JPN"), ("870323", "JPN")}


def test_a_hike_beyond_the_training_range_reports_clipped_pairs(client):
    body = _scenario(client, {"tariff": {"markets": ["JPN"], "rate_pct": 1000}})

    assert body["clipped_pairs"] == 5
    # Clipped to the training range's top, India's 150% on whisky.
    assert all(row["scenario_tariff_pct"] == pytest.approx(150, rel=1e-5) for row in body["top_pairs"])
    assert all(row["effect_usd"] <= 0 for row in body["top_pairs"])


@pytest.mark.parametrize(
    "payload",
    [
        {"tariff": {"markets": ["JPN"], "rate_pct": -5}},
        {"tariff": {"markets": ["JPN"], "rate_pct": 5, "sectors": ["99"]}},
        {"tariff": {"markets": ["JPN"], "rate_pct": 5, "sectors": ["wine"]}},
        {"tariff": {"markets": [], "rate_pct": 5}},
        {"tariff": {"markets": ["XXX"], "rate_pct": 5}},
        {"tariff": {"markets": ["JPN"]}},
        {"tariff": {"markets": ["JPN"], "rate_pct": 5}, "preset": "cptpp"},
    ],
)
def test_an_invalid_tariff_scenario_returns_422(client, payload):
    response = client.post("/scenarios", json=payload)

    assert response.status_code == 422
