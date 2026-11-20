"""API contract for Scenarios and Blocked Opportunity rankings, on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with
India; Japan and Canada are CPTPP members, and the UK is in no Trading Bloc.
"""

import re
from dataclasses import replace

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.caveats import TIERS

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
        "baseline_actual_exports_usd",
        "blocked_opportunity",
        *(column for tier in TIERS for column in tier.columns),  # each Caveat Tier's, as on ranking rows
    }


def test_scenario_pairs_carry_baseline_year_actual_exports_from_the_yearly_history(client, fixture_artefacts):
    history = fixture_artefacts.history
    year = fixture_artefacts.baseline_year
    exports = {
        (hs6, iso3): value
        for hs6, iso3, value in history.filter(history["year"] == year)
        .select("hs6", "iso3", "actual_exports_usd")
        .iter_rows()
    }

    body = _scenario(client, {"add_agreements": ["IND"], "remove_agreements": ["JPN"]})

    pairs = [*body["top_pairs"], *body["blocked_opportunities"]]
    assert pairs
    for row in pairs:
        assert row["baseline_actual_exports_usd"] == pytest.approx(exports.get((row["hs6"], row["iso3"]), 0.0))
    assert any(row["baseline_actual_exports_usd"] == 0 for row in pairs)  # a pair with no Flow that year shows 0
    assert any(row["baseline_actual_exports_usd"] > 0 for row in pairs)


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


# --- The Blocked Opportunities read (#109), on the Blocked world (`conftest.blocked_artefacts`) ------------------

WHISKY, JEWELLERY, CARS, CRUDE_OIL = "220830", "711319", "870323", "270900"
# The Blocked world's Blocked Opportunities (see `conftest.blocked_artefacts`): jewellery is the Flagged Product and
# Hong Kong the Re-export Hub.
BLOCKED_PAIRS = {(WHISKY, "IND"), (JEWELLERY, "IND"), (JEWELLERY, "HKG"), (CARS, "HKG")}


@pytest.fixture(scope="module")
def blocked_client(blocked_artefacts_dir):
    with TestClient(create_app(blocked_artefacts_dir)) as client:
        yield client


def _blocked(client, **params) -> dict:
    response = client.get("/opportunities/blocked", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _totals(rows: list[dict]) -> tuple[dict, list[dict]]:
    """The summary and per-Market totals of `rows`, summed here."""
    summary = {
        "blocked_opportunities": len(rows),
        "blocked_opportunity_value_usd": pytest.approx(sum(row["blocked_opportunity_value_usd"] for row in rows)),
    }
    markets: dict[str, dict] = {}
    for row in rows:
        market = markets.setdefault(
            row["iso3"],
            {"iso3": row["iso3"], "market": row["market"], "blocked_opportunities": 0, "values": []},
        )
        market["blocked_opportunities"] += 1
        market["values"].append(row["blocked_opportunity_value_usd"])
    totals = [
        {
            "iso3": market["iso3"],
            "market": market["market"],
            "blocked_opportunities": market["blocked_opportunities"],
            "blocked_opportunity_value_usd": pytest.approx(sum(market["values"])),
        }
        for market in sorted(markets.values(), key=lambda market: -sum(market["values"]))
    ]
    return summary, totals


def test_the_blocked_read_returns_only_blocked_opportunities_ranked_by_value(blocked_client, client):
    body = _blocked(blocked_client)

    rows = body["opportunities"]
    assert {(row["hs6"], row["iso3"]) for row in rows} == BLOCKED_PAIRS
    assert all(row["blocked_opportunity"] for row in rows)
    values = [row["blocked_opportunity_value_usd"] for row in rows]
    assert values == sorted(values, reverse=True)
    assert (rows[0]["hs6"], rows[0]["iso3"]) == (WHISKY, "IND")
    assert body["order"]["title"] == "Blocked Opportunity value"
    assert "associates" in body["order"]["description"]
    # The unscaled fixture's values are all under the $1M threshold: no Blocked Opportunity at all.
    empty = _blocked(client)
    assert empty["opportunities"] == [] and empty["markets"] == []
    assert empty["summary"] == {"blocked_opportunities": 0, "blocked_opportunity_value_usd": 0}


def test_the_summary_and_per_market_totals_are_sums_over_the_returned_rows(blocked_client):
    body = _blocked(blocked_client)

    summary, markets = _totals(body["opportunities"])
    assert body["summary"] == summary
    assert body["markets"] == markets
    # India's two (whisky and jewellery) total more than Hong Kong's two (jewellery and cars).
    assert [(market["iso3"], market["blocked_opportunities"]) for market in body["markets"]] == [("IND", 2), ("HKG", 2)]


@pytest.mark.parametrize(
    ("params", "keep"),
    [
        ({"product": JEWELLERY}, lambda row: row["hs6"] == JEWELLERY),
        ({"sector": "87"}, lambda row: row["sector"] == "87"),
        ({"region": "South Asia"}, lambda row: row["region"] == "South Asia"),
        ({"min_value_usd": 5e6}, lambda row: row["blocked_opportunity_value_usd"] >= 5e6),
        ({"hide_flagged": True}, lambda row: not row["flagged_product"]),
        ({"hide_hubs": True}, lambda row: not row["reexport_hub"]),
    ],
    ids=lambda value: "-".join(f"{k}={v}" for k, v in value.items()) if isinstance(value, dict) else "",
)
def test_each_filter_narrows_the_rows_summary_and_totals_together(blocked_client, params, keep):
    everything = _blocked(blocked_client)["opportunities"]
    body = _blocked(blocked_client, **params)

    expected = [row for row in everything if keep(row)]
    assert 0 < len(expected) < len(everything)
    assert body["opportunities"] == expected
    summary, markets = _totals(expected)
    assert body["summary"] == summary
    assert body["markets"] == markets


def test_a_limit_keeps_the_top_rows_but_not_the_summary_or_totals(blocked_client):
    everything = _blocked(blocked_client)
    top = _blocked(blocked_client, limit=2)

    assert top["opportunities"] == everything["opportunities"][:2]
    assert top["summary"] == everything["summary"] and top["markets"] == everything["markets"]


def test_potential_with_agreement_is_baseline_export_potential_plus_the_value(blocked_client):
    rows = _blocked(blocked_client)["opportunities"]
    below = [row for row in rows if row["baseline_actual_exports_usd"] <= row["baseline_export_potential_usd"]]

    assert below  # whisky in India: $38.1m exported against $40.6m of Export Potential, 2024
    for row in below:
        baseline = blocked_client.get(f"/explain/{row['hs6']}/{row['iso3']}").json()["baseline_export_potential_usd"]
        assert row["baseline_export_potential_usd"] == pytest.approx(baseline)
        assert row["potential_with_agreement_usd"] == pytest.approx(baseline + row["blocked_opportunity_value_usd"])
        assert row["blocked_opportunity_uplift"] is None


def test_where_actual_exports_exceed_export_potential_the_value_is_an_uplift_with_no_dollar_level(blocked_client):
    # Spec #113 "Blocked Opportunity figures": no "with a Trade Agreement" dollar level below what the UK already sells.
    rows = {(row["hs6"], row["iso3"]): row for row in _blocked(blocked_client)["opportunities"]}
    cars = rows[(CARS, "HKG")]

    assert cars["baseline_actual_exports_usd"] > cars["baseline_export_potential_usd"]
    assert cars["potential_with_agreement_usd"] is None
    assert cars["blocked_opportunity_uplift"] == pytest.approx(
        cars["blocked_opportunity_value_usd"] / cars["baseline_export_potential_usd"]
    )
    # Its value still counts in the ranking, the summary and its Market's total.
    assert cars["blocked_opportunity_value_usd"] > 0


def test_baseline_year_actual_exports_match_the_history_with_0_where_there_is_no_flow(blocked_artefacts, tmp_path):
    # Jewellery in Hong Kong loses its Baseline-year history row, as a pair with no Flow that year can.
    history = blocked_artefacts.history
    year = blocked_artefacts.baseline_year
    no_flow = (pl.col("hs6") == JEWELLERY) & (pl.col("iso3") == "HKG") & (pl.col("year") == year)
    store.save(replace(blocked_artefacts, history=history.filter(~no_flow)), tmp_path)
    exports = {
        (row["hs6"], row["iso3"]): row["actual_exports_usd"]
        for row in history.filter(pl.col("year") == year).iter_rows(named=True)
    }

    with TestClient(create_app(tmp_path)) as client:
        rows = _blocked(client)["opportunities"]

    shown = {(row["hs6"], row["iso3"]): row["baseline_actual_exports_usd"] for row in rows}
    assert shown[(JEWELLERY, "HKG")] == 0
    for pair in BLOCKED_PAIRS - {(JEWELLERY, "HKG")}:
        assert shown[pair] == pytest.approx(exports[pair]) and shown[pair] > 0
    # Baseline-year figures, not the Gap Period means the Leaderboard shows.
    assert any(row["baseline_actual_exports_usd"] != pytest.approx(row["actual_exports_usd"]) for row in rows)


def test_the_blocked_read_refuses_an_order_a_market_or_an_unknown_filter(blocked_client):
    for params in ({"rank_by": "gap"}, {"market": "IND"}, {"opportunity_type": "Expand"}, {"colour": "red"}):
        assert blocked_client.get("/opportunities/blocked", params=params).status_code == 422, params


def test_the_blocked_read_answers_404_for_an_unknown_or_excluded_product(blocked_client):
    unknown = blocked_client.get("/opportunities/blocked", params={"product": "999999"})
    excluded = blocked_client.get("/opportunities/blocked", params={"product": CRUDE_OIL})

    assert unknown.status_code == 404
    assert excluded.status_code == 404 and "Excluded Product" in excluded.json()["detail"]


def test_blocked_value_matches_a_scenario_adding_that_agreement(client):
    [india] = client.get("/opportunities/top", params={"product": FIXTURE_HS6, "market": "IND"}).json()["opportunities"]
    scenario = client.post("/scenarios", json={"add_agreements": ["IND"]}).json()

    pair = next(row for row in scenario["top_pairs"] if row["hs6"] == FIXTURE_HS6)
    assert india["blocked_opportunity_value_usd"] == pytest.approx(pair["effect_usd"])


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
    # A known tariff falls to 0; an unknown one (India has no telephone rate) stays unknown (ADR 0006).
    assert all(
        row["scenario_tariff_pct"] == 0 if row["baseline_tariff_pct"] is not None else row["scenario_tariff_pct"] is None
        for row in body["top_pairs"]
    )
    assert any((row["baseline_tariff_pct"] or 0) > 0 for row in body["top_pairs"])
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
