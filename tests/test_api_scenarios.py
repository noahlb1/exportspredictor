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


def test_presets_leave_out_those_that_change_nothing_in_the_baseline(client):
    # The fixture Baseline has no EU member Market and the UK in no Trading Bloc, so only CPTPP changes anything.
    body = client.get("/scenarios/presets").json()

    assert [preset["name"] for preset in body] == ["cptpp"]
    assert {"name", "title", "description"} == set(body[0])


def test_presets_offer_leave_and_the_eu_in_a_baseline_where_they_change_something(fixture_artefacts, tmp_path):
    # India is made an EU member, so the EU presets reach it, and so is the UK, so Leave has a Trading Bloc to leave.
    features = fixture_artefacts.baseline_features
    india = pl.col("iso3") == "IND"
    flags = {"eu_d": india, "eu_o": india}
    changed = features.with_columns(
        pl.when(rows).then(1).otherwise(pl.col(column)).cast(features.schema[column]).alias(column)
        for column, rows in flags.items()
    )
    store.save(replace(fixture_artefacts, baseline_features=changed), tmp_path)

    with TestClient(create_app(tmp_path)) as member:
        names = [preset["name"] for preset in member.get("/scenarios/presets").json()]
        runs = {name: member.post("/scenarios", json={"preset": name}).json() for name in names}

    assert names == ["eu_rejoin", "eu_customs_union", "cptpp", "leave"]
    assert not any(body["changes_nothing"] for body in runs.values())


def test_eu_rejoin_is_described_by_the_customs_union_and_membership_not_a_trade_agreement_already_held(
    fixture_artefacts, tmp_path
):
    # India is made an EU member, so EU re-join is offered.
    features = fixture_artefacts.baseline_features
    eu = features.with_columns(
        pl.when(pl.col("iso3") == "IND").then(1).otherwise(pl.col("eu_d")).cast(features.schema["eu_d"]).alias("eu_d")
    )
    store.save(replace(fixture_artefacts, baseline_features=eu), tmp_path)

    with TestClient(create_app(tmp_path)) as member:
        presets = {preset["name"]: preset for preset in member.get("/scenarios/presets").json()}
    description = presets["eu_rejoin"]["description"]

    assert description.startswith("The Customs Union and EU membership with every EU member")
    assert "the UK already has a Trade Agreement with each" in description
    assert not description.startswith("A Trade Agreement")


def test_no_offered_preset_runs_to_changes_nothing(client):
    for preset in client.get("/scenarios/presets").json():
        body = client.post("/scenarios", json={"preset": preset["name"]}).json()

        assert body["changes_nothing"] is False, preset["name"]


def test_a_custom_scenario_adding_an_agreement_returns_non_negative_effects(client):
    response = client.post("/scenarios", json={"add_agreements": ["IND"]})

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
        "caveat_texts",  # the caveat text of each tier the pair is in, worded by the API
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


@pytest.mark.parametrize(
    "thresholds",
    [{"min_effect_share": 0}, {"min_effect_usd": 0}, {"min_effect_share": 0.2, "min_effect_usd": 1_000_000}],
)
def test_a_scenario_request_carrying_thresholds_is_refused(client, thresholds):
    # The thresholds are fixed (ADR 0012): no client can bring back a second Blocked Opportunity rule.
    response = client.post("/scenarios", json={"add_agreements": ["IND"], **thresholds})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"add_agreements": ["IND"]},
        {"preset": "cptpp"},
        {"remove_agreements": ["JPN"]},
        {"tariff": {"markets": ["JPN"], "rate_pct": 25}},
        {"add_agreements": ["JPN"]},  # changes nothing
    ],
)
def test_every_scenario_response_states_the_fixed_thresholds_20_percent_and_one_million_dollars(client, payload):
    body = client.post("/scenarios", json=payload).json()

    assert body["thresholds"] == {"min_effect_share": 0.2, "min_effect_usd": 1_000_000}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"preset": "cptpp", "add_agreements": ["IND"]},
        {"preset": "not-a-preset"},
        {"add_agreements": ["XXX"]},
        {"add_agreements": ["IND"], "remove_agreements": ["IND"]},
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
    assert body["markets"] == body["top_pairs"] == []
    assert body["adds_agreement"] is False
    assert body["blocked_opportunities"] is None and body["blocked_count"] is None
    assert "changes nothing" in body["summary"]


def test_by_product_ranks_by_export_gap_by_default(client):
    rows = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()["opportunities"]

    gaps = [row["export_gap_usd"] for row in rows]
    assert gaps == sorted(gaps, reverse=True)


# --- The Blocked Opportunities read (#109), on the Blocked world (`conftest.blocked_artefacts`) ------------------

WHISKY, JEWELLERY, CARS, CRUDE_OIL = "220830", "711319", "870323", "270900"
# The Blocked world's Blocked Opportunities (see `conftest.blocked_artefacts`): jewellery is the Flagged Product and
# Hong Kong the Re-export Hub. Cars in Hong Kong clears both thresholds but has no room to grow.
BLOCKED_PAIRS = {(WHISKY, "IND"), (JEWELLERY, "IND"), (JEWELLERY, "HKG")}


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
    assert body["order"]["title"] == "Rise with agreement"
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
    # India's two (whisky and jewellery) total more than Hong Kong's jewellery; its cars have no room to grow.
    assert [(market["iso3"], market["blocked_opportunities"]) for market in body["markets"]] == [("IND", 2), ("HKG", 1)]


@pytest.mark.parametrize(
    ("params", "keep"),
    [
        ({"product": JEWELLERY}, lambda row: row["hs6"] == JEWELLERY),
        ({"sector": "22"}, lambda row: row["sector"] == "22"),
        ({"region": "South Asia"}, lambda row: row["region"] == "South Asia"),
        ({"min_value_usd": 1e7}, lambda row: row["blocked_opportunity_value_usd"] >= 1e7),
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


def test_potential_with_agreement_is_baseline_export_potential_plus_the_value_with_room_to_grow(blocked_client):
    rows = _blocked(blocked_client)["opportunities"]

    for row in rows:
        baseline = blocked_client.get(f"/explain/{row['hs6']}/{row['iso3']}").json()["baseline_export_potential_usd"]
        assert row["baseline_export_potential_usd"] == pytest.approx(baseline)
        assert row["potential_with_agreement_usd"] == pytest.approx(baseline + row["blocked_opportunity_value_usd"])
        assert row["baseline_actual_exports_usd"] < row["potential_with_agreement_usd"]
        assert row["rise_share"] == pytest.approx(row["blocked_opportunity_value_usd"] / baseline)
        assert "blocked_opportunity_uplift" not in row


def test_a_pair_already_exporting_above_export_potential_with_an_agreement_is_no_blocked_opportunity(blocked_client):
    # ADR 0011 "Room to grow": cars in Hong Kong clear both thresholds, but the UK already sells more there than the
    # model's Export Potential with a Trade Agreement.
    [cars] = blocked_client.get("/opportunities/top", params={"product": CARS, "market": "HKG"}).json()["opportunities"]
    potential = blocked_client.get(f"/explain/{CARS}/HKG").json()["baseline_export_potential_usd"]
    blocked = _blocked(blocked_client)

    assert cars["blocked_opportunity_value_usd"] >= max(1e6, 0.2 * potential)
    assert cars["blocked_opportunity"] is False
    assert (CARS, "HKG") not in {(row["hs6"], row["iso3"]) for row in blocked["opportunities"]}
    # Nor does it count in the summary or its Market's total.
    hong_kong = next(market for market in blocked["markets"] if market["iso3"] == "HKG")
    assert hong_kong["blocked_opportunities"] == 1


def test_the_leaderboard_flags_exactly_the_blocked_reads_pairs(blocked_client):
    rows = blocked_client.get("/opportunities/top", params={"limit": 1000}).json()["opportunities"]

    assert {(row["hs6"], row["iso3"]) for row in rows if row["blocked_opportunity"]} == BLOCKED_PAIRS


def test_the_tariff_part_and_the_agreement_part_sum_to_the_value(blocked_client):
    rows = {(row["hs6"], row["iso3"]): row for row in _blocked(blocked_client)["opportunities"]}

    for row in rows.values():
        assert row["tariff_part_usd"] + row["agreement_part_usd"] == pytest.approx(row["blocked_opportunity_value_usd"])
    whisky = rows[(WHISKY, "IND")]
    assert whisky["tariff_part_usd"] == pytest.approx(0.4 * whisky["blocked_opportunity_value_usd"])
    assert whisky["barrier"] == "No agreement, tariff 150%"


def test_each_blocked_opportunity_carries_the_sentence_splitting_its_value(blocked_client):
    rows = {(row["hs6"], row["iso3"]): row for row in _blocked(blocked_client)["opportunities"]}

    assert rows[(WHISKY, "IND")]["value_split"].startswith(
        "Of the $18.5m rise, the model associates $7.4m (40%, the Tariff part) with the Applied Tariff of 150% going to 0"
    )
    assert "already 0%" in rows[(JEWELLERY, "HKG")]["value_split"]
    assert "none of the $7.1m rise" in rows[(JEWELLERY, "IND")]["value_split"]


def test_a_0_percent_tariff_has_no_tariff_part_and_says_so_in_its_barrier(blocked_client):
    rows = {(row["hs6"], row["iso3"]): row for row in _blocked(blocked_client)["opportunities"]}
    jewellery = rows[(JEWELLERY, "HKG")]

    assert jewellery["applied_tariff_pct"] == 0
    assert jewellery["tariff_part_usd"] == 0
    assert jewellery["agreement_part_usd"] == pytest.approx(jewellery["blocked_opportunity_value_usd"])
    assert jewellery["barrier"] == "No agreement, tariff 0%"


def test_an_unknown_tariff_stays_unknown_with_no_tariff_part(fixture_artefacts, tmp_path):
    # India's jewellery tariff is made unknown, the values are rebuilt from the saved fold models, then scaled into
    # the Blocked world.
    from exportspredictor import scenarios

    unknown = (pl.col("hs6") == JEWELLERY) & (pl.col("iso3") == "IND")
    features = fixture_artefacts.baseline_features.with_columns(
        pl.when(unknown).then(None).otherwise(pl.col("log_applied_tariff")).alias("log_applied_tariff")
    )
    older = replace(fixture_artefacts, baseline_features=features, metadata={**fixture_artefacts.metadata, "version": "1"})
    store.save(older, tmp_path / "source")
    rebuilt = store.load(scenarios.rebuild_blocked_values(tmp_path / "source", version="2"))
    scaled = replace(
        rebuilt,
        baseline=rebuilt.baseline.with_columns(pl.col(r"^.*_usd$") * 1000),
        history=rebuilt.history.with_columns(pl.col(r"^.*_usd$") * 1000),
    )
    store.save(scaled, tmp_path / "served")

    with TestClient(create_app(tmp_path / "served")) as client:
        rows = {(row["hs6"], row["iso3"]): row for row in _blocked(client)["opportunities"]}

    jewellery = rows[(JEWELLERY, "IND")]
    assert jewellery["applied_tariff_pct"] is None
    assert jewellery["tariff_part_usd"] == 0
    assert jewellery["agreement_part_usd"] == pytest.approx(jewellery["blocked_opportunity_value_usd"])
    assert jewellery["barrier"] == "No agreement, tariff unknown"


def test_the_above_export_potential_flag_marks_actual_exports_above_baseline_export_potential(blocked_client):
    rows = {(row["hs6"], row["iso3"]): row for row in _blocked(blocked_client)["opportunities"]}

    # Jewellery in Hong Kong sells above its Export Potential, though below its level with a Trade Agreement.
    assert {pair for pair, row in rows.items() if row["above_export_potential"]} == {(JEWELLERY, "HKG")}
    for row in rows.values():
        above = row["baseline_actual_exports_usd"] > row["baseline_export_potential_usd"]
        assert row["above_export_potential"] is above


def test_the_blocked_read_orders_by_tariff_part_on_request(blocked_client):
    by_value = _blocked(blocked_client)
    by_tariff = _blocked(blocked_client, rank_by="tariff_part")

    order = [(row["hs6"], row["iso3"]) for row in by_tariff["opportunities"]]
    # Whisky in India leads on its Tariff part; the two 0% rows follow by value.
    assert order == [(WHISKY, "IND"), (JEWELLERY, "IND"), (JEWELLERY, "HKG")]
    assert by_tariff["rank_by"] == "tariff_part" and by_value["rank_by"] == "blocked"
    assert by_tariff["order"]["title"] == "Tariff part" and "associates" in by_tariff["order"]["description"]
    assert by_tariff["summary"] == by_value["summary"] and by_tariff["markets"] == by_value["markets"]


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


def test_a_scenarios_blocked_opportunities_are_its_flagged_pairs(client):
    body = client.post("/scenarios", json={"add_agreements": ["HKG"], "limit": 1000}).json()
    flagged = {(row["hs6"], row["iso3"]) for row in body["top_pairs"] if row["blocked_opportunity"]}

    # The fixture's tiny Scenario Effects clear no $1M threshold; the rule's room to grow is tested in the package.
    assert {(row["hs6"], row["iso3"]) for row in body["blocked_opportunities"]} == flagged
    assert body["blocked_count"] == len(flagged)


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
    body = client.post("/scenarios", json={"add_agreements": ["IND"]}).json()

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


def test_the_thresholds_read_is_gone(client):
    # Nothing to read: the thresholds are fixed, and every response applying them states them (ADR 0012).
    assert client.get("/scenarios/thresholds").status_code == 404


@pytest.fixture(scope="module")
def unscenario_client(unscenario_artefacts_dir):
    with TestClient(create_app(unscenario_artefacts_dir)) as client:
        yield client


def test_markets_flag_the_markets_a_scenario_accepts(unscenario_client):
    flags = {market["iso3"]: market["scenario"] for market in unscenario_client.get("/markets").json()}

    assert flags == {"JPN": True, "CAN": True, "IND": True, "HKG": True, "GBR": False, "MYT": False}


def test_a_scenario_naming_any_market_flagged_true_is_accepted(unscenario_client):
    accepted = [market["iso3"] for market in unscenario_client.get("/markets").json() if market["scenario"]]

    assert unscenario_client.post("/scenarios", json={"add_agreements": accepted}).status_code == 200
    tariff = {"tariff": {"markets": accepted, "rate_pct": 5.0}}
    assert unscenario_client.post("/scenarios", json=tariff).status_code == 200


@pytest.mark.parametrize("iso3", ["GBR", "MYT"])
def test_a_scenario_naming_a_market_flagged_false_is_refused(unscenario_client, iso3):
    assert unscenario_client.post("/scenarios", json={"add_agreements": [iso3]}).status_code == 422


def test_rank_orders_are_served_with_titles(client):
    body = client.get("/opportunities/rank-by").json()

    # Blocked Opportunity value is the Blocked Opportunities page's order, not a ranking toggle.
    assert [order["name"] for order in body] == ["gap", "relative"]
    assert [order["title"] for order in body] == ["Export Gap", "Relative Export Gap"]


# --- Tariffs in Scenarios (ADR 0006) ------------------------------------------------------


def _scenario(client, payload: dict) -> dict:
    response = client.post("/scenarios", json={**payload, "limit": 1000})
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


def _post(client, payload: dict) -> dict:
    response = client.post("/scenarios", json={**payload, "limit": 1000})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_tariff_scenario_counts_the_pairs_it_raises_and_lowers(client):
    # India's Baseline tariffs: whisky 150%, medicines 10%, jewellery 20%, telephones unknown, cars 60%.
    body = _post(client, {"tariff": {"markets": ["IND"], "rate_pct": 30}})

    assert len(body["top_pairs"]) == 5  # the unknown tariff is changed too, but counts in neither
    assert body["pairs_raised"] == 2  # medicines and jewellery
    assert body["pairs_lowered"] == 2  # whisky and cars


def test_a_rate_between_two_known_tariffs_raises_one_pair_and_lowers_the_other(client):
    # Medicines (chapter 30) at 10% and jewellery (chapter 71) at 20%, set to 15%.
    body = _post(client, {"tariff": {"markets": ["IND"], "rate_pct": 15, "sectors": ["30", "71"]}})

    assert (body["pairs_raised"], body["pairs_lowered"]) == (1, 1)


def test_a_pair_already_at_the_set_rate_counts_in_neither(client):
    body = _post(client, {"tariff": {"markets": ["IND", "JPN"], "rate_pct": 10}})

    # Japan's five 0% tariffs rise; India's whisky, jewellery and cars fall; medicines stay at 10%.
    assert (body["pairs_raised"], body["pairs_lowered"]) == (5, 3)


@pytest.mark.parametrize(
    "payload",
    [{"add_agreements": ["IND"]}, {"remove_agreements": ["JPN"]}, {"preset": "cptpp"}, {"add_agreements": ["JPN"]}],
)
def test_a_scenario_that_is_no_tariff_change_has_no_raised_or_lowered_counts(client, payload):
    body = _post(client, payload)

    assert body["pairs_raised"] is None and body["pairs_lowered"] is None


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


# --- A Scenario's Blocked Opportunities (#148, ADR 0012): one source, the Blocked Opportunities read -------------


def _default_hidden(client) -> dict[str, bool]:
    """The hide filters of the Caveat Tiers the app hides by default, all on."""
    return {tier["hide_param"]: True for tier in client.get("/caveats").json() if tier["hidden_by_default"]}


def _blocked_in(client, *markets: str, **params) -> tuple[list[dict], int]:
    """The Blocked Opportunities read's rows in `markets` and their count, default-hidden Caveat Tiers hidden."""
    body = _blocked(client, **_default_hidden(client), **params)
    rows = [row for row in body["opportunities"] if row["iso3"] in markets]
    count = sum(market["blocked_opportunities"] for market in body["markets"] if market["iso3"] in markets)
    return rows, count


def test_adding_a_trade_agreement_with_india_lists_the_blocked_opportunities_read_for_india(blocked_client):
    expected, count = _blocked_in(blocked_client, "IND")

    body = _post(blocked_client, {"add_agreements": ["IND"]})

    assert body["adds_agreement"] is True
    assert body["blocked_opportunities"] == expected
    assert body["blocked_count"] == count == 2
    assert {(row["hs6"], row["iso3"]) for row in expected} == {(WHISKY, "IND"), (JEWELLERY, "IND")}
    for row in body["blocked_opportunities"]:
        assert set(row) >= {"blocked_opportunity_value_usd", "tariff_part_usd", "agreement_part_usd", "barrier"}


def test_a_scenario_lists_the_market_s_blocked_opportunities_with_the_default_hidden_caveat_tiers_hidden(
    offset_artefacts_dir,
):
    # In the Offset world, cars in Hong Kong is a Blocked Opportunity and an Offset Pair, a tier hidden by default.
    with TestClient(create_app(offset_artefacts_dir)) as offset:
        everything = _blocked(offset)["opportunities"]
        shown, count = _blocked_in(offset, "HKG")
        body = _post(offset, {"add_agreements": ["HKG"]})

    offset_pairs = {(row["hs6"], row["iso3"]) for row in everything if row["iso3"] == "HKG" and row["offset_pair"]}
    assert offset_pairs == {(CARS, "HKG")}
    assert body["blocked_opportunities"] == shown
    assert body["blocked_count"] == count
    assert not offset_pairs & {(row["hs6"], row["iso3"]) for row in body["blocked_opportunities"]}


@pytest.mark.parametrize(
    "payload",
    [
        {"tariff": {"markets": ["IND"], "rate_pct": 0}},
        {"preset": "eu_customs_union"},
        {"remove_agreements": ["JPN"]},
        {"add_agreements": ["JPN"]},  # Japan already has one
        {"add_agreements": ["CAN"]},  # and so has Canada
    ],
)
def test_a_scenario_adding_no_trade_agreement_where_none_is_in_force_has_no_blocked_opportunities(
    blocked_client, payload
):
    body = _post(blocked_client, payload)

    assert body["adds_agreement"] is False
    assert body["blocked_opportunities"] is None
    assert body["blocked_count"] is None
    assert "Blocked Opportunit" not in body["summary"]


def test_eu_customs_union_adds_no_agreement_where_eu_re_join_adds_one(fixture_artefacts, tmp_path):
    # India is made an EU member: the Customs Union alone adds no Trade Agreement, EU re-join adds one with India.
    features = fixture_artefacts.baseline_features
    eu = features.with_columns(
        pl.when(pl.col("iso3") == "IND").then(1).otherwise(pl.col("eu_d")).cast(features.schema["eu_d"]).alias("eu_d")
    )
    store.save(replace(fixture_artefacts, baseline_features=eu), tmp_path)

    with TestClient(create_app(tmp_path)) as member:
        customs_union = _post(member, {"preset": "eu_customs_union"})
        rejoin = _post(member, {"preset": "eu_rejoin"})

    assert customs_union["changes_nothing"] is False
    assert customs_union["adds_agreement"] is False and customs_union["blocked_count"] is None
    assert rejoin["adds_agreement"] is True and rejoin["blocked_count"] == 0  # the fixture's values are under $1M


def test_cptpp_adds_no_agreement_where_every_member_already_has_one(client):
    # Japan and Canada, the fixture's CPTPP members, both have a Trade Agreement in force.
    body = _post(client, {"preset": "cptpp"})

    assert body["changes_nothing"] is False
    assert body["adds_agreement"] is False and body["blocked_opportunities"] is None


def test_a_mixed_scenario_lists_only_the_blocked_opportunities_of_the_market_it_adds(blocked_client):
    expected, count = _blocked_in(blocked_client, "IND")

    body = _post(blocked_client, {"add_agreements": ["IND"], "remove_agreements": ["JPN"]})

    assert body["adds_agreement"] is True
    assert {row["iso3"] for row in body["markets"]} == {"IND", "JPN"}
    assert body["blocked_opportunities"] == expected
    assert body["blocked_count"] == count == 2


def test_adding_two_markets_lists_both_ranked_by_value_and_limited(blocked_client):
    expected, count = _blocked_in(blocked_client, "IND", "HKG")

    body = _post(blocked_client, {"add_agreements": ["IND", "HKG"]})
    top = blocked_client.post("/scenarios", json={"add_agreements": ["IND", "HKG"], "limit": 2}).json()

    assert body["blocked_opportunities"] == expected
    values = [row["blocked_opportunity_value_usd"] for row in expected]
    assert values == sorted(values, reverse=True)
    assert top["blocked_opportunities"] == expected[:2]
    assert top["blocked_count"] == count == 3
