"""Sanctioned Markets and Restricted Pairs served: the API's caveats and hide filters, the app's badges and filters
wherever the Re-export Hub caveat appears (Leaderboard, Blocked Opportunities, Scenarios, detail panel), and scores and
rankings unchanged by the tiers.

The fixture world has none of the tables' Markets, so the two tiers are declared here over tables of their own: India
under a sanctions regime since 2020, and whisky a Product India prohibits. The wording is the package's.
"""

from dataclasses import replace

import pytest
from conftest import run_page, serve_api
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.caveats import (
    FLAGGED_PRODUCT,
    REEXPORT_HUB,
    CaveatContext,
    restricted_pair_tier,
    sanctioned_market_tier,
    with_caveat_columns,
)
from exportspredictor.pipeline import recaveat
from exportspredictor.rankings import rank_opportunities
from exportspredictor.restrictions import Prohibition, SanctionsRegime

WHISKY, INDIA = "220830", "IND"
SANCTIONED = sanctioned_market_tier(
    (SanctionsRegime(INDIA, "Test", 2020, None, "https://www.gov.uk/test", "https://www.legislation.gov.uk/test"),)
)
RESTRICTED = restricted_pair_tier((Prohibition(INDIA, "India", "whisky", (WHISKY,), "https://example.org"),))
TIERS = (FLAGGED_PRODUCT, REEXPORT_HUB, SANCTIONED, RESTRICTED)
SANCTION_REASON = "the UK's Test sanctions regime, broad since 2020"
RESTRICTION_REASON = "India prohibits imports of whisky"
INDIA_SCENARIO = {"add_agreements": [INDIA]}


@pytest.fixture(scope="module")
def marked(fixture_artefacts):
    baseline = with_caveat_columns(fixture_artefacts.baseline, CaveatContext(fixture_artefacts.baseline_year), tiers=TIERS)
    return replace(fixture_artefacts, baseline=baseline)


@pytest.fixture(scope="module")
def marked_dir(marked, tmp_path_factory):
    root = tmp_path_factory.mktemp("restrictions")
    store.save(marked, root)
    return root


@pytest.fixture(scope="module")
def client(marked_dir):
    with TestClient(create_app(marked_dir, tiers=TIERS)) as client:
        yield client


def _get(client, path: str, **params) -> dict:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


# --- The fixture pipeline --------------------------------------------------------------------------------------------


def test_the_fixture_baseline_carries_both_tiers_unflagged_as_it_has_none_of_the_tables_markets(fixture_artefacts):
    baseline = fixture_artefacts.baseline

    assert set(fixture_artefacts.markets["iso3"]) == {"JPN", "CAN", "IND", "HKG"}
    assert not baseline["sanctioned_market"].any() and baseline["sanctioned_market_reason"].is_null().all()
    assert not baseline["restricted_pair"].any() and baseline["restricted_pair_reason"].is_null().all()


def test_scores_and_rankings_are_unchanged_by_the_tiers(fixture_artefacts, marked):
    tier_columns = [column for tier in (SANCTIONED, RESTRICTED) for column in tier.columns]
    plain = rank_opportunities(fixture_artefacts).drop(tier_columns)
    flagged = rank_opportunities(marked).drop(tier_columns)

    assert marked.baseline["sanctioned_market"].any() and marked.baseline["restricted_pair"].any()
    assert flagged.equals(plain)


def test_recaveat_marks_a_saved_baseline_with_the_declared_tiers_and_changes_nothing_else(fixture_artefacts):
    tier_columns = [column for tier in (SANCTIONED, RESTRICTED) for column in tier.columns]
    older = replace(fixture_artefacts, baseline=fixture_artefacts.baseline.drop(tier_columns))

    marked = recaveat(older, version="recaveated", tiers=TIERS)

    assert marked.version == "recaveated" and marked.metadata["refreshed_from"] == "fixture-test"
    assert marked.baseline.drop(tier_columns).equals(older.baseline)
    assert marked.baseline.filter("sanctioned_market")["iso3"].unique().to_list() == [INDIA]
    assert marked.baseline.filter("restricted_pair").select("hs6", "iso3").rows() == [(WHISKY, INDIA)]
    assert marked.fold_models is older.fold_models


# --- The API ---------------------------------------------------------------------------------------------------------


def test_the_api_serves_both_tiers_with_their_wording(client):
    tiers = {tier["column"]: tier for tier in _get(client, "/caveats")}

    sanctioned, restricted = tiers["sanctioned_market"], tiers["restricted_pair"]
    assert (sanctioned["badge"], sanctioned["plural"], sanctioned["hide_param"], sanctioned["key"]) == (
        "Sanctioned Market", "Sanctioned Markets", "hide_sanctioned", "market"
    )
    assert (restricted["badge"], restricted["plural"], restricted["hide_param"], restricted["key"]) == (
        "Restricted Pair", "Restricted Pairs", "hide_restricted", "pair"
    )
    assert sanctioned["reason_column"] == "sanctioned_market_reason"
    assert restricted["reason_column"] == "restricted_pair_reason"


def test_ranking_rows_and_explanations_carry_both_caveats_and_their_reasons(client):
    rows = _get(client, "/opportunities/top")["opportunities"]

    for row in rows:
        india, whisky_to_india = row["iso3"] == INDIA, (row["hs6"], row["iso3"]) == (WHISKY, INDIA)
        assert (row["sanctioned_market"], row["sanctioned_market_reason"]) == (india, SANCTION_REASON if india else None)
        assert (row["restricted_pair"], row["restricted_pair_reason"]) == (
            whisky_to_india, RESTRICTION_REASON if whisky_to_india else None
        )
    explained = _get(client, f"/explain/{WHISKY}/{INDIA}")
    assert explained["sanctioned_market_reason"] == SANCTION_REASON
    assert explained["restricted_pair_reason"] == RESTRICTION_REASON


@pytest.mark.parametrize("path", ["/opportunities/top", "/opportunities/by-product/220830", "/opportunities/blocked"])
@pytest.mark.parametrize(("hide", "column"), [("hide_sanctioned", "sanctioned_market"), ("hide_restricted", "restricted_pair")])
def test_each_hide_filter_drops_exactly_its_rows(client, path, hide, column):
    everything = _get(client, path)["opportunities"]

    shown = _get(client, path, **{hide: True})["opportunities"]

    assert any(row[column] for row in everything) or path == "/opportunities/blocked"
    assert shown == [row for row in everything if not row[column]]


def test_scenario_rows_carry_every_caveat(client):
    response = client.post("/scenarios", json=INDIA_SCENARIO)
    assert response.status_code == 200, response.text
    body = response.json()

    pairs = body["top_pairs"] + body["blocked_opportunities"]
    assert pairs and all(pair["iso3"] == INDIA for pair in body["top_pairs"])
    for pair in pairs:
        assert (pair["sanctioned_market"], pair["sanctioned_market_reason"]) == (True, SANCTION_REASON)
        assert pair["restricted_pair"] is (pair["hs6"] == WHISKY)
        assert pair["reexport_hub"] is False
        assert "flagged_product" in pair


# --- The app ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_url(marked_dir):
    with serve_api(marked_dir, tiers=TIERS) as url:
        yield url


def test_the_leaderboard_hides_both_by_default_and_shows_their_badges_on_request(monkeypatch, api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = run_page(query={"product": WHISKY})
    [*switch] = page.segmented_control
    switch[0].set_value("Wide table").run()

    assert "India" not in set(page.dataframe[0].value["Market"])
    sanctioned = page.sidebar.checkbox(key="hide_sanctioned")
    restricted = page.sidebar.checkbox(key="hide_restricted")
    assert sanctioned.label == "Hide Sanctioned Markets" and sanctioned.value is True
    assert restricted.label == "Hide Restricted Pairs" and restricted.value is True

    sanctioned.uncheck().run()
    page.sidebar.checkbox(key="hide_restricted").uncheck().run()

    caveats = dict(zip(page.dataframe[0].value["Market"], page.dataframe[0].value["Caveats"], strict=True))
    assert caveats["India"] == "Sanctioned Market, Restricted Pair" and caveats["Japan"] == ""
    assert {"Sanctioned Markets shown", "Restricted Pairs shown"} <= {button.label for button in page.main.button}


def test_the_detail_panel_names_the_regime_and_the_prohibition(monkeypatch, api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = run_page(
        query={"product": WHISKY, "market": INDIA, "hide_sanctioned": "false", "hide_restricted": "false"}
    )

    warnings = [element.value for element in page.warning]
    assert SANCTIONED.caveat_text(SANCTION_REASON) in warnings
    assert RESTRICTED.caveat_text(RESTRICTION_REASON) in warnings
    assert any("broad since 2020" in text for text in warnings)


def test_the_blocked_page_has_both_hide_filters(monkeypatch, api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = run_page("pages/blocked_opportunities.py")

    assert page.sidebar.checkbox(key="blocked_hide_sanctioned").label == "Hide Sanctioned Markets"
    assert page.sidebar.checkbox(key="blocked_hide_restricted").label == "Hide Restricted Pairs"


def test_the_scenarios_page_shows_each_pair_s_caveats(monkeypatch, api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = run_page("pages/scenarios.py")
    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value([INDIA]).run()
    page.button(key="run").click().run()
    assert not page.exception

    pair_tables = [table.value for table in page.dataframe if "Tariff, Scenario %" in table.value.columns]
    assert pair_tables and all("Caveats" in table.columns for table in pair_tables)
    caveats = dict(zip(pair_tables[0]["Product"], pair_tables[0]["Caveats"], strict=True))
    assert caveats["Whiskies"] == "Sanctioned Market, Restricted Pair"

    page.session_state["scenario_drill_pair"] = {"hs6": WHISKY, "iso3": INDIA}
    page.run()
    warnings = [element.value for element in page.warning]
    assert RESTRICTED.caveat_text(RESTRICTION_REASON) in warnings

