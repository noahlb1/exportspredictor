"""Adding a Caveat Tier needs only its declaration (#118): throwaway tiers declared here reach the API and the app.

The fixture artefact's Baseline is marked with the declared tiers (`with_caveat_columns`), and the API is created with
them (`create_app(tiers=...)`); no API or app code names them.
"""

from dataclasses import replace

import polars as pl
import pytest
from conftest import run_page, serve_api
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.caveats import (
    TIERS,
    CaveatContext,
    CaveatTier,
    with_caveat_columns,
)

WHISKY, INDIA = "220830", "IND"

# A Market-keyed tier (India) and a pair-keyed tier with a reason per row (whisky to India).
TEST_MARKET = CaveatTier(
    column="test_market",
    badge="Test Market",
    plural="Test Markets",
    text="Test Market: a throwaway Market-keyed tier.",
    key="market",
    hide_param="hide_test_markets",
    hide_description="Leave out Test Markets",
    flags=lambda context: pl.col("iso3") == INDIA,
)
TEST_PAIR = CaveatTier(
    column="test_pair",
    badge="Test Pair",
    plural="Test Pairs",
    text="Test Pair: a throwaway pair-keyed tier.",
    key="pair",
    hide_param="hide_test_pairs",
    hide_description="Leave out Test Pairs",
    flags=lambda context: (pl.col("hs6") == WHISKY) & (pl.col("iso3") == INDIA),
    reason=lambda context: pl.format("banned since {}", pl.lit(context.baseline_year)),
    row_text="Test Pair: {reason}.",
)
DECLARED = (*TIERS, TEST_MARKET, TEST_PAIR)


@pytest.fixture(scope="module")
def declared_dir(fixture_artefacts, tmp_path_factory):
    marked = with_caveat_columns(
        fixture_artefacts.baseline, CaveatContext(fixture_artefacts.baseline_year), tiers=DECLARED
    )
    root = tmp_path_factory.mktemp("declared_tiers")
    store.save(replace(fixture_artefacts, baseline=marked), root)
    return root


@pytest.fixture(scope="module")
def client(declared_dir):
    with TestClient(create_app(declared_dir, tiers=DECLARED)) as client:
        yield client


def _get(client, path: str, **params) -> dict:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_api_serves_every_declared_tier(client):
    tiers = _get(client, "/caveats")

    assert [tier["column"] for tier in tiers] == ["flagged_product", "reexport_hub", "test_market", "test_pair"]
    market = tiers[2]
    assert market == {
        "column": "test_market",
        "badge": "Test Market",
        "plural": "Test Markets",
        "text": "Test Market: a throwaway Market-keyed tier.",
        "key": "market",
        "hide_param": "hide_test_markets",
        "reason_column": None,
        "row_text": None,
    }
    assert tiers[3]["reason_column"] == "test_pair_reason" and tiers[3]["row_text"] == "Test Pair: {reason}."
    assert _get(client, "/labels")["caveats"]["test_market"]["badge"] == "Test Market"


def test_ranking_rows_and_explanations_carry_each_tier_s_column_and_reason(client):
    rows = _get(client, "/opportunities/top")["opportunities"]

    assert rows and all(row["test_market"] is (row["iso3"] == INDIA) for row in rows)
    for row in rows:
        flagged = (row["hs6"], row["iso3"]) == (WHISKY, INDIA)
        assert row["test_pair"] is flagged
        assert row["test_pair_reason"] == ("banned since 2024" if flagged else None)
    explained = _get(client, f"/explain/{WHISKY}/{INDIA}")
    assert (explained["test_market"], explained["test_pair"], explained["test_pair_reason"]) == (
        True, True, "banned since 2024"
    )


@pytest.mark.parametrize("path", ["/opportunities/top", "/opportunities/by-product/220830", "/opportunities/blocked"])
def test_each_tier_s_hide_filter_drops_its_rows(client, path):
    everything = _get(client, path)["opportunities"]

    shown = _get(client, path, hide_test_markets=True)["opportunities"]

    assert shown == [row for row in everything if not row["test_market"]]


def test_the_hide_filter_drops_rows_of_the_market_from_a_market_ranking(client):
    assert _get(client, f"/opportunities/by-market/{INDIA}")["opportunities"]
    assert not _get(client, f"/opportunities/by-market/{INDIA}", hide_test_markets=True)["opportunities"]


def test_an_undeclared_hide_filter_is_refused(client):
    assert client.get("/opportunities/top", params={"hide_sanctioned": True}).status_code == 422


@pytest.fixture(scope="module")
def declared_api_url(declared_dir):
    with serve_api(declared_dir, tiers=DECLARED) as url:
        yield url


def test_the_app_shows_each_tier_s_badge_filter_and_notice(monkeypatch, declared_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", declared_api_url)
    page = run_page(query={"product": WHISKY})
    [*switch] = page.segmented_control
    switch[0].set_value("Wide table").run()

    caveats = dict(zip(page.dataframe[0].value["Market"], page.dataframe[0].value["Caveats"], strict=True))
    assert caveats["India"] == "Test Market, Test Pair" and caveats["Japan"] == ""
    hide = page.sidebar.checkbox(key="hide_test_markets")
    assert hide.label == "Hide Test Markets" and hide.value is False

    hide.check().run()

    assert "India" not in set(page.dataframe[0].value["Market"])
    assert page.query_params["hide_test_markets"] == "true"
    assert any("Test Markets hidden" in element.value for element in page.main.markdown)


def test_the_app_s_detail_panel_names_a_row_s_reason(monkeypatch, declared_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", declared_api_url)
    page = run_page(query={"product": WHISKY, "market": INDIA})

    warnings = [element.value for element in page.warning]
    assert "Test Market: a throwaway Market-keyed tier." in warnings
    assert "Test Pair: banned since 2024." in warnings


def test_the_blocked_page_has_each_tier_s_hide_filter(monkeypatch, declared_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", declared_api_url)
    page = run_page("pages/blocked_opportunities.py")

    assert page.sidebar.checkbox(key="blocked_hide_test_markets").label == "Hide Test Markets"
