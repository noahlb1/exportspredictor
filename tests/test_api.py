"""API contract tests: FastAPI TestClient against fixture artefacts (the main seam)."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store

FIXTURE_HS6 = "220830"


@pytest.fixture(scope="module")
def client(fixture_artefacts_dir):
    with TestClient(create_app(fixture_artefacts_dir)) as client:
        yield client


def test_health_reports_version_and_baseline_year_once_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "artefact_version": "fixture-test", "baseline_year": 2024}


def test_health_is_not_ready_until_artefacts_load(fixture_artefacts, tmp_path):
    now = [0.0]
    with TestClient(create_app(tmp_path, retry_seconds=60, clock=lambda: now[0])) as client:
        before = client.get("/health")
        store.save(fixture_artefacts, tmp_path)
        within_interval = client.get("/health")
        now[0] = 61.0
        after = client.get("/health")

    assert before.status_code == 503
    assert before.json()["ready"] is False
    # A failed load is not retried on every request (no disk scan per call) ...
    assert within_interval.status_code == 503
    # ... but once the retry interval has passed, the artefacts load.
    assert after.status_code == 200
    assert after.json()["ready"] is True


def test_health_is_not_ready_when_the_newest_folder_is_v1(fixture_artefacts, tmp_path):
    store.save(fixture_artefacts, tmp_path)
    v1 = {k: v for k, v in fixture_artefacts.metadata.items() if k != "artefact_format"}
    store.save(replace(fixture_artefacts, metadata={**v1, "version": "zzz-v1"}), tmp_path)

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert "zzz-v1" in response.json()["reason"]
    assert "no artefact_format" in response.json()["reason"]


def test_opportunities_need_loaded_artefacts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/opportunities/by-product/{FIXTURE_HS6}")

    assert response.status_code == 503


def test_by_product_returns_every_market_sorted_by_export_gap(client):
    body = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()

    gaps = [row["export_gap_usd"] for row in body["opportunities"]]
    assert gaps == sorted(gaps, reverse=True)
    assert {row["iso3"] for row in body["opportunities"]} == {"JPN", "CAN", "IND"}
    assert body["product"] == {"hs6": "220830", "description": "Whiskies", "sector": "22"}
    assert body["baseline_year"] == 2024
    assert body["artefact_version"] == "fixture-test"


def test_by_product_rows_have_the_documented_shape(client):
    row = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()["opportunities"][0]

    assert set(row) == {
        "iso3",
        "market",
        "region",
        "export_potential_usd",
        "actual_exports_usd",
        "export_gap_usd",
        "trade_agreement",
        "blocked_effect_usd",
        "blocked_opportunity",
    }


def test_unknown_product_returns_404_with_a_message(client):
    response = client.get("/opportunities/by-product/999999")

    assert response.status_code == 404
    assert "999999" in response.json()["detail"]


def test_products_search_finds_a_product_by_its_code(client):
    response = client.get("/products", params={"q": "851712"})

    assert response.status_code == 200
    assert [product["hs6"] for product in response.json()] == ["851712"]


def test_products_search_finds_a_product_by_an_everyday_word(client):
    # The HS 2012 description says "Telephones for cellular networks", never "smartphone".
    hs6s = [product["hs6"] for product in client.get("/products", params={"q": "Smartphone"}).json()]

    assert hs6s[0] == "851712"


def test_products_search_matches_words_in_the_description(client):
    body = client.get("/products", params={"q": "fish frozen"}).json()

    assert body == [{"hs6": "030420", "description": "Fish fillets, frozen", "sector": "03"}]


def test_products_search_by_code_prefix_lists_the_sector(client):
    hs6s = [product["hs6"] for product in client.get("/products", params={"q": "22"}).json()]

    assert hs6s == ["220830"]


def test_unknown_search_text_returns_an_empty_list(client):
    response = client.get("/products", params={"q": "zzz-not-a-product"})

    assert response.status_code == 200
    assert response.json() == []


def test_markets_lists_every_market_with_name_and_region(client):
    response = client.get("/markets")

    assert response.status_code == 200
    assert {market["iso3"]: market for market in response.json()}["JPN"] == {
        "iso3": "JPN",
        "name": "Japan",
        "region": "East Asia & Pacific",
    }
    assert {market["iso3"] for market in response.json()} == {"JPN", "CAN", "IND"}


def test_evaluation_returns_the_report_with_headline_breakdowns_and_limitations(client):
    response = client.get("/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["artefact_version"] == "fixture-test"
    assert body["test_year"] == 2023
    assert set(body["accuracy"]) == {"all_exporters", "uk_out_of_fold"}
    for scope in body["accuracy"].values():
        assert set(scope) == {"overall", "by_sector", "by_region", "by_income"}  # by_income empty without income levels
        assert scope["by_sector"] and scope["by_region"]
    assert body["headline"]["text"]
    assert isinstance(body["headline"]["beats_naive"], bool)
    assert body["stability"]["years"] == [2023, 2024]
    titles = " ".join(item["title"] for item in body["limitations"]).lower()
    assert "goods only" in titles and "tariffs" in titles


def test_evaluation_is_404_when_the_artefact_has_none(fixture_artefacts, tmp_path):
    from dataclasses import replace

    store.save(replace(fixture_artefacts, evaluation=None), tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/evaluation")

    assert response.status_code == 404
    assert "evaluation" in response.json()["detail"].lower()


def test_by_market_returns_every_product_sorted_by_export_gap(client):
    body = client.get("/opportunities/by-market/JPN").json()

    gaps = [row["export_gap_usd"] for row in body["opportunities"]]
    assert gaps == sorted(gaps, reverse=True)
    assert {row["hs6"] for row in body["opportunities"]} == {"030420", "220830", "300490", "851712", "870323"}
    assert body["market"] == {"iso3": "JPN", "name": "Japan", "region": "East Asia & Pacific"}
    assert body["baseline_year"] == 2024
    assert body["artefact_version"] == "fixture-test"


def test_by_market_rows_name_the_product(client):
    rows = client.get("/opportunities/by-market/JPN").json()["opportunities"]

    assert {"hs6": "220830", "description": "Whiskies", "sector": "22"}.items() <= next(
        row for row in rows if row["hs6"] == "220830"
    ).items()


def test_unknown_market_returns_404_with_a_message(client):
    response = client.get("/opportunities/by-market/XYZ")

    assert response.status_code == 404
    assert "XYZ" in response.json()["detail"]


def _ranked(client, path, **params) -> list[dict]:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()["opportunities"]


def test_sector_filter_keeps_only_that_sectors_products(client):
    rows = _ranked(client, "/opportunities/by-market/JPN", sector="85")

    assert [row["hs6"] for row in rows] == ["851712"]


def test_region_filter_keeps_only_that_regions_markets(client):
    rows = _ranked(client, f"/opportunities/by-product/{FIXTURE_HS6}", region="South Asia")

    assert [row["iso3"] for row in rows] == ["IND"]


def test_filters_apply_on_both_views(client):
    # Japan is not in South Asia, and whisky is not in Sector 85.
    assert _ranked(client, "/opportunities/by-market/JPN", region="South Asia") == []
    assert _ranked(client, f"/opportunities/by-product/{FIXTURE_HS6}", sector="85") == []


def test_min_gap_usd_filter_drops_smaller_export_gaps(client):
    everything = _ranked(client, "/opportunities/by-market/JPN")
    rows = _ranked(client, "/opportunities/by-market/JPN", min_gap_usd=0)

    assert rows and all(row["export_gap_usd"] >= 0 for row in rows)
    assert rows == [row for row in everything if row["export_gap_usd"] >= 0]
    assert len(rows) < len(everything)


def test_limit_keeps_the_top_of_the_ranking(client):
    everything = _ranked(client, "/opportunities/by-market/JPN")

    assert _ranked(client, "/opportunities/by-market/JPN", limit=2) == everything[:2]


def _keys(value) -> set[str]:
    """Every key in a JSON body, at any depth."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def _every_response(client) -> list:
    """One body from every endpoint (USD only: none may carry £ values or a £ rate)."""
    gets = [
        "/health",
        "/products",
        "/markets",
        "/opportunities/rank-by",
        f"/opportunities/by-product/{FIXTURE_HS6}",
        "/opportunities/by-market/JPN",
        "/evaluation",
        f"/explain/{FIXTURE_HS6}/JPN",
        "/scenarios/presets",
        "/scenarios/thresholds",
    ]
    bodies = [client.get(path).json() for path in gets]
    bodies.append(client.post("/scenarios", json={"add_agreements": ["IND"]}).json())
    return bodies


def test_no_response_carries_gbp_values_or_a_gbp_rate(client):
    for body in _every_response(client):
        keys = {key.lower() for key in _keys(body)}
        assert not {key for key in keys if "gbp" in key}
        assert "£" not in str(body)


def test_min_gap_usd_filters_rankings(client):
    everything = _ranked(client, "/opportunities/by-market/JPN")
    threshold_usd = 5000

    rows = _ranked(client, "/opportunities/by-market/JPN", min_gap_usd=threshold_usd)

    assert rows == [row for row in everything if row["export_gap_usd"] >= threshold_usd]
    assert 0 < len(rows) < len(everything)


def test_the_old_gbp_gap_filter_is_gone(client):
    response = client.get("/opportunities/by-market/JPN", params={"min_gap_gbp": 5000})

    assert response.status_code == 422


def test_rows_show_whether_a_uk_trade_agreement_is_in_force_with_the_market(client):
    # The fixture's UK Trade Agreements: Japan from 2022, Canada from 2021; none with India.
    rows = _ranked(client, f"/opportunities/by-product/{FIXTURE_HS6}")

    assert {row["iso3"]: row["trade_agreement"] for row in rows} == {"JPN": True, "CAN": True, "IND": False}


def test_every_row_of_a_market_shows_its_trade_agreement(client):
    assert {row["trade_agreement"] for row in _ranked(client, "/opportunities/by-market/JPN")} == {True}
    assert {row["trade_agreement"] for row in _ranked(client, "/opportunities/by-market/IND")} == {False}


def test_explain_returns_factors_history_and_the_baseline_it_explains(client):
    response = client.get(f"/explain/{FIXTURE_HS6}/JPN")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "artefact_version",
        "baseline_year",
        "product",
        "market",
        "fold",
        "base_value",
        "log_export_potential",
        "export_potential_usd",
        "actual_exports_usd",
        "export_gap_usd",
        "factors",
        "history",
    }
    assert body["product"]["hs6"] == FIXTURE_HS6
    assert body["market"] == {"iso3": "JPN", "name": "Japan", "region": "East Asia & Pacific"}
    assert set(body["factors"][0]) == {"label", "value_text", "contribution", "multiplier", "direction"}
    assert [point["year"] for point in body["history"]] == sorted(point["year"] for point in body["history"])


def test_explain_matches_the_ranking_row_of_the_pair(client):
    explained = client.get(f"/explain/{FIXTURE_HS6}/JPN").json()
    ranked = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()["opportunities"]
    row = next(r for r in ranked if r["iso3"] == "JPN")

    assert explained["export_potential_usd"] == pytest.approx(row["export_potential_usd"], rel=1e-3)
    assert explained["export_gap_usd"] == pytest.approx(row["export_gap_usd"], rel=1e-3, abs=1e-3)
    total = explained["base_value"] + sum(f["contribution"] for f in explained["factors"])
    assert total == pytest.approx(explained["log_export_potential"], abs=1e-4)


def test_explain_directions_follow_the_sign_of_each_contribution(client):
    factors = client.get(f"/explain/{FIXTURE_HS6}/JPN").json()["factors"]

    for factor in factors:
        expected = "raised" if factor["contribution"] > 0 else "lowered" if factor["contribution"] < 0 else "no effect"
        assert factor["direction"] == expected


@pytest.mark.parametrize("path", ["/explain/999999/JPN", f"/explain/{FIXTURE_HS6}/ZZZ"])
def test_explain_unknown_pair_returns_404_with_a_message(client, path):
    response = client.get(path)

    assert response.status_code == 404
    assert response.json()["detail"]


def test_explain_needs_loaded_artefacts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/explain/{FIXTURE_HS6}/JPN")

    assert response.status_code == 503


def test_evaluation_includes_lift_and_the_event_backtest(client):
    body = client.get("/evaluation").json()

    lift = body["lift"]
    assert lift["lift"] is not None and len(lift["ci"]) == 2 and "tendency" in lift["text"]
    event = body["event_backtest"]["events"][0]
    direction = event["directions"][0]
    assert direction["controls"] and direction["pooled"]["observed"] is not None
    assert direction["pooled"]["predicted"] is not None
    assert body["event_backtest"]["text"] and body["event_backtest"]["caveat"]
