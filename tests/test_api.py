"""API contract tests: FastAPI TestClient against fixture artefacts (the main seam)."""

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
    with TestClient(create_app(tmp_path)) as client:
        before = client.get("/health")
        store.save(fixture_artefacts, tmp_path)
        after = client.get("/health")

    assert before.status_code == 503
    assert before.json()["ready"] is False
    assert after.status_code == 200
    assert after.json()["ready"] is True


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
