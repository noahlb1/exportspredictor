"""The API serving the real v2 artefacts, when they are present (skipped otherwise).

`scripts/wt.ps1` points `EXPORTSPREDICTOR_ARTEFACTS_DIR` and `EXPORTSPREDICTOR_DATA_DIR` at
the real roots. These tests go through the API, as the app does, plus the real derived data
the artefacts were built from.
"""

import json

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor import baci, features, scope
from exportspredictor.paths import artefacts_dir, data_dir

GOLD, CRUDE_OIL = "710812", "270900"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(artefacts_dir())) as client:
        bundle = client.app.state.artefacts
        if bundle is None:
            pytest.skip(f"No real format-{store.ARTEFACT_FORMAT} artefacts served: {client.app.state.load_error}")
        if bundle.metadata.get("source", "fixture") == "fixture":
            pytest.skip("The artefacts are fixture artefacts")
        yield client


@pytest.fixture(scope="module")
def real(client):
    """The artefact bundle the API serves (loaded once, by the API)."""
    return client.app.state.artefacts


def test_health_reports_format_4_and_the_gap_period_of_the_served_artefacts(client, real):
    body = client.get("/health").json()

    assert body["ready"] is True
    assert body["artefact_version"] == real.version
    assert body["artefact_format"] == 4
    assert body["gap_years"] == [2022, 2024]
    assert body["baseline_year"] == 2024


def test_every_excluded_product_is_listed_with_a_reason_and_never_ranked(client):
    excluded = client.get("/products/excluded").json()
    codes = {row["hs6"] for row in excluded}

    assert {GOLD, CRUDE_OIL} <= codes
    assert all(row["reason"] for row in excluded)
    for hs6 in sorted(codes):
        response = client.get(f"/opportunities/by-product/{hs6}")
        assert response.status_code == 404, hs6
        assert "Excluded Product" in response.json()["detail"], hs6


def test_the_served_baseline_holds_no_excluded_product(client, real):
    excluded = pl.DataFrame({"hs6": [row["hs6"] for row in client.get("/products/excluded").json()]})

    assert real.baseline.height > 1_000_000
    assert real.baseline.join(excluded, on="hs6").is_empty()
    assert real.history.join(excluded, on="hs6").is_empty()
    for sector in ("27", "71", "97"):  # the Sectors the Excluded Products come from
        rows = client.get("/opportunities/top", params={"sector": sector, "limit": 1000}).json()["opportunities"]
        assert rows and not {row["hs6"] for row in rows} & set(excluded["hs6"]), sector


def test_the_exporter_scope_is_the_top_100_economies_the_features_were_built_on(real):
    country_year = data_dir() / "derived" / "country_year.parquet"
    built = features.features_dir() / "features.json"
    if not (country_year.exists() and built.exists()):
        pytest.skip(f"The real derived tables are not under {data_dir()}")
    served = real.metadata["exporter_scope"]

    assert len(served) == scope.TOP_EXPORTERS
    assert baci.UK in served
    assert served == scope.exporter_scope(pl.read_parquet(country_year), uk=baci.UK)
    assert served == json.loads(built.read_text(encoding="utf-8"))["exporter_scope"]


def test_every_market_view_states_its_market_reach_from_the_year_before_the_baseline(client, real):
    # ADR 0010: the UK reaches France far more widely than Iran; each sentence is learned from 2023's UK Flows.
    france = client.get("/opportunities/by-market/FRA").json()["market_reach"]
    iran = client.get("/opportunities/by-market/IRN").json()["market_reach"]

    assert france["year"] == iran["year"] == real.baseline_year - 1
    assert france["offset"] > 0 > iran["offset"]
    assert france["products_sent"] > france["products_expected"] and iran["products_sent"] < iran["products_expected"]
    detail = client.get("/explain/220830/FRA").json()["market_reach"]
    assert detail["text"] == france["text"]
