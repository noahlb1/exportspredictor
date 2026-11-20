"""Sanctioned Markets and Restricted Pairs on the served real artefact, when it is present (skipped otherwise).

The served version must have been marked with the declared tiers (`python -m exportspredictor.pipeline recaveat`).
"""

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.paths import artefacts_dir
from exportspredictor.restrictions import PROHIBITIONS, SANCTIONS_REGIMES

WHISKY = "220830"


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
def baseline(client) -> pl.DataFrame:
    return client.app.state.artefacts.baseline


def test_no_sanctioned_market_row_lacks_its_caveat(client, baseline):
    year = client.app.state.artefacts.baseline_year
    sanctioned = [regime.iso3 for regime in SANCTIONS_REGIMES if regime.covers(year)]

    rows = baseline.filter(pl.col("iso3").is_in(sanctioned))

    assert rows.height > 0 and set(rows["iso3"]) >= {"RUS", "BLR"}
    assert rows["sanctioned_market"].all() and rows["sanctioned_market_reason"].is_not_null().all()
    assert baseline.filter(~pl.col("iso3").is_in(sanctioned))["sanctioned_market"].sum() == 0


def test_no_restricted_pair_row_lacks_its_caveat(baseline):
    covered = pl.any_horizontal(
        (pl.col("iso3") == prohibition.iso3) & pl.col("hs6").str.starts_with(code)
        for prohibition in PROHIBITIONS
        for code in prohibition.hs_codes
    )

    rows = baseline.filter(covered)

    assert rows.height > 0
    assert rows["restricted_pair"].all() and rows["restricted_pair_reason"].is_not_null().all()
    assert baseline.filter(~covered)["restricted_pair"].sum() == 0


def test_whisky_to_saudi_arabia_is_a_restricted_pair_and_russia_a_sanctioned_market(client):
    whisky = client.get("/opportunities/top", params={"product": WHISKY, "market": "SAU"}).json()["opportunities"]
    russia = client.get("/opportunities/top", params={"market": "RUS", "limit": 5}).json()["opportunities"]

    assert [row["restricted_pair_reason"] for row in whisky] == ["Saudi Arabia prohibits imports of alcoholic beverages"]
    assert russia and all(
        row["sanctioned_market_reason"] == "the UK's Russia sanctions regime, broad since 2022" for row in russia
    )
