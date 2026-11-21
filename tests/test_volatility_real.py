"""Volatile and one-off Flagged Products on the served real artefact, when it is present (skipped otherwise).

The served version must have been marked with the volatility rule (`python -m exportspredictor.pipeline recaveat`).
"""

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.paths import artefacts_dir
from exportspredictor.volatility import METADATA_KEY, ONE_OFF_PRODUCTS


@pytest.fixture(scope="module")
def bundle():
    with TestClient(create_app(artefacts_dir())) as client:
        bundle = client.app.state.artefacts
        if bundle is None:
            pytest.skip(f"No real format-{store.ARTEFACT_FORMAT} artefacts served: {client.app.state.load_error}")
        if bundle.metadata.get("source", "fixture") == "fixture":
            pytest.skip("The artefacts are fixture artefacts")
        yield bundle


def test_the_rule_read_only_years_before_the_validation_year(bundle):
    rule = bundle.metadata[METADATA_KEY]

    assert rule["years"] and max(rule["years"]) < bundle.metadata["validation_year"]
    assert rule["threshold"] is not None


def test_every_named_one_off_product_is_a_flagged_product_with_a_reason(bundle):
    named = bundle.baseline.filter(pl.any_horizontal(pl.col("hs6").str.starts_with(code) for code in ONE_OFF_PRODUCTS))

    assert named.height > 0
    assert named["flagged_product"].all() and named["flagged_product_reason"].is_not_null().all()


def test_the_flagged_products_are_the_recorded_list_and_sector_71(bundle):
    flagged = set(bundle.baseline.filter("flagged_product")["hs6"])
    recorded = set(bundle.metadata[METADATA_KEY]["products"])

    assert recorded <= flagged
    assert all(hs6.startswith("71") for hs6 in flagged - recorded)
