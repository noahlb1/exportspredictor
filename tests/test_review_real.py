"""The Review Pack on the real artefacts and BACI Flows, when they are present (skipped otherwise).

Point `EXPORTSPREDICTOR_ARTEFACTS_DIR` and `EXPORTSPREDICTOR_DATA_DIR` at the real roots to run these.
"""

import polars as pl
import pytest

from exportspredictor import artefacts as store
from exportspredictor import baci, review
from exportspredictor.paths import artefacts_dir, data_dir


@pytest.fixture(scope="module")
def pack():
    try:
        bundle = store.load_latest(artefacts_dir())
    except store.ArtefactsNotFound:
        pytest.skip("No real artefacts under EXPORTSPREDICTOR_ARTEFACTS_DIR")
    except store.UnreadableArtefacts as error:
        pytest.skip(f"The newest real artefacts are not artefact format {store.ARTEFACT_FORMAT}: {error}")
    if bundle.metadata.get("source", "fixture") == "fixture":
        pytest.skip("The artefacts are fixture artefacts")
    country_map = data_dir() / "derived" / "country_map.parquet"
    if not ((baci.flows_dir() / f"flows_{bundle.baseline_year}.parquet").exists() and country_map.exists()):
        pytest.skip("No BACI Flows for the Baseline year, or no country map, under EXPORTSPREDICTOR_DATA_DIR")
    flows = baci.scan_flows()
    imports = review.market_total_imports(flows, bundle.baseline_year, country_map=pl.read_parquet(country_map))
    return review.build(bundle, imports, review.SEED)


def test_every_table_of_the_real_review_pack_is_non_empty_at_its_documented_size(pack):
    assert set(pack) == set(review.TABLES)
    for name, table in pack.items():
        assert not table.is_empty(), name
    assert pack["top_opportunities"].height == review.TOP
    assert pack["top_opportunities_by_market"].height == review.GROUPS * review.GROUP_TOP
    assert pack["top_opportunities_by_product"].height == review.GROUPS * review.GROUP_TOP
    assert pack["pair_sample"].height == review.SAMPLE_SIZE


def test_baci_gives_total_imports_for_every_market_in_the_real_review_pack(pack):
    covered = set(pack["market_total_imports"]["iso3"])
    for name, table in pack.items():
        if "iso3" in table.columns:
            assert set(table["iso3"]) <= covered, (name, set(table["iso3"]) - covered)
