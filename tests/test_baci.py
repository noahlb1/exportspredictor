"""BACI download and conversion, run against a tiny zip shaped like the real release."""

import json
import zipfile

import polars as pl
import pytest

from exportspredictor import baci

COUNTRIES = """country_code,country_name,country_iso2,country_iso3
392,Japan,JP,JPN
490,"Other Asia, nes",,S19
826,United Kingdom,GB,GBR
276,Germany,DE,DEU
280,Fed. Rep. of Germany (...1990),DE,DEU
384,CÃ´te d'Ivoire,CI,CIV
"""
PRODUCTS = """code,description
010121,"Horses: live, pure-bred breeding animals"
220830,Whiskies
851712,Telephones for cellular networks or for other wireless networks
"""
# BACI writes `k` with its leading zero in some releases and without it in others.
YEAR_FILES = {
    2012: "t,i,j,k,v,q\n2012,826,392,220830,1500.5,12.0\n2012,276,392,010121,0.032,\n",
    2013: "t,i,j,k,v,q\n2013,826,392,10121,2.0,1.0\n2013,490,826,851712,900.25,3.5\n2013,276,826,851712,10.0,\n",
}
WORLD_BANK = [
    {"page": 1},
    [
        {"id": "JPN", "name": "Japan", "region": {"value": "East Asia & Pacific "}, "incomeLevel": {"value": "High income"}},
        {"id": "GBR", "name": "United Kingdom", "region": {"value": "Europe & Central Asia"}, "incomeLevel": {"value": "High income"}},
        {"id": "DEU", "name": "Germany", "region": {"value": "Europe & Central Asia"}, "incomeLevel": {"value": "High income"}},
        {"id": "EUU", "name": "European Union", "region": {"value": "Aggregates"}, "incomeLevel": {"value": "Aggregates"}},
    ],
]


@pytest.fixture
def sources(tmp_path):
    zip_path = tmp_path / "src" / f"BACI_HS12_{baci.BACI_VERSION}.zip"
    zip_path.parent.mkdir()
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"country_codes_{baci.BACI_VERSION}.csv", COUNTRIES)
        archive.writestr(f"product_codes_HS12_{baci.BACI_VERSION}.csv", PRODUCTS)
        for year, text in YEAR_FILES.items():
            archive.writestr(f"BACI_HS12_Y{year}_{baci.BACI_VERSION}.csv", text)
    regions = tmp_path / "src" / "countries.json"
    regions.write_text(json.dumps(WORLD_BANK), encoding="utf-8")
    return {"baci_url": zip_path.as_uri(), "countries_url": regions.as_uri()}


@pytest.fixture
def data_root(tmp_path, sources):
    root = tmp_path / "data"
    baci.prepare(root, **sources)
    return root


def test_flows_cover_every_year_in_the_release_with_compact_types(data_root):
    flows = baci.scan_flows(data_root).collect()

    assert sorted(flows["year"].unique()) == [2012, 2013]
    assert flows.columns == ["year", "exporter", "importer", "hs6", "value_usd"]
    assert flows.schema["year"] == pl.Int16
    assert flows.schema["value_usd"] == pl.Float64
    for column in ("exporter", "importer", "hs6"):
        assert isinstance(flows.schema[column], pl.Enum)


def test_product_codes_keep_their_leading_zeros(data_root):
    flows = baci.scan_flows(data_root).collect()

    horses = flows.filter(pl.col("hs6") == "010121").sort("year")
    assert horses["hs6"].cast(pl.String).to_list() == ["010121", "010121"]


def test_flows_are_in_usd_with_countries_as_iso3(data_root):
    flows = baci.scan_flows(data_root).filter(pl.col("year") == 2013).collect().sort("value_usd")

    assert flows.select(pl.col("exporter", "importer").cast(pl.String), "value_usd").rows() == [
        ("GBR", "JPN", 2000.0),
        ("DEU", "GBR", 10000.0),
        ("S19", "GBR", 900250.0),
    ]


def test_product_lookup_has_descriptions_and_sectors(data_root):
    products = baci.products(data_root)

    assert products.row(0, named=True) == {
        "hs6": "010121",
        "description": "Horses: live, pure-bred breeding animals",
        "sector": "01",
    }
    assert products.height == 3


def test_market_lookup_has_names_regions_and_income_levels(data_root):
    markets = {row["iso3"]: row for row in baci.markets(data_root).iter_rows(named=True)}

    assert set(markets) == {"JPN", "S19", "GBR", "DEU", "CIV"}  # historical codes are dropped
    assert markets["JPN"]["region"] == "East Asia & Pacific"
    assert markets["GBR"]["name"] == "United Kingdom"
    assert markets["GBR"]["income_level"] == "High income"
    assert markets["CIV"]["region"] == "Unclassified"  # not in this test's World Bank list
    # BACI's file double-encodes accented names as UTF-8; the lookup repairs them.
    assert markets["CIV"]["name"] == "Côte d'Ivoire"
    # BACI's "Other Asia, nes" stands in for Taiwan, which the World Bank does not list.
    assert markets["S19"]["region"] == "East Asia & Pacific"


def test_rerunning_prepare_skips_work_already_done(tmp_path, sources, data_root):
    written = sorted(p for p in data_root.rglob("*") if p.is_file())
    stamps = {p: p.stat().st_mtime_ns for p in written}

    report = baci.prepare(data_root, **sources)

    assert sorted(p for p in data_root.rglob("*") if p.is_file()) == written
    assert {p: p.stat().st_mtime_ns for p in written} == stamps
    assert report == baci.PrepareReport(downloaded=[], extracted=[], converted_years=[], built_lookups=False)


def test_prepare_finishes_a_partial_conversion(sources, data_root):
    missing = baci.flows_dir(data_root) / "flows_2013.parquet"
    missing.unlink()

    report = baci.prepare(data_root, **sources)

    assert report.converted_years == [2013]
    assert missing.exists()
