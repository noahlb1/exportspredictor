"""Gravity and WDI features: country mapping, the missing-value rule and the build entry point."""

import json
import zipfile

import polars as pl

from exportspredictor import gravity


def _baci_countries():
    return pl.DataFrame(
        {
            "country_code": [826, 251, 490, 842, 652, 999],
            "country_name": ["United Kingdom", "France", "Other Asia, nes", "USA", "Saint Barthelemy", "Atlantis"],
            "country_iso3": ["GBR", "FRA", "S19", "USA", "BLM", "ATL"],
        }
    )


def test_baci_codes_map_to_iso3_with_taiwan_override_and_unresolved_listed():
    mapping = gravity.map_baci_countries(
        _baci_countries(), gravity_iso3={"GBR", "FRA", "TWN", "USA", "GLP"}, wdi_iso3={"GBR", "FRA", "USA"}
    )

    by_code = {row["baci_code"]: row for row in mapping.iter_rows(named=True)}
    assert by_code[826]["iso3"] == "GBR"
    assert by_code[490]["iso3"] == "TWN"  # BACI's "Other Asia, nes" is Taiwan
    assert by_code[490]["in_gravity"] and not by_code[490]["in_wdi"]
    assert by_code[999]["iso3"] == "ATL"
    assert not by_code[999]["in_gravity"]
    # Saint Barthelemy is not in Gravity; its pair features come from Guadeloupe, next door.
    assert (by_code[652]["iso3"], by_code[652]["gravity_proxy"], by_code[652]["in_gravity"]) == ("BLM", "GLP", True)
    assert gravity.unresolved(mapping)["baci_code"].to_list() == [999]


def _source(rows):
    return pl.DataFrame(rows, schema=["iso3", "year", "gdp_usd", "population"], orient="row")


def test_gdp_and_population_follow_the_source_order_then_interpolate_then_carry():
    wdi = _source([("AAA", 2012, 100.0, 10.0), ("AAA", 2014, 300.0, 30.0), ("CCC", 2012, 50.0, None)])
    imf = _source([("AAA", 2013, 222.0, 22.0), ("AAA", 2014, 999.0, 99.0)])
    cepii = _source([("CCC", 2013, None, 5.0)])

    filled = gravity.fill_country_year(
        {"wdi": wdi, "imf_weo": imf, "cepii_gravity": cepii}, iso3=["AAA", "CCC", "DDD"], years=[2012, 2013, 2014]
    )

    rows = {(r["iso3"], r["year"]): r for r in filled.iter_rows(named=True)}
    assert len(rows) == 9
    # The first source holding a value wins.
    assert (rows["AAA", 2013]["gdp_usd"], rows["AAA", 2013]["gdp_source"]) == (222.0, "imf_weo")
    assert (rows["AAA", 2014]["gdp_usd"], rows["AAA", 2014]["gdp_source"]) == (300.0, "wdi")
    # Each variable is filled on its own: GDP is carried forward, population comes from a source.
    assert (rows["CCC", 2013]["gdp_usd"], rows["CCC", 2013]["gdp_source"]) == (50.0, "carried")
    assert (rows["CCC", 2013]["population"], rows["CCC", 2013]["population_source"]) == (5.0, "cepii_gravity")
    # Years before the first observation take the nearest observed value.
    assert (rows["CCC", 2012]["population"], rows["CCC", 2012]["population_source"]) == (5.0, "carried")
    assert (rows["CCC", 2014]["population"], rows["CCC", 2014]["population_source"]) == (5.0, "carried")
    # A country with no data at all stays missing.
    assert rows["DDD", 2013]["gdp_usd"] is None
    assert rows["DDD", 2013]["gdp_source"] is None


def test_gaps_between_observed_years_are_interpolated():
    wdi = _source([("AAA", 2012, 100.0, 10.0), ("AAA", 2015, 400.0, 40.0)])

    filled = gravity.fill_country_year({"wdi": wdi}, iso3=["AAA"], years=[2012, 2013, 2014, 2015]).sort("year")

    assert filled["gdp_usd"].to_list() == [100.0, 200.0, 300.0, 400.0]
    assert filled["gdp_source"].to_list() == ["wdi", "interpolated", "interpolated", "wdi"]


def _gravity_rows(rows):
    columns = ["year", "iso3_o", "iso3_d", "country_exists_o", "country_exists_d", "dist", "contig"]
    columns += ["comlang_off", "col_dep_ever", "comcol"]
    return pl.DataFrame(rows, schema=columns, orient="row")


def test_pair_features_take_each_pairs_latest_known_value_among_existing_countries():
    raw = _gravity_rows(
        [
            (2020, "GBR", "IRL", 1, 1, 460.0, 1, 1, 1, 0),
            (2021, "GBR", "IRL", 1, 1, 463.0, 1, None, 1, 0),
            (2021, "SDN", "GBR", 1, 1, 5000.0, 0, 1, 1, 0),
            (2021, "SDN", "GBR", 0, 1, 5100.0, 0, 0, 0, 0),  # the pre-2011 Sudan: no longer exists
        ]
    )

    pairs = gravity.gravity_pairs(raw.lazy())

    rows = {(r["iso3_o"], r["iso3_d"]): r for r in pairs.iter_rows(named=True)}
    assert set(rows) == {("GBR", "IRL"), ("SDN", "GBR")}
    assert rows["GBR", "IRL"] == {
        "iso3_o": "GBR",
        "iso3_d": "IRL",
        "distance_km": 463.0,
        "contiguous": 1,
        "common_language": 1,
        "colonial_tie": 1,
        "common_colonizer": 0,
    }
    assert rows["SDN", "GBR"]["distance_km"] == 5000.0


def test_a_proxied_country_takes_its_proxys_pair_features_in_both_directions():
    raw = _gravity_rows(
        [
            (2021, "GLP", "FRA", 1, 1, 6800.0, 0, 1, 1, 0),
            (2021, "FRA", "GLP", 1, 1, 6800.0, 0, 1, 1, 0),
        ]
    )

    pairs = gravity.gravity_pairs(raw.lazy(), proxies={"BLM": "GLP"})

    rows = {(r["iso3_o"], r["iso3_d"]): r["distance_km"] for r in pairs.iter_rows(named=True)}
    assert rows == {("GLP", "FRA"): 6800.0, ("FRA", "GLP"): 6800.0, ("BLM", "FRA"): 6800.0, ("FRA", "BLM"): 6800.0}


def _write_sources(data_dir):
    """Tiny copies of every raw file, laid out as the download leaves them."""
    (data_dir / "raw" / "gravity").mkdir(parents=True)
    gravity_csv = _gravity_rows(
        [
            (2011, "GBR", "FRA", 1, 1, 340.0, 0, 0, 0, 0),
            (2021, "GBR", "FRA", 1, 1, 344.0, 0, 0, 0, 0),
            (2021, "FRA", "GBR", 1, 1, 344.0, 0, 0, 0, 0),
            (2021, "GBR", "TWN", 1, 1, 9800.0, 0, 0, 0, 0),
            (2021, "TWN", "GBR", 1, 1, 9800.0, 0, 0, 0, 0),
        ]
    ).with_columns(
        gdp_o=pl.when(pl.col("iso3_o") == "FRA").then(2_900_000_000.0),  # thousand USD
        pop_o=pl.when(pl.col("iso3_o") == "FRA").then(67_000.0),  # thousands
        fta_wto=pl.lit(0),
    )
    with zipfile.ZipFile(data_dir / "raw" / "gravity" / gravity.GRAVITY_ZIP, "w") as archive:
        archive.writestr(gravity.GRAVITY_CSV, gravity_csv.write_csv())

    (data_dir / "raw" / "wdi").mkdir(parents=True)
    for indicator, value in [("NY.GDP.MKTP.CD", 3.0e12), ("SP.POP.TOTL", 68e6)]:
        records = [
            {"countryiso3code": "GBR", "date": str(year), "value": value} for year in range(2012, 2025)
        ]
        records.append({"countryiso3code": "", "date": "2024", "value": 1.0})
        (data_dir / "raw" / "wdi" / f"{indicator}.json").write_text(json.dumps([{"page": 1}, records]))

    (data_dir / "raw" / "imf_weo").mkdir(parents=True)
    for indicator, value in [("NGDPD", 800.0), ("LP", 23.4)]:
        payload = {"values": {indicator: {"TWN": {str(year): value for year in range(2012, 2025)}}}}
        (data_dir / "raw" / "imf_weo" / f"{indicator}.json").write_text(json.dumps(payload))

    (data_dir / "raw" / "baci").mkdir(parents=True)
    (data_dir / "raw" / "baci" / "country_codes_V202601.csv").write_text(
        "country_code,country_name,country_iso2,country_iso3\n"
        "826,United Kingdom,GB,GBR\n251,France,FR,FRA\n490,\"Other Asia, nes\",,S19\n"
    )


def test_build_writes_country_map_pair_features_and_country_years_for_2012_to_2024(tmp_path):
    _write_sources(tmp_path)

    outputs = gravity.build(tmp_path, download_missing=False)

    country_map = pl.read_parquet(outputs["country_map"])
    assert dict(zip(country_map["baci_code"], country_map["iso3"])) == {826: "GBR", 251: "FRA", 490: "TWN"}
    assert gravity.unresolved(country_map).is_empty()

    pairs = pl.read_parquet(outputs["gravity_pairs"])
    assert pairs.filter(iso3_o="GBR", iso3_d="FRA")["distance_km"].item() == 344.0

    country_year = pl.read_parquet(outputs["country_year"])
    assert country_year.height == 3 * 13  # every BACI country, every year 2012-2024
    assert country_year["gdp_usd"].null_count() == 0
    by_key = {(r["iso3"], r["year"]): r for r in country_year.iter_rows(named=True)}
    assert (by_key["GBR", 2024]["gdp_usd"], by_key["GBR", 2024]["gdp_source"]) == (3.0e12, "wdi")
    assert (by_key["TWN", 2024]["gdp_usd"], by_key["TWN", 2024]["gdp_source"]) == (800.0e9, "imf_weo")
    assert by_key["TWN", 2024]["population"] == 23.4e6
    # France is in neither WDI nor IMF here, so Gravity's 2021 value (thousands) is used, then carried.
    assert (by_key["FRA", 2021]["gdp_usd"], by_key["FRA", 2021]["gdp_source"]) == (2.9e12, "cepii_gravity")
    assert (by_key["FRA", 2024]["population"], by_key["FRA", 2024]["population_source"]) == (67e6, "carried")

    # The Gravity extract keeps every column from 2012 on, for the policy cross-check.
    extract = pl.read_parquet(outputs["gravity_extract"])
    assert extract["year"].min() == 2021 and "fta_wto" in extract.columns  # the 2011 row is dropped
