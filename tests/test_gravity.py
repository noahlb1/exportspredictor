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
    # No secondary source is on disk and no country lacks a variable: the filled table is the same.
    assert pl.read_parquet(outputs["country_year_filled"]).equals(country_year)

    # The Gravity extract keeps every column from 2012 on, for the policy cross-check.
    extract = pl.read_parquet(outputs["gravity_extract"])
    assert extract["year"].min() == 2021 and "fta_wto" in extract.columns  # the 2011 row is dropped


# --- Gravity gaps: secondary sources for Markets the primary sources miss (#126) ----------


def test_secondary_sources_fill_only_the_variables_no_primary_source_holds():
    primary = gravity.fill_country_year(
        {"wdi": _source([("AAA", 2012, 100.0, 10.0), ("PRK", 2012, None, 24e6), ("PRK", 2013, None, 25e6)])},
        iso3=["AAA", "PRK", "GIB", "IOT"], years=[2012, 2013, 2014],
    )
    secondary = {
        "unsd_ama": _source([("AAA", 2012, 999.0, 999.0), ("PRK", 2012, 15e9, 1.0), ("PRK", 2014, 17e9, 1.0)]),
        "cia_factbook": _source([("GIB", 2013, 2.0e9, None), ("PRK", 2012, 1.0, 1.0)]),
    }

    filled = gravity.fill_gaps(primary, secondary, years=[2012, 2013, 2014])

    rows = {(r["iso3"], r["year"]): r for r in filled.iter_rows(named=True)}
    # A country a primary source covers keeps its values, even where a secondary source differs.
    assert (rows["AAA", 2012]["gdp_usd"], rows["AAA", 2012]["gdp_source"]) == (100.0, "wdi")
    assert (rows["AAA", 2013]["gdp_usd"], rows["AAA", 2013]["gdp_source"]) == (100.0, "carried")
    # North Korea's GDP comes from the first secondary source, with its population kept from WDI.
    assert (rows["PRK", 2012]["gdp_usd"], rows["PRK", 2012]["gdp_source"]) == (15e9, "unsd_ama")
    assert (rows["PRK", 2013]["gdp_usd"], rows["PRK", 2013]["gdp_source"]) == (16e9, "interpolated (unsd_ama)")
    assert (rows["PRK", 2013]["population"], rows["PRK", 2013]["population_source"]) == (25e6, "wdi")
    # One estimate is carried to every year, and says where it came from.
    assert [rows["GIB", y]["gdp_usd"] for y in (2012, 2013, 2014)] == [2.0e9] * 3
    assert rows["GIB", 2014]["gdp_source"] == "carried (cia_factbook)"
    # With no source anywhere, the value stays missing.
    assert rows["GIB", 2013]["population"] is None
    assert rows["IOT", 2013]["gdp_usd"] is None and rows["IOT", 2013]["gdp_source"] is None


def test_factbook_figures_are_read_in_dollars_and_persons_with_their_year():
    assert gravity.factbook_figure("$2.044 billion (2014 est.)") == (2.044e9, 2014)
    assert gravity.factbook_figure("$206.4 million (2015 est.)") == (206.4e6, 2015)
    assert gravity.factbook_figure("$7,711,583 (2017 est.)") == (7_711_583.0, 2017)
    assert gravity.factbook_figure("29,733 (2025 est.)") == (29_733.0, 2025)
    assert gravity.factbook_figure("3,142 (2021)") == (3_142.0, 2021)
    assert gravity.factbook_figure("no permanent inhabitants") is None
    assert gravity.factbook_figure("NA") is None


def test_the_factbook_gives_gdp_at_the_official_exchange_rate_and_population(tmp_path):
    entries = {
        "GIB": {"Economy": {"GDP (official exchange rate)": {"text": "$2.044 billion (2014 est.)"}},
                "People and Society": {"Population": {"total": {"text": "29,733 (2025 est.)"}}}},
        # Only a purchasing-power estimate: a different measure, so no GDP.
        "NIU": {"Economy": {"Real GDP (purchasing power parity)": {"text": "$18.7 million (2021 est.)"}},
                "People and Society": {"Population": {"total": {"text": "1,815 (2024 est.)"}}}},
        "IOT": {"People and Society": {"Population": {"total": {"text": "no permanent inhabitants"}}}},
    }
    for iso3, entry in entries.items():
        (tmp_path / f"{iso3}.json").write_text(json.dumps(entry), encoding="utf-8")

    table = gravity.read_factbook(tmp_path)

    rows = {(r["iso3"], r["year"]): (r["gdp_usd"], r["population"]) for r in table.iter_rows(named=True)}
    assert rows == {("GIB", 2014): (2.044e9, None), ("GIB", 2025): (None, 29_733.0), ("NIU", 2024): (None, 1_815.0)}


def test_unsd_ama_gives_gdp_and_population_by_un_country_code():
    def sheet(header, rows):
        width = len(header)
        return pl.DataFrame([["Title", *[None] * (width - 1)], header, *rows], schema=[f"c{i}" for i in range(width)],
                            orient="row")

    gdp = sheet(["CountryID", "Country", "IndicatorName", "2012", "2013"],
                [["500", "Montserrat", "Gross Domestic Product (GDP)", "63170000", "59726666.5"],
                 ["500", "Montserrat", "Exports of goods and services", "1", "1"],
                 ["999", "Atlantis", "Gross Domestic Product (GDP)", "5", "5"]])
    # The population sheet has a currency column before the measure.
    population = sheet(["CountryID", "Country", "Currency", "Measure", "2012", "2013"],
                       [["500", "Montserrat", None, "Population", "4796", "4753"],
                        ["500", "Montserrat", "East Caribbean Dollar", "AMA exchange rate", "2.7", "2.7"]])

    table = gravity.unsd_ama(gdp, population, {500: "MSR"})

    rows = {(r["iso3"], r["year"]): (r["gdp_usd"], r["population"]) for r in table.sort("year").iter_rows(named=True)}
    assert rows == {("MSR", 2012): (63_170_000.0, 4796.0), ("MSR", 2013): (59_726_666.5, 4753.0)}


def test_cbs_statline_gives_the_caribbean_netherlands_gdp_in_dollars(tmp_path):
    path = tmp_path / "84789ENG.json"
    path.write_text(json.dumps({"value": [
        {"CarribeanNetherlands": "CN01  ", "Periods": "2012JJ00", "GDPValueAtCurrentPrices_1": 593},
        {"CarribeanNetherlands": "CN01  ", "Periods": "2024JJ00", "GDPValueAtCurrentPrices_1": 941},
        {"CarribeanNetherlands": "GM9001", "Periods": "2012JJ00", "GDPValueAtCurrentPrices_1": 417},  # Bonaire alone
    ]}), encoding="utf-8")

    table = gravity.read_cbs_statline(path)

    assert table.sort("year").rows() == [("BES", 2012, 593e6, None), ("BES", 2024, 941e6, None)]


def test_an_estimate_dated_just_after_the_last_year_stands_for_it_but_history_never_does():
    primary = gravity.fill_country_year({"wdi": _source([("AAA", 2012, 1.0, 1.0)])}, iso3=["AAA", "SPM", "SUN"],
                                        years=[2012, 2013])
    secondary = {
        # Saint Pierre and Miquelon's only population figure is a 2015 estimate (here, two years late).
        "cia_factbook": _source([("SPM", 2015, None, 5_070.0), ("SPM", 2016, None, 9.0)]),
        # The Soviet Union's last GDP is from 1990: it must not become a 2012 GDP.
        "unsd_ama": _source([("SUN", 1990, 1e12, 2.9e8)]),
    }

    filled = gravity.fill_gaps(primary, secondary, years=[2012, 2013])

    spm = filled.filter(iso3="SPM").sort("year")
    assert spm["population"].to_list() == [5_070.0, 5_070.0]
    assert filled.filter(iso3="SUN")["gdp_usd"].to_list() == [None, None]
