"""Gravity and WDI features: country mapping, the missing-value rule and the build entry point."""

import polars as pl

from exportspredictor import gravity


def _baci_countries():
    return pl.DataFrame(
        {
            "country_code": [826, 251, 490, 842, 999],
            "country_name": ["United Kingdom", "France", "Other Asia, nes", "USA", "Atlantis"],
            "country_iso3": ["GBR", "FRA", "S19", "USA", "ATL"],
        }
    )


def test_baci_codes_map_to_iso3_with_taiwan_override_and_unresolved_listed():
    mapping = gravity.map_baci_countries(
        _baci_countries(), gravity_iso3={"GBR", "FRA", "TWN", "USA"}, wdi_iso3={"GBR", "FRA", "USA"}
    )

    by_code = {row["baci_code"]: row for row in mapping.iter_rows(named=True)}
    assert by_code[826]["iso3"] == "GBR"
    assert by_code[490]["iso3"] == "TWN"  # BACI's "Other Asia, nes" is Taiwan
    assert by_code[490]["in_gravity"] and not by_code[490]["in_wdi"]
    assert by_code[999]["iso3"] == "ATL"
    assert not by_code[999]["in_gravity"]
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
