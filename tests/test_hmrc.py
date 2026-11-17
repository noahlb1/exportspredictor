"""The HMRC OTS 2023 cross-check of BACI's UK totals.

The bulk records are written in HMRC's fixed-width export layout (bulk data set technical
specification, uktradeinfo.com), so the reader is tested on the real layout. Expected
values are worked by hand from the records below.
"""

import zipfile

import polars as pl
import pytest

from exportspredictor import hmrc


def _record(comcode: str, country: str, value_gbp: int, *, line_type: str = "1", month: str = "202301") -> str:
    """One 85-character line of a BDSexpYYMM file (positions from the technical specification)."""
    line = (
        month  # 1-6 PERREF
        + line_type  # 7 TYPE
        + "202507"  # 8-13 MONTHAC
        + comcode.ljust(8)  # 14-21 COMCODE
        + "00150"  # 22-26 SITC
        + "001"  # 27-29 COD-SEQ
        + country.ljust(2)  # 30-31 COD-ALPHA
        + "007DOV"  # 32-37 port
        + "     "  # 38-42 country of origin
        + "60"  # 43-44 mode of transport
        + f"{value_gbp:012d}"  # 45-56 STAT-VALUE
        + f"{0:012d}"  # 57-68 NET-MASS
        + f"{0:012d}"  # 69-80 SUPP-UNIT
        + "0"  # 81 suppression
        + "exp"  # 82-84 flow
        + "0"  # 85 record type
    )
    assert len(line) == 85
    return line


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "bdsexp_23archive.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "BDSexp2301.txt",
            "\r\n".join(
                [
                    _record("22083011", "JP", 600),  # whisky to Japan, HS 2022 220830
                    _record("22083019", "JP", 400),  # same HS6, another CN8 (and another port)
                    _record("85176200", "FR", 1_000),  # HS 2022 851762 -> HS 2012 851762
                    _record("02------", "FR", 50, line_type="2"),  # below-threshold estimate: chapter only
                ]
            ),
        )
        z.writestr(
            "BDSexp2302.txt",
            "\r\n".join(
                [
                    _record("22083011", "JP", 1_000, month="202302"),
                    _record("99500000", "ZY", 300, month="202302"),  # low-value consignments, no country
                    _record("85249100", "QV", 200, month="202302"),  # HS 2022 only code, estimated country
                ]
            ),
        )
    return path


def test_read_exports_parses_the_fixed_width_layout_across_months(archive):
    records = hmrc.read_exports(archive)

    assert records.height == 7
    assert records["value_gbp"].sum() == 3_550
    first = records.row(0, named=True)
    assert first["month"] == 202301
    assert first["comcode"] == "22083011"
    assert first["country"] == "JP"
    assert first["value_gbp"] == 600
    assert first["line_type"] == 1


CONCORDANCE = pl.DataFrame(
    {"hs22": ["220830", "851762", "852491"], "hs12": ["220830", "851762", "901380"]}
)
MARKETS = pl.DataFrame({"country": ["JP", "FR"], "iso3": ["JPN", "FRA"]})


def test_allocate_sums_ports_and_months_to_hs12_products_and_markets_and_keeps_the_rest(archive):
    ots = hmrc.allocate(hmrc.read_exports(archive), CONCORDANCE, MARKETS)

    rows = {(r["hs6"], r["iso3"]): r["value_gbp"] for r in ots.iter_rows(named=True)}
    assert rows == {
        ("220830", "JPN"): 2_000,  # two CN8 codes, two months
        ("851762", "FRA"): 1_000,
        (None, "FRA"): 50,  # chapter-only estimate: a Market but no Product
        (None, None): 300,  # low-value consignments: neither
        ("901380", None): 200,  # HS 2022 code converted to HS 2012, country not attributed
    }


def test_market_codes_map_hmrc_country_codes_to_iso3_and_skip_world_bank_aggregates(tmp_path):
    countries = tmp_path / "countries.json"
    countries.write_text(
        '[{"page": 1}, ['
        '{"id": "JPN", "iso2Code": "JP", "region": {"value": "East Asia & Pacific"}},'
        '{"id": "EAS", "iso2Code": "Z4", "region": {"value": "Aggregates"}}'
        "]]",
        encoding="utf-8",
    )

    codes = dict(hmrc.market_codes(countries).iter_rows())

    assert codes["JP"] == "JPN"
    assert codes["TW"] == "TWN"  # not in the World Bank list
    assert codes["XS"] == "SRB"  # HMRC (Eurostat) code for Serbia
    assert "Z4" not in codes


def test_crosscheck_reports_the_difference_and_where_it_comes_from(archive):
    ots = hmrc.allocate(hmrc.read_exports(archive), CONCORDANCE, MARKETS)
    baci_uk = pl.DataFrame(
        {
            "hs6": ["220830", "851762", "030111"],
            "iso3": ["JPN", "FRA", "FRA"],
            "value_usd": [4_000.0, 1_000.0, 500.0],
        }
    )

    report = hmrc.crosscheck(ots, baci_uk, usd_per_gbp=2.0)

    assert report["ots_usd"] == 7_100
    assert report["baci_usd"] == 5_500
    assert report["difference_usd"] == 1_600
    assert report["difference_share"] == pytest.approx(1_600 / 5_500)
    assert report["no_product_usd"] == 700  # chapter-only estimate and low-value consignments
    assert report["no_market_usd"] == 400  # a Product, but no Market
    assert report["allocated_ots_usd"] == 6_000
    sectors = {r["sector"]: r for r in report["by_sector"]}
    assert sectors["85"]["difference_usd"] == 1_000
    assert sectors["03"]["difference_usd"] == -500
    assert report["by_sector"][0]["sector"] == "85"  # largest absolute difference first
    markets = {r["iso3"]: r for r in report["by_market"]}
    assert markets["FRA"]["difference_usd"] == 600
    assert markets["JPN"]["difference_usd"] == 0
    assert report["pairs"]["in_both"] == 2
    assert report["usd_per_gbp"] == 2.0
    assert "BACI" in report["text"] and "HMRC" in report["text"]


@pytest.mark.parametrize(("year", "usd_per_gbp"), [(2022, 1.2362), (2023, 1.2434)])
def test_crosscheck_converts_hmrc_pounds_at_that_years_own_rate(archive, year, usd_per_gbp):
    # Bank of England XUAAUSS annual averages, US$ per £.
    ots = hmrc.allocate(hmrc.read_exports(archive), CONCORDANCE, MARKETS)
    baci_uk = pl.DataFrame({"hs6": ["220830"], "iso3": ["JPN"], "value_usd": [4_000.0]})

    report = hmrc.crosscheck(ots, baci_uk, year=year)

    assert report["usd_per_gbp"] == usd_per_gbp
    assert report["ots_usd"] == pytest.approx(ots["value_gbp"].sum() * usd_per_gbp)
