"""Applied Tariffs from TRAINS (ADR 0006): download, HS revision, EU schedule, gap fill, carry-forward.

Small TRAINS-shaped inputs are written here by hand, so each expected rate is a worked
example. The fixture world's tariffs cover the 2021 cut-off and a post-2021 Trade Agreement.
"""

import urllib.error

import polars as pl
import pytest

from exportspredictor import tariffs

AVAILABILITY = """<?xml version="1.0" encoding="utf-8"?>
<wits:datasource xmlns:wits="http://wits.worldbank.org"><wits:dataavailability>
{reporters}
</wits:dataavailability></wits:datasource>"""
REPORTER = """<wits:reporter countrycode="{code}" iso3Code="{iso3}" isgroup="No" grouptype="N/A">
<wits:name>{iso3}</wits:name><wits:year>{year}</wits:year>
<wits:reporternernomenclature reporternernomenclaturecode="{nomen}">HS</wits:reporternernomenclature>
<wits:partnerlist>{partners}</wits:partnerlist></wits:reporter>"""
COUNTRIES = """<?xml version="1.0" encoding="utf-8"?>
<wits:datasource xmlns:wits="http://wits.worldbank.org"><wits:countries>
<wits:country countrycode="392" isreporter="1" ispartner="1" isgroup="No" grouptype="N/A">
<wits:iso3Code>JPN</wits:iso3Code><wits:name>Japan</wits:name></wits:country>
<wits:country countrycode="826" isreporter="1" ispartner="1" isgroup="No" grouptype="N/A">
<wits:iso3Code>GBR</wits:iso3Code><wits:name>United Kingdom</wits:name></wits:country>
<wits:country countrycode="642" isreporter="0" ispartner="1" isgroup="No" grouptype="N/A">
<wits:iso3Code>ROM</wits:iso3Code><wits:name>Romania</wits:name></wits:country>
</wits:countries></wits:datasource>"""


def _series(reporter, partner, year, hs, rate, kind="MFN", nomen="H4"):
    return (
        f'<Series FREQ="A" DATATYPE="Reported" PRODUCTCODE="{hs}" PARTNER="{partner}" REPORTER="{reporter}">'
        f'<Obs TIME_PERIOD="{year}" OBS_VALUE="{rate}" TARIFFTYPE="{kind}" OBS_VALUE_MEASURE="SimpleAverage" '
        f'NOMENCODE="{nomen}" /></Series>'
    )


def _tariff_xml(*series):
    return ('<?xml version="1.0"?><message:StructureSpecificData><message:DataSet>'
            + "".join(series) + "</message:DataSet></message:StructureSpecificData>")


class FakeWits:
    """Answers WITS URLs from memory, counts calls, and can fail once on chosen URLs."""

    def __init__(self, availability, responses, fail_once=()):
        self.availability, self.responses = availability, responses
        self.fail_once, self.calls = set(fail_once), []

    def __call__(self, url, **_):
        self.calls.append(url)
        if url in self.fail_once:
            self.fail_once.discard(url)
            raise ConnectionResetError("interrupted")
        if url == tariffs.AVAILABILITY_URL:
            return self.availability.encode()
        if url == tariffs.COUNTRIES_URL:
            return COUNTRIES.encode()
        if url not in self.responses:
            raise urllib.error.HTTPError(url, 404, "Not Found - NoRecordsFound", {}, None)
        return self.responses[url].encode()


def _wits():
    reporters = [
        REPORTER.format(code="392", iso3="JPN", year=2020, nomen="H5", partners="000;826;"),
        REPORTER.format(code="392", iso3="JPN", year=2021, nomen="H5", partners="000;826;"),
        REPORTER.format(code="826", iso3="GBR", year=2021, nomen="H5", partners="000;"),
    ]
    responses = {
        tariffs.tariff_url("392", "000", 2020): _tariff_xml(_series("392", "000", 2020, "220830", 2.0, nomen="H5")),
        tariffs.tariff_url("392", "826", 2020): _tariff_xml(
            _series("392", "826", 2020, "220830", 1.0, "PREF", "H5")),
        tariffs.tariff_url("392", "000", 2021): _tariff_xml(_series("392", "000", 2021, "220830", 3.0, nomen="H5")),
        tariffs.tariff_url("826", "000", 2021): _tariff_xml(_series("826", "000", 2021, "870323", 10.0, nomen="H5")),
    }
    return AVAILABILITY.format(reporters="\n".join(reporters)), responses


def test_an_interrupted_download_resumes_without_refetching_cached_reporter_years(tmp_path):
    availability, responses = _wits()
    interrupted = tariffs.tariff_url("826", "000", 2021)
    first = FakeWits(availability, responses, fail_once=[interrupted])

    with pytest.raises(ConnectionResetError):
        tariffs.download(tmp_path, fetch=first, workers=1, retries=0)
    second = FakeWits(availability, responses)
    tariffs.download(tmp_path, fetch=second, workers=1, retries=0)

    assert second.calls == [interrupted]  # availability, countries and JPN were cached
    raw = tariffs.read_raw(tmp_path)
    assert sorted(set(zip(raw["reporter"], raw["year"], strict=True))) == [("GBR", 2021), ("JPN", 2020), ("JPN", 2021)]
    assert raw.filter(pl.col("partner") == "826")["rate"].to_list() == [1.0]  # 404 partners are empty


def test_the_download_records_its_pull_date(tmp_path):
    availability, responses = _wits()
    tariffs.download(tmp_path, fetch=FakeWits(availability, responses), workers=1)

    manifest = tariffs.manifest(tmp_path)
    assert manifest["reporter_years"] == 3
    assert len(manifest["pulled"]) == 10  # an ISO date


# --- The panel: worked examples -------------------------------------------------------------

COUNTRY_CODES = {"392": "JPN", "826": "GBR", "276": "DEU", "250": "FRA", "840": "USA", "124": "CAN"}
NO_AGREEMENTS = pl.DataFrame(schema={"exporter": pl.String, "importer": pl.String, "year": pl.Int16,
                                     "trade_agreement": pl.Int8, "customs_union": pl.Int8})


def _raw(*rows):
    """Raw TRAINS rows: (reporter, partner, year, product, nomenclature, tariff_type, rate)."""
    return pl.DataFrame(list(rows), schema=tariffs.RAW_SCHEMA, orient="row")


def _concordance(*pairs, nomenclature="H5"):
    return pl.DataFrame([(nomenclature, old, new) for old, new in pairs],
                        schema={"nomenclature": pl.String, "product": pl.String, "hs6": pl.String}, orient="row")


def _rates(panel, rows, agreements=NO_AGREEMENTS):
    """applied_tariffs on (exporter, importer, hs6, year) tuples."""
    frame = pl.DataFrame(rows, schema={"exporter": pl.String, "importer": pl.String, "hs6": pl.String,
                                       "year": pl.Int16}, orient="row")
    return tariffs.applied_tariffs(frame, panel, agreements)


def test_hs17_codes_mapping_to_one_hs12_code_take_their_mean():
    raw = _raw(
        ("JPN", "000", 2021, "999901", "H5", "MFN", 4.0),
        ("JPN", "000", 2021, "999902", "H5", "MFN", 8.0),
        ("JPN", "826", 2021, "999901", "H5", "PREF", 0.0),  # a preference on one of the two codes
    )
    panel = tariffs.panel(raw, _concordance(("999901", "999900"), ("999902", "999900")), COUNTRY_CODES)

    rates = _rates(panel, [("USA", "JPN", "999900", 2021), ("GBR", "JPN", "999900", 2021)])

    assert rates["mfn_tariff"].to_list() == [6.0, 6.0]
    assert rates["applied_tariff"].to_list() == [6.0, 4.0]  # GBR: mean of 0 (preference) and 8 (MFN)


def test_a_missing_year_is_filled_from_the_nearest_year_within_three_years():
    raw = _raw(
        ("JPN", "000", 2013, "220830", "H4", "MFN", 5.0),
        ("JPN", "000", 2016, "220830", "H4", "MFN", 2.0),
    )
    panel = tariffs.panel(raw, _concordance(), COUNTRY_CODES)
    years = [2012, 2014, 2015, 2019, 2020, 2021]

    rates = _rates(panel, [("GBR", "JPN", "220830", y) for y in years])

    # 2014 and 2015 are equidistant-or-nearer to a real year; 2019 is 3 years from 2016;
    # 2020 and 2021 are 4 or more years from any data, so they stay missing.
    assert rates["applied_tariff"].to_list() == [5.0, 5.0, 2.0, 2.0, None, None]


def test_eu_members_apply_the_eu_schedule_and_intra_eu_pairs_pay_nothing():
    raw = _raw(
        ("EUN", "000", 2019, "870323", "H4", "MFN", 10.0),
        ("EUN", "392", 2019, "870323", "H4", "PREF", 3.0),
        ("JPN", "000", 2019, "870323", "H4", "MFN", 6.0),
        ("JPN", "N04", 2019, "870323", "H4", "PREF", 0.0),  # Japan's preference to the EU as a group
    )
    union = pl.DataFrame([("DEU", "FRA", 2019, 1, 1)], schema=NO_AGREEMENTS.schema, orient="row")
    panel = tariffs.panel(raw, _concordance(), COUNTRY_CODES)

    rates = _rates(panel, [("JPN", "FRA", "870323", 2019), ("USA", "FRA", "870323", 2019),
                           ("DEU", "FRA", "870323", 2019), ("GBR", "FRA", "870323", 2019),
                           ("DEU", "JPN", "870323", 2019), ("USA", "JPN", "870323", 2019)], union)

    # Japan's preference in France; MFN for the USA; 0 inside the Customs Union; the UK in
    # 2019 applies the EU schedule but has no Customs Union row here, so it pays MFN.
    # Germany gets Japan's EU preference; the USA pays Japan's MFN.
    assert rates["applied_tariff"].to_list() == [3.0, 10.0, 0.0, 10.0, 0.0, 6.0]


def test_the_uk_leaves_the_eu_schedule_and_eu_group_preferences_after_2020():
    raw = _raw(
        ("EUN", "000", 2021, "870323", "H4", "MFN", 10.0),
        ("GBR", "000", 2021, "870323", "H4", "MFN", 8.0),
        ("JPN", "000", 2021, "870323", "H4", "MFN", 6.0),
        ("JPN", "A36", 2021, "870323", "H4", "PREF", 0.0),
    )
    panel = tariffs.panel(raw, _concordance(), COUNTRY_CODES)

    rates = _rates(panel, [("USA", "GBR", "870323", 2021), ("GBR", "JPN", "870323", 2021),
                           ("FRA", "JPN", "870323", 2021)])

    assert rates["applied_tariff"].to_list() == [8.0, 6.0, 0.0]


# --- The fixture world: 2021 cut-off and a post-2021 Trade Agreement -------------------------


def _fixture_rates(world):
    return tariffs.applied_tariffs(world.flows, world.tariffs, world.policy)


def test_years_after_2021_carry_2021_forward_and_are_marked_imputed(fixture_world):
    rates = _fixture_rates(fixture_world).filter((pl.col("exporter") == "USA") & (pl.col("importer") == "IND"))

    by_year = rates.group_by("year").agg(pl.col("applied_tariff").sort_by("hs6"), pl.col("tariff_imputed").all(),
                                         pl.col("tariff_imputed").any().alias("any_imputed")).sort("year")
    assert by_year["tariff_imputed"].to_list() == [False, True, True, True]
    assert by_year["any_imputed"].to_list() == [False, True, True, True]
    first, *carried = by_year["applied_tariff"].to_list()
    assert carried == [first] * 3
    assert None not in first and max(first) > 0


def test_a_trade_agreement_entering_force_after_2021_zeroes_the_carried_tariff(fixture_world):
    rates = _fixture_rates(fixture_world).filter((pl.col("exporter") == "GBR") & (pl.col("importer") == "JPN"))

    in_2021 = rates.filter(pl.col("year") == 2021)
    assert (in_2021["applied_tariff"] > 0).any()  # before the agreement, UK goods pay MFN
    assert rates.filter(pl.col("year") > 2021)["applied_tariff"].to_list() == [0.0] * 15
    # The 2021 MFN rate stays available, so a Scenario can restore it.
    carried = rates.filter(pl.col("year") == 2024).sort("hs6")["mfn_tariff"]
    assert carried.to_list() == in_2021.sort("hs6")["mfn_tariff"].to_list()


def test_agreements_already_in_force_in_2021_keep_their_preferential_rates(fixture_world):
    rates = _fixture_rates(fixture_world).filter((pl.col("exporter") == "DEU") & (pl.col("importer") == "JPN"))

    assert rates.filter(pl.col("year") == 2024)["applied_tariff"].to_list() == \
        rates.filter(pl.col("year") == 2021)["applied_tariff"].to_list()


# --- Real data (skipped until the TRAINS download has finished) -----------------------------

# Reporters per year in TRAINS, from the live availability check in docs/research/hs6-tariff-data.md.
RESEARCH_NOTE_REPORTERS = {2012: 137, 2013: 140, 2014: 111, 2015: 139, 2016: 129,
                           2017: 112, 2018: 116, 2019: 121, 2020: 144, 2021: 144}


@pytest.fixture(scope="module")
def real_download():
    if not tariffs.manifest_path().exists():
        pytest.skip(f"TRAINS not downloaded to {tariffs.raw_dir()}")
    return tariffs.raw_dir()


def test_the_real_download_covers_the_reporter_years_in_the_research_note(real_download):
    listed = tariffs.read_availability((real_download / "availability.xml").read_text(encoding="utf-8-sig"))
    listed = listed.filter(pl.col("year").is_in(tariffs.TRAINS_YEARS))
    cached = {(folder.stem, int(folder.parent.name)) for folder in real_download.glob("*/*.parquet")}

    assert dict(listed.group_by("year").len().iter_rows()) == RESEARCH_NOTE_REPORTERS
    assert cached == set(zip(listed["reporter"], listed["year"], strict=True))  # Sudan's two codes share one file


def test_the_data_lineage_records_the_trains_pull_date(real_download):
    from exportspredictor import sources

    trains = next(row for row in sources.source_versions() if row["source"] == "UNCTAD TRAINS")

    assert trains["downloaded"] == tariffs.manifest()["pulled"]
    assert len(trains["downloaded"]) == 10


def test_the_real_panel_puts_eu_members_on_the_eu_schedule_and_the_uk_on_its_own_from_2021(real_download):
    if not (tariffs.derived_dir() / "schedules.parquet").exists():
        pytest.skip("Applied Tariff panel not built (python -m exportspredictor.tariffs)")
    schedules = tariffs.load().schedules

    def schedule(importer, year):
        return schedules.filter((pl.col("importer") == importer) & (pl.col("year") == year))["schedule"].item()

    assert [schedule("DEU", 2015), schedule("GBR", 2015), schedule("GBR", 2021), schedule("GBR", 2024)] == \
        ["EUN", "EUN", "GBR", "GBR"]
    assert schedules.filter(pl.col("year") > 2021)["tariff_imputed"].all()


def test_a_country_listed_under_two_codes_is_fetched_once_from_the_fuller_entry(tmp_path):
    reporters = [REPORTER.format(code="729", iso3="SUD", year=2017, nomen="H5", partners="000;"),
                 REPORTER.format(code="736", iso3="SDN", year=2017, nomen="H5", partners="000;826;")]
    wits = FakeWits(AVAILABILITY.format(reporters="\n".join(reporters)), {})

    tariffs.download(tmp_path, fetch=wits, workers=1)

    assert [url for url in wits.calls if "/reporter/" in url] == [
        tariffs.tariff_url("736", "000", 2017), tariffs.tariff_url("736", "826", 2017)]
