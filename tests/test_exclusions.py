"""The Excluded Products list (ADR 0005): which Products it covers, and the reason it gives for each."""

import polars as pl
import pytest

from exportspredictor.exclusions import (
    EXCLUSIONS,
    REEXPORT_HUBS,
    drop_excluded,
    excluded_products,
    exclusion_reason,
    flag_columns,
    is_flagged_product,
)


@pytest.mark.parametrize(
    "hs6",
    [
        "270900",  # crude oil (Sector 27)
        "271121",  # natural gas
        "271600",  # electricity
        "710691",  # silver
        "710812",  # gold
        "711011",  # platinum
        "711299",  # precious-metal scrap
        "711890",  # coins
        "970600",  # antiques
        "284410",  # radioactive elements
    ],
)
def test_the_list_covers_the_adr_products(hs6):
    assert exclusion_reason(hs6)


@pytest.mark.parametrize(
    "hs6",
    [
        "271210",  # petroleum jelly: Sector 27's one exception
        "711319",  # jewellery: a Flagged Product, not an Excluded one
        "710239",  # diamonds
        "220830",  # whisky
        "284590",  # isotopes other than 2844
    ],
)
def test_other_products_are_not_excluded(hs6):
    assert exclusion_reason(hs6) is None


def test_every_entry_gives_a_reason_naming_what_it_covers():
    assert [e.codes for e in EXCLUSIONS] == [
        "Sector 27 except 2712",
        "7106, 7108, 7110, 7112 and 7118",
        "9706",
        "2844",
    ]
    assert all(len(e.reason) > 40 for e in EXCLUSIONS)
    assert "gold" in exclusion_reason("710812").lower()


def test_drop_excluded_keeps_only_other_products():
    rows = pl.DataFrame({"hs6": ["270900", "220830", "710812", "271210"], "value_usd": [1.0, 2.0, 3.0, 4.0]})

    assert drop_excluded(rows)["hs6"].to_list() == ["220830", "271210"]


def test_drop_excluded_reads_categorical_codes():
    rows = pl.DataFrame({"hs6": ["270900", "220830"]}, schema={"hs6": pl.Categorical})

    assert drop_excluded(rows)["hs6"].cast(pl.String).to_list() == ["220830"]


def test_the_excluded_products_table_lists_each_one_with_its_reason():
    products = pl.DataFrame(
        {
            "hs6": ["220830", "270900", "710812"],
            "description": ["Whiskies", "Crude oil", "Gold, unwrought"],
            "sector": ["22", "27", "71"],
        }
    )

    table = excluded_products(products)

    assert table.columns == ["hs6", "description", "sector", "reason"]
    assert table["hs6"].to_list() == ["270900", "710812"]
    assert table["reason"].to_list() == [exclusion_reason("270900"), exclusion_reason("710812")]


# --- Flagged Products and Re-export Hubs: scored as normal, shown with a caveat ------------


@pytest.mark.parametrize(
    ("hs6", "flagged"),
    [
        ("710110", True),  # pearls (7101)
        ("710239", True),  # diamonds (7102)
        ("710510", True),  # precious-stone dust (7105)
        ("710700", True),  # base metal clad with silver (7107)
        ("710900", True),  # base metal clad with gold (7109)
        ("711100", True),  # base metal clad with platinum (7111)
        ("711319", True),  # gold jewellery (7113)
        ("711719", True),  # imitation jewellery (7117)
        ("710812", False),  # gold: Excluded, not Flagged
        ("710691", False),  # silver: Excluded
        ("710490", True),  # synthetic stones (7104)
        ("711810", False),  # coins: Excluded
        ("220830", False),  # whisky
        ("071000", False),  # frozen vegetables: 0710 is not 7101
    ],
)
def test_flagged_products_are_the_rest_of_sector_71(hs6, flagged):
    assert is_flagged_product(hs6) is flagged


def test_re_export_hubs_are_the_adr_markets():
    assert REEXPORT_HUBS == frozenset({"NLD", "BEL", "IRL", "HKG", "SGP", "ARE", "CHE", "LUX"})


def test_flag_columns_mark_rows_by_the_constants():
    rows = pl.DataFrame({"hs6": ["711319", "220830"], "iso3": ["HKG", "JPN"]})

    flagged = rows.with_columns(flag_columns())

    assert flagged["flagged_product"].to_list() == [True, False]
    assert flagged["reexport_hub"].to_list() == [True, False]
