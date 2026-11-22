"""Caveat Tiers (ADR 0005): a tier is a declaration, and the package reads every declared tier."""

import polars as pl
import pytest

from exportspredictor.caveats import (
    TIERS,
    CaveatContext,
    CaveatTier,
    with_caveat_columns,
)

ROWS = pl.DataFrame(
    {"hs6": ["711319", "220830", "220830", "870323"], "iso3": ["HKG", "JPN", "SAU", "SAU"]}
)
CONTEXT = CaveatContext(baseline_year=2024)


def _tier(column: str, key: str, flags, **declared) -> CaveatTier:
    return CaveatTier(
        column=column,
        badge=column.replace("_", " ").title(),
        plural=column.replace("_", " ").title() + "s",
        text=f"{column} text",
        key=key,
        hide_param=f"hide_{column}",
        hide_description=f"Leave out {column}",
        flags=flags,
        **declared,
    )


# A throwaway tier of each key: a Market (Saudi Arabia from 2020), a Product (whisky) and a pair (whisky to Saudi
# Arabia), the last with a reason per row.
MARKET = _tier(
    "test_market", "market", lambda context: pl.col("iso3").is_in(["SAU"] if context.baseline_year >= 2020 else [])
)
PRODUCT = _tier("test_product", "product", lambda context: pl.col("hs6") == "220830")
PAIR = _tier(
    "test_pair",
    "pair",
    lambda context: (pl.col("hs6") == "220830") & (pl.col("iso3") == "SAU"),
    reason=lambda context: pl.lit("alcohol imports banned"),
    row_text="Test pair: {reason}.",
)


def test_the_declared_tiers_are_flagged_products_re_export_hubs_sanctioned_markets_restricted_and_offset_pairs():
    assert [(tier.column, tier.badge, tier.key, tier.hide_param) for tier in TIERS] == [
        ("flagged_product", "Flagged Product", "product", "hide_flagged"),
        ("reexport_hub", "Re-export Hub", "market", "hide_hubs"),
        ("sanctioned_market", "Sanctioned Market", "market", "hide_sanctioned"),
        ("restricted_pair", "Restricted Pair", "pair", "hide_restricted"),
        ("offset_pair", "Offset Pair", "pair", "hide_offset"),
    ]


def test_the_declared_tiers_mark_the_baseline_as_before():
    marked = with_caveat_columns(ROWS, CONTEXT)

    assert marked.columns == [
        "hs6", "iso3", "flagged_product", "flagged_product_reason", "reexport_hub", "sanctioned_market", "sanctioned_market_reason",
        "restricted_pair", "restricted_pair_reason", "offset_pair", "offset_pair_reason",
    ]
    assert not marked["offset_pair"].any()  # the context holds no Offset Pair
    assert marked["flagged_product"].to_list() == [True, False, False, False]
    assert marked["reexport_hub"].to_list() == [True, False, False, False]
    assert marked["sanctioned_market"].to_list() == [False, False, False, False]
    assert marked["restricted_pair"].to_list() == [False, False, True, False]  # whisky to Saudi Arabia


def test_a_tier_of_each_key_adds_its_column_and_a_reason_column_when_it_declares_reasons():
    marked = with_caveat_columns(ROWS, CONTEXT, tiers=(MARKET, PRODUCT, PAIR))

    assert marked.columns == ["hs6", "iso3", "test_market", "test_product", "test_pair", "test_pair_reason"]
    assert marked["test_market"].to_list() == [False, False, True, True]
    assert marked["test_product"].to_list() == [False, True, True, False]
    assert marked["test_pair"].to_list() == [False, False, True, False]
    assert marked["test_pair_reason"].to_list() == [None, None, "alcohol imports banned", None]


def test_a_tier_reads_the_baseline_year():
    marked = with_caveat_columns(ROWS, CaveatContext(baseline_year=2019), tiers=(MARKET,))

    assert not marked["test_market"].any()


def test_marking_again_replaces_the_columns_in_place():
    once = with_caveat_columns(ROWS, CONTEXT, tiers=(MARKET, PAIR))

    assert with_caveat_columns(once, CONTEXT, tiers=(MARKET, PAIR)).equals(once)


def test_a_tier_that_reads_a_column_outside_its_key_is_refused():
    wrong = _tier("wrong", "market", lambda context: pl.col("hs6") == "220830")

    with pytest.raises(ValueError, match="wrong.*market.*hs6"):
        with_caveat_columns(ROWS, CONTEXT, tiers=(wrong,))


def test_a_row_s_caveat_text_names_its_reason_when_the_tier_declares_one():
    assert PAIR.caveat_text("alcohol imports banned") == "Test pair: alcohol imports banned."
    assert MARKET.caveat_text(None) == "test_market text"
    assert PAIR.reason_column == "test_pair_reason" and MARKET.reason_column is None
