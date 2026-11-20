"""The Leaderboard, the app's default page, run with AppTest through the router against a live local API on
fixture artefacts: its sidebar filters (Product search and Market included), the filters in the URL, Reset filters,
the active-filters line, the table and the CSV."""

import io
import json
import re
from dataclasses import replace

import httpx
import pandas as pd
import polars as pl
import pytest
import streamlit
from conftest import run_page, serve_api
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store

GAP = "Export Gap"  # $ millions
RELATIVE = "Relative Export Gap"
RANGE = "Potential Range"  # hidden by default
WHISKY = "220830"
JEWELLERY = "711319"  # the fixture's Flagged Product; Hong Kong is its Re-export Hub
CRUDE_OIL = "270900"  # the fixture world's Excluded Product


@pytest.fixture
def downloads(monkeypatch):
    """Record what each download button offers, then render it as usual."""
    offered = []
    original = streamlit.download_button

    def record(label, data, *args, **kwargs):
        offered.append(data)
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", record)
    return offered


def _home(monkeypatch, api_url: str, **query: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(query=query)


def _search(monkeypatch, api_url: str, text: str) -> AppTest:
    page = _home(monkeypatch, api_url)
    page.sidebar.text_input(key="product_query").input(text).run()
    return page


def _table(page: AppTest) -> pd.DataFrame:
    """The ranking table with every column: the Wide table view (the default view shows three)."""
    _wide(page)
    return page.dataframe[0].value


def _wide(page: AppTest) -> AppTest:
    """Switch to the Wide table view, if the page shows a ranking and is not there yet."""
    [*switch] = page.segmented_control
    if switch and switch[0].value != "Wide table":
        switch[0].set_value("Wide table").run()
    return page


def _active_filters(page: AppTest) -> str:
    [line] = [element.value for element in page.main.markdown if " shown of " in element.value]
    return line


def _texts(page: AppTest) -> list[str]:
    return [element.value for element in [*page.main.markdown, *page.main.caption, *page.main.info]]


def test_home_opens_on_the_leaderboard_of_every_uk_pair(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    assert not page.exception
    table = _table(page)
    assert len(table) == 20  # every fixture UK Product–Market pair
    gaps = list(table[GAP])
    assert gaps == sorted(gaps, reverse=True)
    assert {"HS6", "Product", "Market", "Opportunity Type", RELATIVE, "Caveats"} <= set(table.columns)
    assert set(table["Opportunity Type"]) == {"Expand"}
    assert "Re-export Hub" in set(table["Caveats"]) and "Flagged Product" in set(table["Caveats"])
    # The summary metrics row is gone: the only figures are the detail panel's three
    assert [metric.label for metric in page.metric] == ["Export Potential", "Actual Exports", "Export Gap"]


def test_the_filters_sit_in_the_sidebar_with_a_reset_button(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    sidebar = page.sidebar
    assert {box.key for box in sidebar.selectbox} >= {"market", "sector", "region", "opportunity_type", "trade_agreement"}
    assert {"product_query"} <= {box.key for box in sidebar.text_input}
    assert {"min_gap_usd"} <= {box.key for box in sidebar.number_input}
    assert {box.label for box in sidebar.checkbox} >= {"Hide Flagged Products", "Hide Re-export Hubs"}
    assert "rank_by" in {radio.key for radio in sidebar.radio}
    assert "Reset filters" in [button.label for button in sidebar.button]
    assert not page.main.selectbox and not page.main.checkbox


def test_the_active_filters_line_states_the_counts_and_the_total_positive_export_gap(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    line = _active_filters(page)
    gaps = [gap * 1e6 for gap in _table(page)[GAP]]  # the table's $ millions, back in dollars
    assert "No filters" in line and "20 pairs shown of 20" in line
    positive = sum(gap for gap in gaps if gap > 0)
    assert f"${positive / 1e3:,.1f}k" in line and "Total positive Export Gap" in line

    page.sidebar.selectbox(key="market").set_value("JPN").run()

    line = _active_filters(page)
    assert "Market: Japan" in line and "5 pairs shown of 5" in line


def test_the_active_filters_line_says_pair_for_one_pair(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url, product="220830", market="JPN")

    assert "1 pair shown of 1." in _active_filters(page)


# --- Product search -----------------------------------------------------------------------------


def test_a_product_found_by_its_code_ranks_that_products_markets(monkeypatch, fixture_api_url):
    page = _search(monkeypatch, fixture_api_url, WHISKY)

    assert not page.exception
    table = _table(page)
    assert set(table["HS6"]) == {WHISKY}
    assert set(table["Market"]) == {"Japan", "Canada", "India", "Hong Kong"}
    gaps = list(table[GAP])
    assert gaps == sorted(gaps, reverse=True)
    assert f"Product: {WHISKY}" in _active_filters(page)


def test_a_product_found_by_everyday_words_ranks_that_products_markets(monkeypatch, fixture_api_url):
    page = _search(monkeypatch, fixture_api_url, "smartphone")

    assert page.sidebar.selectbox(key="product").value == "851712"
    assert set(_table(page)["HS6"]) == {"851712"}


def test_choosing_among_several_matches_ranks_that_product(monkeypatch, fixture_api_url):
    page = _search(monkeypatch, fixture_api_url, "8")  # the prefix of 851712 and 870323

    assert len(page.sidebar.selectbox(key="product").options) == 2
    page.sidebar.selectbox(key="product").select_index(1).run()

    assert set(_table(page)["HS6"]) == {"870323"}


def test_search_text_that_matches_nothing_says_so(monkeypatch, fixture_api_url):
    page = _search(monkeypatch, fixture_api_url, "zzz-not-a-product")

    assert not page.exception
    assert any("zzz-not-a-product" in element.value for element in page.warning)


@pytest.mark.parametrize("found", ["search", "url"])
def test_an_excluded_product_shows_its_reason_and_no_ranking(monkeypatch, fixture_api_url, found):
    if found == "search":
        page = _search(monkeypatch, fixture_api_url, "crude")
        assert "(Excluded Product)" in page.sidebar.selectbox(key="product").options[0]
    else:
        page = _home(monkeypatch, fixture_api_url, product=CRUDE_OIL)

    assert not page.exception
    warning = " ".join(element.value for element in page.warning)
    assert "Excluded Product" in warning and "pipelines" in warning
    assert any("Excluded Products" in text and "Model Evaluation" in text for text in _texts(page))
    assert not page.dataframe  # no ranking is shown


# --- The other filters -----------------------------------------------------------------------------


def test_the_market_filter_ranks_that_markets_products(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    page.sidebar.selectbox(key="market").set_value("JPN").run()

    table = _table(page)
    assert set(table["Market"]) == {"Japan"}
    assert set(table["HS6"]) == {"711319", "220830", "300490", "851712", "870323"}
    gaps = list(table[GAP])
    assert gaps == sorted(gaps, reverse=True)


def test_the_region_filter_narrows_the_market_choice(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    page.sidebar.selectbox(key="region").set_value("South Asia").run()

    assert page.sidebar.selectbox(key="market").options == ["All Markets", "India"]
    assert set(_table(page)["Market"]) == {"India"}


def test_the_sector_filter_lists_sectors_by_name(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    sector = page.sidebar.selectbox(key="sector")

    assert "Beverages, spirits and vinegar" in sector.options
    assert "Electrical machinery and equipment" in sector.options
    sector.set_value("85").run()

    assert set(_table(page)["HS6"]) == {"851712"}
    assert "Sector: Electrical machinery and equipment" in _active_filters(page)


def test_filters_change_the_rows(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    page.sidebar.selectbox(key="trade_agreement").set_value(False).run()
    page.sidebar.checkbox(key="hide_hubs").check().run()
    assert set(_table(page)["Market"]) == {"India"}

    page.sidebar.selectbox(key="opportunity_type").set_value("New Product").run()
    assert not page.dataframe
    assert any("No Opportunities" in element.value for element in page.info)


def test_the_min_gap_filter_drops_smaller_gaps(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url, market="CAN")
    everything = _table(page)

    page.sidebar.number_input(key="min_gap_usd").set_value(0).run()

    table = _table(page)
    assert 0 < len(table) < len(everything) and (table[GAP] >= 0).all()


def test_relative_toggle_shows_the_minimum_selector_and_reranks(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    assert "min_potential_usd" not in {box.key for box in page.selectbox}

    page.sidebar.radio(key="rank_by").set_value("relative").run()

    minimum = page.sidebar.selectbox(key="min_potential_usd")
    assert minimum.value == 1e6 and minimum.options == ["$0", "$100.0k", "$1.0m", "$10.0m", "$100.0m"]
    # Every fixture pair's Export Potential is below $1M.
    assert not page.dataframe and "0 pairs shown of 0" in _active_filters(page)

    minimum.set_value(0.0).run()

    relative = list(_table(page)[RELATIVE])
    assert len(relative) == 20 and relative == sorted(relative, reverse=True)


def test_rows_show_caveat_badges_and_the_hide_filters_are_off_by_default(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url, product=JEWELLERY)
    assert page.sidebar.checkbox(key="hide_flagged").value is False
    assert page.sidebar.checkbox(key="hide_hubs").value is False
    caveats = dict(zip(_table(page)["Market"], _table(page)["Caveats"], strict=True))
    assert caveats["Japan"] == "Flagged Product" and caveats["Hong Kong"] == "Flagged Product, Re-export Hub"

    page.sidebar.checkbox(key="hide_hubs").check().run()
    assert set(_table(page)["Market"]) == {"Japan", "Canada", "India"}
    page.sidebar.checkbox(key="hide_flagged").check().run()
    assert not page.dataframe


def test_each_row_shows_whether_a_uk_trade_agreement_is_in_force(monkeypatch, fixture_api_url):
    table = _table(_home(monkeypatch, fixture_api_url, product=WHISKY))

    assert dict(zip(table["Market"], table["UK Trade Agreement"], strict=True)) == {
        "Japan": "Yes",
        "Canada": "Yes",
        "India": "No",
        "Hong Kong": "No",
    }


# --- Filters in the URL, for the session, and Reset filters -----------------------------------------


def test_query_parameters_set_the_filters(monkeypatch, fixture_api_url):
    page = _home(
        monkeypatch, fixture_api_url, sector="22", trade_agreement="no", hide_hubs="true", rank_by="relative",
        min_potential_usd="0",
    )

    sidebar = page.sidebar
    assert sidebar.selectbox(key="sector").value == "22"
    assert sidebar.selectbox(key="trade_agreement").value is False
    assert sidebar.checkbox(key="hide_hubs").value is True
    assert sidebar.radio(key="rank_by").value == "relative"
    assert list(_table(page)["Market"]) == ["India"]  # whisky, no UK agreement, not a Re-export Hub


def test_a_link_naming_one_product_and_one_market_leaves_that_pair(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url, product=WHISKY, market="JPN")

    assert page.sidebar.text_input(key="product_query").value == WHISKY
    assert page.sidebar.selectbox(key="market").value == "JPN"
    table = _table(page)
    assert list(zip(table["HS6"], table["Market"], strict=True)) == [(WHISKY, "Japan")]


def test_changing_a_filter_writes_it_to_the_url_leaving_defaults_out(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    assert page.query_params == {}

    page.sidebar.selectbox(key="market").set_value("JPN").run()
    page.sidebar.text_input(key="product_query").input(WHISKY).run()
    page.sidebar.checkbox(key="hide_flagged").check().run()

    assert page.query_params == {"market": "JPN", "product": WHISKY, "hide_flagged": "true"}

    page.sidebar.checkbox(key="hide_flagged").uncheck().run()
    assert page.query_params == {"market": "JPN", "product": WHISKY}


def test_reset_filters_restores_the_full_ranking_and_a_clean_url(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url, product=WHISKY, market="JPN", hide_hubs="true")
    assert len(_table(page)) == 1

    page.sidebar.button(key="reset_filters").click().run()

    assert len(_table(page)) == 20
    assert page.query_params == {}
    assert page.sidebar.text_input(key="product_query").value == ""
    assert page.sidebar.selectbox(key="market").value is None
    assert "No filters" in _active_filters(page)


def test_filters_persist_for_the_session_across_pages(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    page.sidebar.selectbox(key="market").set_value("JPN").run()

    page.switch_page("pages/scenarios.py")
    page.query_params.clear()  # a page switch leaves the Leaderboard's parameters behind
    page.run()
    page.switch_page("pages/leaderboard.py")
    page.query_params.clear()
    page.run()

    assert not page.exception
    assert page.sidebar.selectbox(key="market").value == "JPN"
    assert set(_table(page)["Market"]) == {"Japan"}
    assert page.query_params == {"market": "JPN"}


# --- The table and the CSV -----------------------------------------------------------------------

SHOWN_COLUMNS = [
    "Product",
    "Market",
    "Export Gap",
    "Export Potential",
    "Actual Exports",
    "Applied Tariff %",
    "Relative Export Gap",
    "UK Trade Agreement",
    "Blocked Opportunity value",
    "Blocked Opportunity",
    "Caveats",
]
HIDDEN_COLUMNS = ["Opportunity Type", "Potential Range", "HS6", "Sector", "Region"]
MONEY = ["Export Gap", "Export Potential", "Actual Exports", "Blocked Opportunity value"]


def _column_config(page: AppTest) -> dict:
    """Each column's config, by the heading it configures."""
    config = json.loads(_wide(page).dataframe[0].proto.columns)
    return {value.get("label") or key.removeprefix("name:"): value for key, value in config.items()}


def _api_rows(api_url: str, **params) -> list[dict]:
    return httpx.get(f"{api_url}/opportunities/top", params=params).json()["opportunities"]


def test_the_default_columns_are_shown_in_order_and_the_others_hidden(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    order = list(_wide(page).dataframe[0].proto.column_order)
    assert order == SHOWN_COLUMNS
    assert set(HIDDEN_COLUMNS) <= set(_table(page).columns)  # in the data, offered by Show/hide columns
    assert not set(HIDDEN_COLUMNS) & set(order)
    assert not any(config.get("pinned") for config in _column_config(page).values())


def test_money_columns_are_numeric_dollar_millions_with_the_unit_stated_once(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    table = _table(page)
    rows = {(row["hs6"], row["market"]): row for row in _api_rows(fixture_api_url)}
    for _, shown in table.iterrows():
        row = rows[(shown["HS6"], shown["Market"])]
        assert shown["Export Gap"] == pytest.approx(row["export_gap_usd"] / 1e6)
        assert shown["Export Potential"] == pytest.approx(row["export_potential_usd"] / 1e6)
        assert shown["Actual Exports"] == pytest.approx(row["actual_exports_usd"] / 1e6)
        assert shown["Blocked Opportunity value"] == pytest.approx(row["blocked_opportunity_value_usd"] / 1e6)
    config = _column_config(page)
    for column in MONEY:
        assert pd.api.types.is_float_dtype(table[column])  # numeric, so they sort as numbers
        assert config[column]["type_config"]["format"] == "%.1f"
    # Once above the ranking (the panel's yearly figures state their own unit)
    assert len([text for text in _texts(page) if "$ millions" in text and "Export Gap" in text]) == 1
    assert not any("USD" in column for column in table.columns)  # the unit is not in the headings


def test_short_headings_carry_tooltips_and_the_four_terms_use_the_apis_wording(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)

    terms = httpx.get(f"{fixture_api_url}/labels").json()["terms"]
    helps = {name: config.get("help") for name, config in _column_config(page).items()}
    assert all(helps.get(column) for column in [*SHOWN_COLUMNS, *HIDDEN_COLUMNS])
    assert helps["Export Potential"] == terms["export_potential"]
    assert helps["Actual Exports"] == terms["actual_exports"]
    assert helps["Export Gap"] == terms["export_gap"]
    assert helps["Opportunity Type"] == terms["opportunity_type"]


def test_each_row_shows_the_applied_tariff_in_percent(monkeypatch, fixture_api_url):
    table = _table(_home(monkeypatch, fixture_api_url, product=WHISKY))

    tariffs = dict(zip(table["Market"], table["Applied Tariff %"], strict=True))
    assert tariffs["India"] == pytest.approx(150.0, rel=1e-5) and tariffs["Japan"] == 0.0


@pytest.mark.parametrize("query", [{}, {"product": WHISKY}, {"market": "JPN"}], ids=["all", "product", "market"])
def test_the_csv_holds_every_filtered_row_in_full_dollars_with_every_column(
    monkeypatch, fixture_api_url, downloads, query
):
    page = _home(monkeypatch, fixture_api_url, **query)

    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str}, keep_default_na=False)
    rows = _api_rows(fixture_api_url, **query)
    assert list(zip(exported["HS6"], exported["Market"], strict=True)) == [(r["hs6"], r["market"]) for r in rows]
    for column, field in [
        ("Export Gap, 2022–2024 average (USD, nominal)", "export_gap_usd"),
        ("Export Potential, 2022–2024 average (USD, nominal)", "export_potential_usd"),
        ("Actual Exports, 2022–2024 average (USD, nominal)", "actual_exports_usd"),
        ("Blocked Opportunity value, 2024, illustrative (USD, nominal)", "blocked_opportunity_value_usd"),
        ("Potential Range low, 2022–2024 average (USD, nominal)", "potential_range_low_usd"),
        ("Potential Range high, 2022–2024 average (USD, nominal)", "potential_range_high_usd"),
    ]:
        assert list(exported[column]) == pytest.approx([row[field] for row in rows]), column
    others = set(_table(page).columns) - {*MONEY, "Potential Range"}
    assert others <= set(exported.columns)  # every other column, under its table heading


def test_the_leaderboard_shows_each_potential_range_but_no_uncertain_opportunity_badge_or_filter(
    monkeypatch, fixture_api_url
):
    page = _home(monkeypatch, fixture_api_url)

    table = _table(page)
    ranges = list(table[RANGE])
    assert all(re.fullmatch(r"-?[0-9.,]+–[0-9.,]+", text) for text in ranges)  # $ millions, low–high
    # The fixture has Uncertain Opportunities among these rows; the badge marked almost every row, so it is gone
    assert not table["Caveats"].str.contains("Uncertain Opportunity").any()
    assert not any("Uncertain Opportunit" in caption.value for caption in page.caption)
    assert not any("Uncertain" in checkbox.label for checkbox in page.checkbox)


# --- Long rankings: the top 500, Show all up to 6,000 ------------------------------------------------


def _many_pairs(fixture_artefacts, root, products: int):
    """The fixture artefacts with `products` more Products, each a copy of whisky's UK pairs ranked below them."""
    codes = [f"9{i:05d}" for i in range(products)]

    def copies(table: pl.DataFrame, **changed) -> pl.DataFrame:
        template = table.filter(pl.col("hs6") == WHISKY)
        extra = [template.with_columns(hs6=pl.lit(code), **changed) for code in codes]
        return pl.concat([table, *extra], how="vertical_relaxed")

    many = replace(
        fixture_artefacts,
        baseline=copies(fixture_artefacts.baseline, export_gap_usd=pl.lit(-1e9)),
        baseline_features=copies(fixture_artefacts.baseline_features),
        products=copies(fixture_artefacts.products, description=pl.lit("Constructed product")),
    )
    store.save(many, root)
    return root


@pytest.fixture(scope="module")
def several_hundred_api(fixture_artefacts, tmp_path_factory):
    """620 UK pairs: more than 500, no more than 6,000."""
    with serve_api(_many_pairs(fixture_artefacts, tmp_path_factory.mktemp("many"), 150)) as url:
        yield url


@pytest.fixture(scope="module")
def too_many_api(fixture_artefacts, tmp_path_factory):
    """6,420 UK pairs: more than 6,000."""
    with serve_api(_many_pairs(fixture_artefacts, tmp_path_factory.mktemp("too_many"), 1600)) as url:
        yield url


def test_show_all_is_offered_only_when_the_ranking_has_501_to_6000_pairs(
    monkeypatch, fixture_api_url, several_hundred_api, too_many_api
):
    assert not _home(monkeypatch, fixture_api_url).toggle  # 20 pairs: every one shown
    assert not _home(monkeypatch, too_many_api).toggle  # 6,420 pairs: too many to show

    page = _home(monkeypatch, several_hundred_api)
    assert len(_table(page)) == 500 and "500 pairs shown of 620" in _active_filters(page)

    page.toggle(key="show_all").set_value(True).run()

    assert len(_table(page)) == 620 and "620 pairs shown of 620" in _active_filters(page)
    assert page.query_params.get("show_all") == "true"


def test_show_all_turns_off_when_the_filtered_count_leaves_the_range(monkeypatch, several_hundred_api):
    page = _home(monkeypatch, several_hundred_api, show_all="true")
    assert page.toggle(key="show_all").value is True and len(_table(page)) == 620

    page.sidebar.selectbox(key="market").set_value("JPN").run()  # 151 pairs

    assert not page.toggle and "show_all" not in page.query_params
    page.sidebar.selectbox(key="market").set_value(None).run()
    assert page.toggle(key="show_all").value is False and len(_table(page)) == 500


def test_reset_filters_switches_show_all_off(monkeypatch, several_hundred_api):
    page = _home(monkeypatch, several_hundred_api, show_all="true")

    page.sidebar.button(key="reset_filters").click().run()

    assert page.toggle(key="show_all").value is False and len(_table(page)) == 500
    assert page.query_params == {}


def test_above_6000_pairs_the_page_shows_the_top_500_and_points_to_the_filters(monkeypatch, too_many_api, downloads):
    page = _home(monkeypatch, too_many_api)

    assert len(_table(page)) == 500
    texts = " ".join(_texts(page))
    assert "top 500 of 6,420 pairs" in texts.lower() and "filters" in texts
    assert len(pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str})) == 6000
    assert "top 6,000 of 6,420" in texts.lower()
