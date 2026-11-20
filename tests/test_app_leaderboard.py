"""The Leaderboard, the app's default page, run with AppTest through the router against a live local API on
fixture artefacts: its sidebar filters (Product search and Market included), the filters in the URL, Reset filters,
the active-filters line, the table and the CSV."""

import io
import re

import pandas as pd
import pytest
import streamlit
from conftest import run_page
from streamlit.testing.v1 import AppTest

GAP = "Export Gap, 2022–2024 average (USD, nominal)"
RELATIVE = "Relative Export Gap, 2022–2024 average"
RANGE = "Potential Range, 2022–2024 average (USD, nominal)"
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
    return page.dataframe[0].value


def _active_filters(page: AppTest) -> str:
    [line] = [element.value for element in page.main.markdown if "pairs shown of" in element.value]
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
    assert not page.metric  # the summary metrics row is gone


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
    gaps = list(_table(page)[GAP])
    assert "No filters" in line and "20 pairs shown of 20" in line
    positive = sum(gap for gap in gaps if gap > 0)
    assert f"${positive / 1e3:,.1f}k" in line and "Total positive Export Gap" in line

    page.sidebar.selectbox(key="market").set_value("JPN").run()

    line = _active_filters(page)
    assert "Market: Japan" in line and "5 pairs shown of 5" in line


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

    assert dict(zip(table["Market"], table["UK Trade Agreement in force"], strict=True)) == {
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


def test_ranking_values_are_labelled_as_gap_period_averages(monkeypatch, fixture_api_url):
    columns = list(_table(_home(monkeypatch, fixture_api_url)).columns)

    for value in ("Export Potential", "Actual Exports", "Export Gap"):
        assert f"{value}, 2022–2024 average (USD, nominal)" in columns
    # The Blocked Opportunity value is a 2024 Scenario Effect, not an average, and Illustrative.
    assert "Blocked Opportunity value, illustrative (USD, nominal)" in columns


@pytest.mark.parametrize("query", [{}, {"product": WHISKY}, {"market": "JPN"}], ids=["all", "product", "market"])
def test_csv_export_matches_the_leaderboard_on_screen(monkeypatch, fixture_api_url, downloads, query):
    page = _home(monkeypatch, fixture_api_url, **query)

    on_screen = _table(page).drop(columns="Why?")  # links are navigation, not data
    # keep_default_na: a row without caveats exports an empty Caveats cell, not a missing value
    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str, "Sector": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(exported, on_screen.reset_index(drop=True), check_dtype=False)


def test_the_leaderboard_shows_each_potential_range_but_no_uncertain_opportunity_badge_or_filter(
    monkeypatch, fixture_api_url
):
    page = _home(monkeypatch, fixture_api_url)

    table = _table(page)
    ranges = list(table[RANGE])
    assert all(re.fullmatch(r"\$[0-9.,]+(bn|m|k)?–\$[0-9.,]+(bn|m|k)?", text) for text in ranges)
    # The fixture has Uncertain Opportunities among these rows; the badge marked almost every row, so it is gone
    assert not table["Caveats"].str.contains("Uncertain Opportunity").any()
    assert not any("Uncertain Opportunit" in caption.value for caption in page.caption)
    assert not any("Uncertain" in checkbox.label for checkbox in page.checkbox)
