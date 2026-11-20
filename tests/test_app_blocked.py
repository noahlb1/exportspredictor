"""The Blocked Opportunities page, run with AppTest through the router against a live local API.

Most tests use the Blocked world (`conftest.blocked_artefacts`): the fixture with its amounts × 1,000, so it has four
Blocked Opportunities, whisky and jewellery in India, jewellery and cars in Hong Kong. The fixture alone has none.
"""

import io
import json
import re
from dataclasses import replace

import httpx
import pandas as pd
import pytest
import streamlit
from conftest import run_page
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store

PAGE = "pages/blocked_opportunities.py"
WHISKY, JEWELLERY, CARS, CRUDE_OIL = "220830", "711319", "870323", "270900"
BLOCKED_PAIRS = [(WHISKY, "IND"), (JEWELLERY, "IND"), (JEWELLERY, "HKG"), (CARS, "HKG")]
PRODUCT_COLUMNS = ["Product", "Blocked Opportunity value", "Applied Tariff %", "Potential with agreement", "Actual Exports"]
# #102's sentence, on a pair whose Baseline-year Actual Exports are above Export Potential with a Trade Agreement.
EXTRA = (
    "The UK already exports more here than the Export Potential the model associates with a Trade Agreement. The "
    "Blocked Opportunity value is still the rise in Export Potential the model associates with the agreement; it does "
    "not depend on Actual Exports."
)
# Wording that would read a Blocked Opportunity value as a cause or a forecast.
CAUSAL = re.compile(
    r"(?<!not )\b(causes?|caused|will (raise|add|unlock|increase)|would (raise|add|unlock|increase)|unlocks?)\b",
    re.IGNORECASE,
)


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


def _page(monkeypatch, api_url: str, **query: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(PAGE, query=query)


def _read(api_url: str, **params) -> dict:
    return httpx.get(f"{api_url}/opportunities/blocked", params=params).json()


def _choose(page: AppTest, hs6: str, iso3: str) -> AppTest:
    """Choose a Market and one of its Products, as clicking them would (AppTest cannot click a row)."""
    page.session_state["blocked_market"] = iso3
    page.session_state["blocked_pair"] = (hs6, iso3)
    return page.run()


def _markets(page: AppTest) -> pd.DataFrame:
    return page.dataframe[0].value


def _products(page: AppTest) -> pd.DataFrame:
    return page.dataframe[1].value


def _texts(page: AppTest) -> list[str]:
    """Every piece of text the page shows, tooltips and table headings included, but the footer every page shares."""
    texts = [
        element.value
        for element in [*page.markdown, *page.caption[:-1], *page.info, *page.warning, *page.subheader, *page.header]
    ]
    texts += [element.proto.help for element in page.markdown if element.proto.help]
    for metric in page.metric:
        texts += [metric.label, metric.value, metric.proto.help]
    texts += [expander.label for expander in page.expander]
    for table in page.dataframe:
        texts += list(table.value.columns)
        texts += [json.dumps(config) for config in json.loads(table.proto.columns or "{}").values()]
    return texts


def test_markets_are_listed_largest_total_first_with_the_top_market_chosen(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    assert not page.exception
    assert page.header[0].value == "Blocked Opportunities"
    markets = _markets(page)
    assert list(markets.columns) == ["Market", "Blocked Opportunities", "Total value"]
    assert list(markets["Market"]) == ["India", "Hong Kong"]
    assert list(markets["Blocked Opportunities"]) == [2, 2]
    totals = {market["market"]: market["blocked_opportunity_value_usd"] for market in _read(blocked_api_url)["markets"]}
    assert list(markets["Total value"]) == pytest.approx([totals["India"] / 1e6, totals["Hong Kong"] / 1e6])
    # The top Market's Products are shown on open.
    assert set(_products(page)["Product"]) == {
        "Whiskies",
        "Articles of jewellery and parts thereof, of precious metal other than silver",
    }


def test_the_chosen_markets_products_have_the_stated_columns_in_baseline_year_figures(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    page.session_state["blocked_market"] = "HKG"
    page.run()

    products = _products(page)
    assert list(products.columns) == PRODUCT_COLUMNS
    rows = [row for row in _read(blocked_api_url)["opportunities"] if row["iso3"] == "HKG"]
    assert list(products["Product"]) == [row["description"] for row in rows]  # every one, by value
    for field, column in [
        ("blocked_opportunity_value_usd", "Blocked Opportunity value"),
        ("potential_with_agreement_usd", "Potential with agreement"),
        ("baseline_actual_exports_usd", "Actual Exports"),
    ]:
        assert list(products[column]) == pytest.approx([row[field] / 1e6 for row in rows])
    assert any("Hong Kong" in element.value and "2 Blocked Opportunities" in element.value for element in page.markdown)
    config = json.loads(page.dataframe[1].proto.columns)
    helps = {name.removeprefix("name:"): column.get("help", "") for name, column in config.items()}
    assert "Export Potential with a Trade Agreement" in helps["Potential with agreement"]
    assert "llustrative" in helps["Potential with agreement"]
    assert "Trade Agreement alone" in helps["Applied Tariff %"] and "not known" in helps["Applied Tariff %"]


@pytest.mark.parametrize(("hs6", "iso3"), BLOCKED_PAIRS)
def test_no_export_gap_appears_anywhere_on_the_page(monkeypatch, blocked_api_url, hs6, iso3):
    page = _choose(_page(monkeypatch, blocked_api_url), hs6, iso3)

    assert not page.exception
    assert [text for text in _texts(page) if "Export Gap" in text] == []


def test_the_panel_leads_with_the_barrier_then_three_baseline_year_figures(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")

    assert any("The Barrier" in box.value for box in page.warning)
    labels = [metric.label for metric in page.metric]
    assert labels == ["Baseline Export Potential", "Export Potential with a Trade Agreement", "Actual Exports"]
    # Whisky in India, 2024: $40.6m Export Potential, $12.2m Blocked Opportunity value, $38.1m exported.
    assert [metric.value for metric in page.metric] == ["$40.6m", "$52.8m", "$38.1m"]


@pytest.mark.parametrize(("hs6", "iso3"), BLOCKED_PAIRS)
def test_the_extra_sentence_shows_only_where_actual_exports_are_above_potential_with_agreement(
    monkeypatch, blocked_api_url, hs6, iso3
):
    page = _choose(_page(monkeypatch, blocked_api_url), hs6, iso3)

    [barrier] = [box.value for box in page.warning if "The Barrier" in box.value]
    # Cars in Hong Kong: $9.8m exported in 2024, above $5.7m with an agreement. The others are below.
    assert (EXTRA in barrier) == ((hs6, iso3) == (CARS, "HKG"))


def test_the_active_filters_line_states_the_count_and_the_illustrative_total(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    line = next(element.value for element in page.markdown if "No filters" in element.value)
    # 12.2 + 5.6 + 4.9 + 1.1 = $23.8m
    assert "4 Blocked Opportunities" in line and "illustrative" in line and "$23.8m" in line
    caption = " ".join(element.value for element in page.caption)
    assert "$ millions" in caption and "2024" in caption and "Scenarios page" in caption
    assert "20%" in caption and "$1.0m" in caption


def test_sidebar_filters_narrow_the_markets_and_reset_restores_them(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    page.sidebar.checkbox(key="blocked_hide_hubs").check().run()
    assert list(_markets(page)["Market"]) == ["India"]
    assert page.query_params["hide_hubs"] == "true"
    assert any("Re-export Hubs hidden" in element.value for element in page.markdown)

    page.sidebar.button(key="blocked_reset_filters").click().run()
    assert list(_markets(page)["Market"]) == ["India", "Hong Kong"]
    assert "hide_hubs" not in page.query_params


def test_the_page_has_no_market_order_type_or_agreement_filter(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    keys = {widget.key for widget in [*page.sidebar.selectbox, *page.sidebar.radio, *page.sidebar.number_input]}
    assert keys == {"blocked_sector", "blocked_region", "blocked_min_value_usd"}


def test_a_product_search_keeps_that_products_blocked_opportunities(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    page.sidebar.text_input(key="blocked_product_query").input("jewellery").run()

    assert list(_markets(page)["Market"]) == ["India", "Hong Kong"]
    assert list(_markets(page)["Blocked Opportunities"]) == [1, 1]
    assert page.query_params["product"] == JEWELLERY


def test_url_parameters_set_the_filters(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url, region="South Asia", min_value_usd="6000000")

    assert list(_markets(page)["Market"]) == ["India"]
    assert list(_products(page)["Product"]) == ["Whiskies"]


def test_filters_are_kept_apart_from_the_leaderboards(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    page.sidebar.checkbox(key="blocked_hide_hubs").check().run()

    page.query_params.clear()  # a browser clears them on a page switch; AppTest keeps them
    page.switch_page("pages/leaderboard.py").run()

    assert page.session_state["hide_hubs"] is False
    assert page.session_state["blocked_hide_hubs"] is True


def test_csv_holds_every_filtered_blocked_opportunity_in_full_dollars(monkeypatch, blocked_api_url, downloads):
    _page(monkeypatch, blocked_api_url)

    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str})
    rows = _read(blocked_api_url)["opportunities"]
    assert list(zip(exported["HS6"], exported["Market"], strict=True)) == [(r["hs6"], r["market"]) for r in rows]
    value = next(column for column in exported.columns if column.startswith("Blocked Opportunity value"))
    assert "illustrative" in value and "2024" in value and "USD" in value
    assert list(exported[value]) == pytest.approx([row["blocked_opportunity_value_usd"] for row in rows])
    assert not [column for column in exported.columns if "Export Gap" in column]


def test_the_illustrative_note_sits_in_an_expander_under_the_tables(monkeypatch, blocked_api_url):
    note = _read(blocked_api_url)["note"]

    page = _page(monkeypatch, blocked_api_url)

    [expander] = [e for e in page.expander if note["label"] in e.label]
    text = " ".join(element.value for element in [*expander.markdown, *expander.info])
    assert note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["text"] in text


def test_every_text_uses_associational_language(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), CARS, "HKG")

    texts = _texts(page)
    assert not [text for text in texts if CAUSAL.search(text)]
    about_value = [
        element.value for element in [*page.caption, *page.warning] if "Blocked Opportunity value" in element.value
    ]
    assert about_value and all("associat" in text for text in about_value)


def test_with_no_blocked_opportunity_the_page_says_so(monkeypatch, fixture_api_url):
    page = _page(monkeypatch, fixture_api_url)

    assert not page.exception
    assert not page.dataframe
    assert any("No Blocked Opportunities" in element.value for element in page.info)


def test_an_excluded_product_in_the_url_shows_its_reason(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url, product=CRUDE_OIL)

    assert not page.dataframe
    assert any("Excluded Product" in element.value for element in page.warning)


def test_the_leaderboard_no_longer_offers_a_blocked_ranking(monkeypatch, fixture_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", fixture_api_url)
    home = run_page()

    assert home.radio(key="rank_by").options == ["Export Gap", "Relative Export Gap"]


def test_with_no_benchmark_or_event_backtests_the_note_keeps_the_label_and_says_they_were_not_run(
    monkeypatch, blocked_artefacts, tmp_path, live_api
):
    report = {key: value for key, value in blocked_artefacts.evaluation.items() if key != "event_backtest"}
    store.save(replace(blocked_artefacts, ppml_benchmark=None, evaluation=report), tmp_path)

    with live_api(tmp_path) as api_url:
        note = _read(api_url)["note"]
        page = _page(monkeypatch, api_url)

    assert not page.exception
    text = " ".join(element.value for element in [*page.markdown, *page.info])
    assert note["label"] in text and note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["run"] is False and note[part]["text"] in text


# --- Final review of #103: the Barrier first, Potential Range kept, one read ---------------------------------------


def _in_order(page: AppTest) -> list:
    """Every element of the main area, in the order the page draws them."""
    return list(page.main)  # depth first: each block, then what it holds


def test_the_panel_opens_with_the_barrier_box_headed_by_the_pair_then_its_caveats(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), JEWELLERY, "HKG")  # a Re-export Hub: it has a caveat

    texts = [str(getattr(node, "value", "")) for node in _in_order(page)]
    barrier = next(i for i, text in enumerate(texts) if "The Barrier" in text)
    assert "Articles of jewellery" in texts[barrier] and "Hong Kong" in texts[barrier]
    hs6 = next(i for i, text in enumerate(texts) if text.startswith(f"HS6 {JEWELLERY}"))
    caveat = next(i for i, text in enumerate(texts) if text.startswith("Re-export Hub"))
    assert barrier < hs6 and barrier < caveat
    assert not [element for element in page.subheader if "Hong Kong" in element.value]  # the box names the pair


def test_the_facts_keep_the_potential_range_but_no_relative_export_gap(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")

    facts = [element.value for element in page.markdown if element.value.startswith("**") and ":**" in element.value]
    assert any(fact.startswith("**Potential Range:**") for fact in facts)
    assert not [fact for fact in facts if "Relative Export Gap" in fact]


def test_the_page_makes_one_read_and_states_the_thresholds_it_carries(monkeypatch, blocked_api_url):
    from app import client

    paths = []
    original = client.get

    def record(path, **params):
        paths.append(path)
        return original(path, **params)

    monkeypatch.setattr(client, "get", record)
    streamlit.cache_data.clear()
    page = _page(monkeypatch, blocked_api_url)

    assert not page.exception
    assert "/scenarios/thresholds" not in paths
    assert paths.count("/opportunities/blocked") == 1
    caption = " ".join(element.value for element in page.caption)
    assert "20%" in caption and "$1.0m" in caption
