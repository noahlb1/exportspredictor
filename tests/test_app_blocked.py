"""The Blocked Opportunities page, run with AppTest through the router against a live local API.

Most tests use the Blocked world (`conftest.blocked_artefacts`): the fixture with its amounts × 1,000, so it has three
Blocked Opportunities, whisky and jewellery in India and jewellery in Hong Kong (cars in Hong Kong clear both thresholds
but have no room to grow). Whisky in India has a Tariff part (40% of its value); jewellery in Hong Kong has a 0%
tariff and Actual Exports above Export Potential. The fixture alone has none.

The page reads as a before and after (ADR 0011): each Product's Actual Exports, Export Potential and Export
Potential with a Trade Agreement, the rise, and how much of it the model associates with the tariff.
"""

import io
import json
import re
from dataclasses import replace

import httpx
import pandas as pd
import polars as pl
import pytest
import streamlit
from conftest import run_page
from streamlit import dataframe_util
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Dataframe

from exportspredictor import artefacts as store

PAGE = "pages/blocked_opportunities.py"
WHISKY, JEWELLERY, CARS, CRUDE_OIL = "220830", "711319", "870323", "270900"
JEWELLERY_NAME = "Articles of jewellery and parts thereof, of precious metal other than silver"
BLOCKED_PAIRS = [(WHISKY, "IND"), (JEWELLERY, "IND"), (JEWELLERY, "HKG")]
TOTAL = "Total value, illustrative"  # the left table's total heading: visibly Illustrative
PRODUCT_COLUMNS = [
    "Product",
    "Barrier",
    "Actual Exports",
    "Export Potential",
    "With a Trade Agreement",
    "Rise with agreement",
    "Rise %",
    "Tariff part",
    "Above Export Potential",
]
# Wording that would read a Blocked Opportunity value as a cause or a forecast.
CAUSAL = re.compile(
    r"(?<!not )\b(causes?|caused|will (raise|add|unlock|increase)|would (raise|add|unlock|increase)|unlocks?)\b",
    re.IGNORECASE,
)
# Wording that credits a rise to taking a tariff away (never on a 0% or unknown tariff).
REMOVING_THE_TARIFF = re.compile(r"remov\w* (the|a|its|any) \S*\s*tariff", re.IGNORECASE)
CHART = "vega_lite_chart"


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


def _row(api_url: str, hs6: str, iso3: str) -> dict:
    return next(row for row in _read(api_url)["opportunities"] if (row["hs6"], row["iso3"]) == (hs6, iso3))


def _click(page: AppTest, *, market: str | None = None, product: str | None = None, pair=None) -> AppTest:
    """Choose a Market (By Market), a Product (By Product) or a pair, (HS6, ISO3), as a click does: AppTest cannot
    click a table row, so the choice is written where the page remembers it."""
    if market is not None:
        page.session_state["blocked_group_iso3"] = market
    if product is not None:
        page.session_state["blocked_group_hs6"] = product
    if pair is not None:
        page.session_state["blocked_pair"] = pair
    return page.run()


def _choose(page: AppTest, hs6: str, iso3: str) -> AppTest:
    """Choose a Market and one of its Products, as clicking them would."""
    return _click(page, market=iso3, pair=(hs6, iso3))


def _view(page: AppTest, view: str) -> AppTest:
    page.segmented_control[0].set_value(view).run()
    return page


def _markets(page: AppTest) -> pd.DataFrame:
    """The left table: By Market, the Markets; By Product, the Products."""
    return page.dataframe[0].value


def _shown(table) -> pd.DataFrame:
    """A table in the columns it shows."""
    return table.value[list(table.proto.column_order)] if table.proto.column_order else table.value


def _products(page: AppTest) -> pd.DataFrame:
    """The right table, in the columns it shows: By Market, the chosen Market's Products."""
    return _shown(page.dataframe[1])


def _layout_tables(page: AppTest) -> list[int]:
    """Where the layout's own tables (left and right, or the Wide table) sit in the page's draw order; the panel's
    tables are not among them."""
    return [i for i, node in enumerate(page.main) if isinstance(node, Dataframe) and (node.key or "").startswith("blocked_")]


def _panel_box(page: AppTest) -> str:
    """The panel's neutral box naming the chosen pair, "Product → Market"."""
    [box] = [box.value for box in page.info if "→" in box.value]
    return box.split("\n")[0].strip("*")


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


def _panel_texts(page: AppTest) -> list[str]:
    return [str(getattr(node, "value", "")) for node in page.main if hasattr(node, "value")]


def _charts(page: AppTest) -> list[pd.DataFrame]:
    """The data of each chart on the page (Streamlit sends a chart's inline values as an Arrow table)."""
    return [dataframe_util.convert_arrow_bytes_to_pandas_df(chart.proto.data.data) for chart in page.get(CHART)]


# --- The shared layout: By Market, By Product, Wide table, the panel underneath -----------------------


def test_markets_are_listed_largest_total_first_with_the_top_markets_products_shown(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    assert not page.exception
    assert page.header[0].value == "Blocked Opportunities"
    assert page.segmented_control[0].value == "By Market"
    markets = _markets(page)
    assert list(markets.columns) == ["Market", "Blocked Opportunities", TOTAL]
    assert list(markets["Market"]) == ["India", "Hong Kong"]
    assert list(markets["Blocked Opportunities"]) == [2, 1]
    assert [round(value, 1) for value in markets[TOTAL]] == [25.6, 5.1]  # $18.5m + $7.1m; $5.1m
    helps = {name.removeprefix("name:"): column.get("help", "") for name, column in
             json.loads(page.dataframe[0].proto.columns).items()}
    assert "llustrative" in helps[TOTAL] and "under the filters" in helps["Blocked Opportunities"]
    # The top Market's Products are shown on open, under a line giving the Market's total as Illustrative.
    assert set(_products(page)["Product"]) == {"Whiskies", JEWELLERY_NAME}
    assert any("Total value, illustrative: $25.6m." in element.value for element in page.markdown)


def test_the_top_markets_top_pair_is_chosen_on_open_with_its_panel_under_the_tables(monkeypatch, blocked_api_url):
    # A pair is chosen on open, so its panel shows without a click.
    page = _page(monkeypatch, blocked_api_url)

    assert _panel_box(page) == "Whiskies → India"
    nodes = list(page.main)
    tables = _layout_tables(page)
    box = next(i for i, node in enumerate(nodes) if "Whiskies → India" in str(getattr(node, "value", "")))
    assert len(tables) == 2 and max(tables) < box
    assert any("figure" in chart.columns for chart in _charts(page))


def test_by_product_lists_products_with_their_totals_and_the_chosen_products_markets(monkeypatch, blocked_api_url):
    page = _view(_page(monkeypatch, blocked_api_url), "By Product")

    assert not page.exception
    products = _markets(page)
    assert list(products.columns) == ["Product", "Blocked Opportunities", TOTAL]
    assert list(products["Product"]) == ["Whiskies", JEWELLERY_NAME]
    assert list(products["Blocked Opportunities"]) == [1, 2]
    assert [round(value, 1) for value in products[TOTAL]] == [18.5, 12.2]  # jewellery: $7.1m + $5.1m
    assert list(_shown(page.dataframe[1])["Market"]) == ["India"]

    _click(page, product=JEWELLERY)

    right = _shown(page.dataframe[1])
    assert list(right.columns) == ["Market", *PRODUCT_COLUMNS[1:]]
    assert list(right["Market"]) == ["India", "Hong Kong"]
    assert any(f"**{JEWELLERY_NAME}**" in text and "2 Markets" in text for text in _texts(page))
    assert _panel_box(page) == f"{JEWELLERY_NAME} → India"

    _click(page, pair=(JEWELLERY, "HKG"))
    assert _panel_box(page) == f"{JEWELLERY_NAME} → Hong Kong"


def test_the_right_table_lists_every_pair_of_the_market_when_the_main_read_is_capped(monkeypatch, blocked_api_url):
    # The read limit lowered to 2 caps the main read at whisky and jewellery in India; Hong Kong's
    # jewellery is left out of it, yet listed once Hong Kong is chosen, from the read with Hong Kong's filter.
    monkeypatch.setattr("app.ui.read_limit", lambda: 2)
    page = _page(monkeypatch, blocked_api_url)
    assert any("top 2 of 3 Blocked Opportunities" in caption.value for caption in page.caption)
    capped = _read(blocked_api_url, limit=2)
    assert capped["summary"]["blocked_opportunities"] == 3
    assert (JEWELLERY, "HKG") not in [(row["hs6"], row["iso3"]) for row in capped["opportunities"]]

    _click(page, market="HKG")

    assert list(_products(page)["Product"]) == [JEWELLERY_NAME]
    assert list(_markets(page)["Market"]) == ["India", "Hong Kong"]  # the totals cover every pair, whatever the cap
    assert _panel_box(page) == f"{JEWELLERY_NAME} → Hong Kong"


def test_the_wide_table_lists_every_blocked_opportunity_with_its_product_and_market(monkeypatch, blocked_api_url):
    page = _view(_page(monkeypatch, blocked_api_url), "Wide table")

    assert len(_layout_tables(page)) == 1
    wide = _shown(page.dataframe[0])
    assert list(wide.columns) == ["Product", "Market", *PRODUCT_COLUMNS[1:]]
    rows = _read(blocked_api_url)["opportunities"]
    assert list(zip(wide["Product"], wide["Market"], strict=True)) == [(r["description"], r["market"]) for r in rows]
    assert _panel_box(page) == "Whiskies → India"


def test_the_chosen_pair_survives_a_view_switch_where_it_is_still_shown(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), JEWELLERY, "HKG")
    assert _panel_box(page) == f"{JEWELLERY_NAME} → Hong Kong"

    _view(page, "Wide table")
    assert _panel_box(page) == f"{JEWELLERY_NAME} → Hong Kong"

    _view(page, "By Product")  # the top Product, whisky, does not hold it: its top pair is shown
    assert _panel_box(page) == "Whiskies → India"
    _click(page, product=JEWELLERY)
    assert _panel_box(page) == f"{JEWELLERY_NAME} → Hong Kong"


def test_the_view_is_kept_in_the_url(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    assert "view" not in page.query_params  # the default view is left out

    _view(page, "By Product")
    assert page.query_params["view"] == "product"
    _view(page, "By Market")
    assert "view" not in page.query_params

    linked = _page(monkeypatch, blocked_api_url, view="wide")
    assert linked.segmented_control[0].value == "Wide table"
    assert len(_layout_tables(linked)) == 1


def test_the_view_is_kept_apart_from_the_leaderboards(monkeypatch, blocked_api_url):
    page = _view(_page(monkeypatch, blocked_api_url), "Wide table")

    page.query_params.clear()  # a browser clears them on a page switch; AppTest keeps them
    page.switch_page("pages/leaderboard.py").run()
    assert page.segmented_control[0].value == "By Market"

    page.switch_page(PAGE).run()
    assert page.segmented_control[0].value == "Wide table"  # kept for the session


def test_the_chosen_markets_products_have_the_stated_columns_in_baseline_year_figures(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    page.session_state["blocked_group_iso3"] = "IND"
    page.run()

    products = _products(page)
    assert list(products.columns) == PRODUCT_COLUMNS
    rows = [row for row in _read(blocked_api_url)["opportunities"] if row["iso3"] == "IND"]
    assert list(products["Product"]) == [row["description"] for row in rows]  # every one, by value (the rise)
    assert list(products["Barrier"]) == [row["barrier"] for row in rows]
    for field, column in [
        ("baseline_actual_exports_usd", "Actual Exports"),
        ("baseline_export_potential_usd", "Export Potential"),
        ("potential_with_agreement_usd", "With a Trade Agreement"),
        ("blocked_opportunity_value_usd", "Rise with agreement"),
        ("tariff_part_usd", "Tariff part"),
    ]:
        assert list(products[column]) == pytest.approx([row[field] / 1e6 for row in rows])
    assert list(products["Rise %"]) == pytest.approx([row["rise_share"] for row in rows])
    # Whisky in India, 2024: $38.1m exported, $61.6m Export Potential, $80.0m with an agreement, $7.4m Tariff part.
    whisky = products.iloc[0]
    assert whisky["Product"] == "Whiskies" and whisky["Barrier"] == "No agreement, tariff 150%"
    assert [round(whisky[c], 1) for c in PRODUCT_COLUMNS[2:6]] == [38.1, 61.6, 80.0, 18.5]
    assert whisky["Tariff part"] == pytest.approx(7.4, abs=0.05)
    assert any("**India**" in element.value and "2 Products" in element.value for element in page.markdown)


def test_product_headings_carry_tooltips_in_the_glossarys_terms(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    config = json.loads(page.dataframe[1].proto.columns)
    helps = {name.removeprefix("name:"): column.get("help", "") for name, column in config.items()}
    assert set(PRODUCT_COLUMNS) <= set(helps)
    assert "Export Potential with a Trade Agreement" in helps["With a Trade Agreement"]
    assert "llustrative" in helps["With a Trade Agreement"] and "llustrative" in helps["Rise with agreement"]
    assert "setting the Applied Tariff alone to 0" in helps["Tariff part"]
    assert "0% or unknown" in helps["Tariff part"]
    assert "above" in helps["Above Export Potential"] and "Export Potential" in helps["Above Export Potential"]
    assert "Rise share" in helps["Rise %"] and "Baseline Export Potential" in helps["Rise %"]


def test_a_pair_with_actual_exports_above_export_potential_is_marked(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    page.session_state["blocked_group_iso3"] = "HKG"
    page.run()

    products = _products(page)
    assert list(products["Product"]) == [JEWELLERY_NAME]
    assert list(products["Above Export Potential"]) == [True]
    assert list(products["Barrier"]) == ["No agreement, tariff 0%"]


def test_a_pair_with_no_room_to_grow_is_not_listed(monkeypatch, blocked_api_url):
    # Hong Kong: jewellery is listed; cars ($9.8m exported, above $4.3m + $1.1m with a Trade Agreement) are not.
    page = _page(monkeypatch, blocked_api_url)
    page.session_state["blocked_group_iso3"] = "HKG"
    page.run()

    products = _products(page)
    assert not [name for name in products["Product"] if name.startswith("Motor cars")]
    assert products["With a Trade Agreement"].iloc[0] == pytest.approx(23.9, abs=0.05)  # $18.8m + $5.1m


def test_products_can_be_ordered_by_tariff_part(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    order = page.sidebar.radio(key="blocked_rank_by")
    assert order.value == "blocked"  # the rise with agreement, by default

    order.set_value("tariff_part").run()

    assert page.query_params["rank_by"] == "tariff_part"
    assert any("Ordered by Tariff part" == button.label for button in page.main.button)
    by_tariff = [row["description"] for row in _read(blocked_api_url, rank_by="tariff_part")["opportunities"]
                 if row["iso3"] == "IND"]
    assert list(_products(page)["Product"]) == by_tariff
    assert list(_markets(page)["Market"]) == ["India", "Hong Kong"]  # Markets keep their order by total


def test_a_link_can_order_by_tariff_part(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url, rank_by="tariff_part")

    assert page.sidebar.radio(key="blocked_rank_by").value == "tariff_part"


# --- The page's own sentences -----------------------------------------------------------------------------------


def test_the_active_filters_line_states_the_count_and_the_illustrative_total(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    line = next(element.value for element in page.markdown if "No filters" in element.value)
    # 18.5 + 7.1 + 5.1 = $30.7m
    assert "3 Blocked Opportunities" in line and "illustrative" in line and "$30.7m" in line
    caption = " ".join(element.value for element in page.caption)
    assert "$ millions" in caption and "2024" in caption and "Scenario Builder" in caption
    assert "20%" in caption and "$1M" in caption
    # The note that totals understate, beside the summed values, with the share the artefact records.
    note = httpx.get(f"{blocked_api_url}/labels").json()["totals_note"]["text"]
    assert note in caption and "about 82% of Actual Exports" in note


def test_the_caption_says_with_no_uk_trade_agreement_in_force_and_states_room_to_grow(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    caption = next(element.value for element in page.caption if "room to grow" in element.value)
    assert httpx.get(f"{blocked_api_url}/labels").json()["blocked_rule"] in caption  # one wording (ADR 0012)
    assert "with no UK Trade Agreement in force" in caption
    assert "where none is in force" not in caption  # read as "no tariff"
    assert "room to grow" in caption and "Actual Exports" in caption
    assert not REMOVING_THE_TARIFF.search(caption)


def test_the_page_states_the_tariff_carried_forward_caveat_and_what_a_trade_agreement_means(
    monkeypatch, blocked_api_url
):
    labels = httpx.get(f"{blocked_api_url}/labels").json()

    page = _page(monkeypatch, blocked_api_url)  # no pair chosen: both are on the page itself

    texts = [element.value for element in [*page.caption, *page.markdown, *page.info]]
    assert any(labels["tariff_carried_forward"] in text for text in texts)
    assert any(labels["trade_agreement_meaning"] in text for text in texts)


def test_the_markets_table_explains_its_totals_and_links_the_market_to_a_scenario(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    note = next(element.value for element in page.caption if "Scenario Effect" in element.value)
    assert "only" in note and "every pair" in note
    [link] = [link for link in page.get("page_link") if link.proto.page == "Scenario_Builder"]
    assert link.proto.label == "Scenario Builder: a Trade Agreement with India"  # names the page it opens
    assert link.proto.query_string == "add=IND"  # the Scenario Builder prefills it (`?add=`)

    # Under the Markets table: after it, before the chosen Market's Products.
    nodes = list(page.main)
    tables = _layout_tables(page)
    at = next(i for i, node in enumerate(nodes) if node.type == "page_link" and node.proto.page == "Scenario_Builder")
    assert tables[0] < at < tables[1]

    page.session_state["blocked_group_iso3"] = "HKG"
    page.run()
    [link] = [link for link in page.get("page_link") if link.proto.page == "Scenario_Builder"]
    assert "Hong Kong" in link.proto.label and link.proto.query_string == "add=HKG"

    _view(page, "By Product")  # a Product's Markets are no one Scenario: no note or link
    assert not [link for link in page.get("page_link") if link.proto.page == "Scenario_Builder"]
    assert not [element for element in page.caption if "Scenario Effect" in element.value]


# --- The panel: a neutral box, the before-and-after chart, the split, then the caveats ----------------------------


def test_the_panel_opens_with_a_neutral_box_naming_the_pair_and_its_barrier(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")

    assert not page.exception
    assert not [box for box in page.warning if "Whiskies" in box.value]  # no warning box
    [box] = [box for box in page.info if "Whiskies → India" in box.value]
    assert "No agreement, tariff 150%" in box.value


def test_the_panel_shows_actual_exports_export_potential_and_the_level_with_an_agreement_as_one_chart(
    monkeypatch, blocked_api_url
):
    page = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")

    [before_after] = [chart for chart in _charts(page) if "figure" in chart.columns]
    assert list(before_after["figure"]) == [
        "Actual Exports", "Export Potential", "Export Potential with a Trade Agreement"
    ]
    assert [round(value, 1) for value in before_after["value"]] == [38.1, 61.6, 80.0]  # $ millions, 2024
    assert list(before_after["text"]) == ["$38.1m", "$61.6m", "$80.0m"]
    assert not page.metric  # the chart replaces the three metrics


def test_a_tariff_rows_split_sentence_names_the_tariff_part_and_the_agreement_part(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")

    [split] = [text for text in _panel_texts(page) if "$18.5m" in text and "Tariff part" in text]
    assert "$7.4m" in split and "150%" in split and "$11.1m" in split and "Agreement part" in split
    assert "associates" in split


def test_a_0_percent_rows_text_never_speaks_of_removing_the_tariff(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), JEWELLERY, "HKG")

    assert not page.exception
    assert not [text for text in _texts(page) if REMOVING_THE_TARIFF.search(text)]
    [split] = [text for text in _panel_texts(page) if "$5.1m" in text and "average" in text]
    assert "0%" in split and "entirely" in split and "associat" in split


def test_an_above_export_potential_row_says_the_model_already_rates_current_trade_as_strong(
    monkeypatch, blocked_api_url
):
    page = _choose(_page(monkeypatch, blocked_api_url), JEWELLERY, "HKG")

    assert any("already above" in text and "Export Potential" in text for text in _panel_texts(page))
    whisky = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")
    assert not any("already above" in text for text in _panel_texts(whisky))


def test_an_unknown_tariff_rows_text_never_speaks_of_removing_the_tariff(
    monkeypatch, blocked_artefacts, tmp_path, live_api
):
    # Jewellery in India's tariff is made unknown (its Tariff part is already 0 in the Blocked world).
    unknown = (pl.col("hs6") == JEWELLERY) & (pl.col("iso3") == "IND")
    features = blocked_artefacts.baseline_features.with_columns(
        pl.when(unknown).then(None).otherwise(pl.col("log_applied_tariff")).alias("log_applied_tariff")
    )
    store.save(replace(blocked_artefacts, baseline_features=features), tmp_path)

    with live_api(tmp_path) as api_url:
        assert _row(api_url, JEWELLERY, "IND")["barrier"] == "No agreement, tariff unknown"
        page = _choose(_page(monkeypatch, api_url), JEWELLERY, "IND")

    assert not page.exception
    assert not [text for text in _texts(page) if REMOVING_THE_TARIFF.search(text)]
    [split] = [text for text in _panel_texts(page) if "not known" in text]
    assert "average" in split and "entirely" in split


def test_the_panel_opens_with_the_box_then_the_chart_then_the_split_then_the_caveats(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), JEWELLERY, "HKG")  # a Re-export Hub: it has a caveat

    nodes = list(page.main)  # depth first, in the order the page draws them
    texts = [str(getattr(node, "value", "")) for node in nodes]
    box = next(i for i, text in enumerate(texts) if "→ Hong Kong" in text)
    chart = next(i for i, node in enumerate(nodes) if node.type == CHART)
    split = next(i for i, text in enumerate(texts) if "$5.1m" in text and "average" in text)
    caveat = next(i for i, text in enumerate(texts) if text.startswith("Re-export Hub"))
    assert box < chart < split < caveat
    assert not [element for element in page.subheader if "Hong Kong" in element.value]  # the box names the pair


def test_the_facts_repeat_nothing_the_box_or_chart_already_says(monkeypatch, blocked_api_url):
    page = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")

    facts = [element.value for element in page.markdown if element.value.startswith("**") and ":**" in element.value]
    names = [fact.split(":**")[0].strip("*") for fact in facts]
    assert "Potential Range" in names and "Opportunity Type" in names
    for repeated in ("Applied Tariff, 2024", "UK Trade Agreement in force", "Blocked Opportunity value",
                     "Relative Export Gap"):
        assert repeated not in names


@pytest.mark.parametrize(("hs6", "iso3"), BLOCKED_PAIRS)
def test_no_export_gap_appears_anywhere_on_the_page(monkeypatch, blocked_api_url, hs6, iso3):
    page = _choose(_page(monkeypatch, blocked_api_url), hs6, iso3)

    assert not page.exception
    assert [text for text in _texts(page) if "Export Gap" in text] == []


@pytest.mark.parametrize("view", ["By Product", "Wide table"])
def test_no_export_gap_appears_in_the_other_views(monkeypatch, blocked_api_url, view):
    page = _view(_page(monkeypatch, blocked_api_url), view)

    assert not page.exception
    assert [text for text in _texts(page) if "Export Gap" in text] == []
    assert not [text for text in _texts(page) if CAUSAL.search(text)]


# --- Filters ----------------------------------------------------------------------------------------------------


def test_sidebar_filters_narrow_the_markets_and_reset_restores_them(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    page.sidebar.checkbox(key="blocked_hide_hubs").check().run()
    assert list(_markets(page)["Market"]) == ["India"]
    assert page.query_params["hide_hubs"] == "true"
    assert any("Re-export Hubs hidden" == button.label for button in page.main.button)

    page.sidebar.button(key="blocked_reset_filters").click().run()
    assert list(_markets(page)["Market"]) == ["India", "Hong Kong"]
    assert "hide_hubs" not in page.query_params


def test_a_filter_chip_removes_only_its_filter(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    page.sidebar.checkbox(key="blocked_hide_hubs").check().run()

    next(button for button in page.main.button if button.label == "Re-export Hubs hidden").click().run()

    assert list(_markets(page)["Market"]) == ["India", "Hong Kong"]
    assert page.session_state["blocked_hide_hubs"] is False and "hide_hubs" not in page.query_params
    assert any("No filters" in element.value for element in page.markdown)


def test_the_page_has_an_order_but_no_market_type_or_agreement_filter(monkeypatch, blocked_api_url):
    # The page has an order (by the rise or by Tariff part), but no Market filter.
    page = _page(monkeypatch, blocked_api_url)

    keys = {widget.key for widget in [*page.sidebar.selectbox, *page.sidebar.radio, *page.sidebar.number_input]}
    assert keys == {"blocked_sector", "blocked_region", "blocked_min_value_usd", "blocked_rank_by"}


def test_a_product_search_keeps_that_products_blocked_opportunities(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)

    page.sidebar.text_input(key="blocked_product_query").input("jewellery").run()

    assert list(_markets(page)["Market"]) == ["India", "Hong Kong"]
    assert list(_markets(page)["Blocked Opportunities"]) == [1, 1]
    assert page.query_params["product"] == JEWELLERY


def test_url_parameters_set_the_filters(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url, region="South Asia", min_value_usd="10000000")

    assert list(_markets(page)["Market"]) == ["India"]
    assert list(_products(page)["Product"]) == ["Whiskies"]


def test_filters_are_kept_apart_from_the_leaderboards(monkeypatch, blocked_api_url):
    page = _page(monkeypatch, blocked_api_url)
    page.sidebar.checkbox(key="blocked_hide_hubs").check().run()

    page.query_params.clear()  # a browser clears them on a page switch; AppTest keeps them
    page.switch_page("pages/leaderboard.py").run()

    assert page.session_state["hide_hubs"] is False
    assert page.session_state["blocked_hide_hubs"] is True


# --- CSV, note, language and edge cases ---------------------------------------------------------------------------


def test_csv_holds_every_filtered_blocked_opportunity_in_full_dollars(monkeypatch, blocked_api_url, downloads):
    _page(monkeypatch, blocked_api_url)

    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str})
    rows = _read(blocked_api_url)["opportunities"]
    assert list(zip(exported["HS6"], exported["Market"], strict=True)) == [(r["hs6"], r["market"]) for r in rows]
    value = next(column for column in exported.columns if column.startswith("Blocked Opportunity value"))
    assert "illustrative" in value and "2024" in value and "USD" in value
    assert list(exported[value]) == pytest.approx([row["blocked_opportunity_value_usd"] for row in rows])
    assert not [column for column in exported.columns if "Export Gap" in column]
    with_agreement = next(column for column in exported.columns if column.startswith("Export Potential with"))
    assert list(exported[with_agreement]) == pytest.approx([row["potential_with_agreement_usd"] for row in rows])
    # The Barrier, both parts, the rise as a share and the mark.
    assert list(exported["Barrier"]) == [row["barrier"] for row in rows]
    for prefix, field in [("Tariff part", "tariff_part_usd"), ("Agreement part", "agreement_part_usd")]:
        column = next(column for column in exported.columns if column.startswith(prefix))
        assert "illustrative" in column and "USD" in column
        assert list(exported[column]) == pytest.approx([row[field] for row in rows])
    assert list(exported["Rise share"]) == pytest.approx([row["rise_share"] for row in rows])
    assert list(exported["Above Export Potential"]) == [row["above_export_potential"] for row in rows]


def test_the_illustrative_note_sits_in_an_expander_under_the_tables(monkeypatch, blocked_api_url):
    note = _read(blocked_api_url)["note"]

    page = _page(monkeypatch, blocked_api_url)

    [expander] = [e for e in page.expander if note["label"] in e.label]
    text = " ".join(element.value for element in [*expander.markdown, *expander.info])
    assert note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["text"] in text


@pytest.mark.parametrize(("hs6", "iso3"), BLOCKED_PAIRS)
def test_every_text_uses_associational_language(monkeypatch, blocked_api_url, hs6, iso3):
    page = _choose(_page(monkeypatch, blocked_api_url), hs6, iso3)

    texts = _texts(page)
    assert not [text for text in texts if CAUSAL.search(text)]
    about_value = [
        element.value
        for element in [*page.caption, *page.warning, *page.info, *page.markdown]
        if "Blocked Opportunity value" in element.value or "rise" in element.value
    ]
    about_value = [text for text in about_value if "Export Potential" in text]
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


def test_the_page_makes_one_read_and_one_for_the_chosen_market_and_states_the_thresholds(monkeypatch, blocked_api_url):
    # The right table is a second read with the chosen Market's filter.
    from app import client

    reads = []
    original = client.get

    def record(path, **params):
        reads.append((path, params.get("market")))
        return original(path, **params)

    monkeypatch.setattr(client, "get", record)
    streamlit.cache_data.clear()
    page = _page(monkeypatch, blocked_api_url)

    assert not page.exception
    assert sorted(market or "" for path, market in reads if path == "/opportunities/blocked") == ["", "IND"]
    caption = " ".join(element.value for element in page.caption)
    assert "20%" in caption and "$1M" in caption


def test_the_page_sends_no_one_to_try_other_thresholds(monkeypatch, blocked_api_url):
    # The thresholds are fixed (ADR 0012): there are no other thresholds to try.
    page = _page(monkeypatch, blocked_api_url)

    assert not page.exception
    assert "Try other thresholds" not in " ".join(_texts(page))


def test_the_minimum_value_filter_reads_as_a_filter_on_the_list(monkeypatch, blocked_api_url):
    # Worded so it isn't taken for the fixed thresholds: it only narrows the list shown.
    page = _page(monkeypatch, blocked_api_url)

    minimum = page.sidebar.number_input(key="blocked_min_value_usd")
    assert minimum.label.startswith("Minimum Blocked Opportunity value")
    assert "threshold" not in minimum.label.lower()
    assert "Narrows the list only" in minimum.help and "fixed" in minimum.help


def test_the_tariff_carried_forward_caveat_is_shown_once_with_a_pair_chosen(monkeypatch, blocked_api_url):
    # The page states it under the summary; the panel does not repeat it as a warning.
    caveat = httpx.get(f"{blocked_api_url}/labels").json()["tariff_carried_forward"]
    explained = httpx.get(f"{blocked_api_url}/explain/{WHISKY}/IND").json()
    assert explained["tariff_caveat"]  # whisky's 2024 tariff is carried forward, so the panel would warn

    page = _choose(_page(monkeypatch, blocked_api_url), WHISKY, "IND")

    assert len([text for text in _panel_texts(page) if caveat in text]) == 1
    assert not [box for box in page.warning if caveat in box.value]


def test_the_order_choices_are_the_apis_blocked_orders(monkeypatch, blocked_api_url):
    orders = httpx.get(f"{blocked_api_url}/opportunities/blocked/rank-by").json()
    page = _page(monkeypatch, blocked_api_url)

    order = page.sidebar.radio(key="blocked_rank_by")
    assert order.options == [entry["title"] for entry in orders]
    assert all(entry["description"] in order.proto.help for entry in orders)


# --- The panel as the shared panel module's function (the Scenario Builder calls it too) ---------------


def _render_blocked_panel(hs6: str, iso3: str) -> None:
    """A script for AppTest: one Blocked Opportunity's panel, drawn by the shared panel module alone."""
    import httpx

    from app import client, panel

    rows = httpx.get(f"{client.api_url()}/opportunities/blocked").json()["opportunities"]
    [row] = [row for row in rows if (row["hs6"], row["iso3"]) == (hs6, iso3)]
    panel.blocked_opportunity(row)


def test_the_shared_panel_module_draws_a_blocked_opportunitys_panel_on_its_own(monkeypatch, blocked_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", blocked_api_url)
    page = AppTest.from_function(_render_blocked_panel, args=(WHISKY, "IND"), default_timeout=30).run()

    assert not page.exception
    [box] = [box for box in page.info if "Whiskies → India" in box.value]
    assert "No agreement, tariff 150%" in box.value
    [before_after] = [chart for chart in _charts(page) if "figure" in chart.columns]
    assert list(before_after["text"]) == ["$38.1m", "$61.6m", "$80.0m"]
    assert any("$7.4m" in element.value and "Tariff part" in element.value for element in page.markdown)
