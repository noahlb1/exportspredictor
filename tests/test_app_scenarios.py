"""The Streamlit Scenario Builder, run with AppTest against a live API on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with India.

AppTest cannot click a table row, so a test chooses a Market or a pair by setting the page's remembered selection
(`scenario_market`, `scenario_drill_pair`, `scenario_blocked_pair`), as a click does.
"""

import json
from dataclasses import replace

import httpx
import pandas as pd
import polars as pl
import pytest
import streamlit
from conftest import run_page, serve_api
from streamlit.testing.v1 import AppTest

from app import client
from exportspredictor import artefacts as store

SCENARIOS = "pages/scenarios.py"
BLOCKED_PAGE = "pages/blocked_opportunities.py"
NOTE_EXPANDER = ": what these figures are read against"  # the end of the Scenario note expander's label
# The page's tables, in order once a Scenario has run.
MARKETS, DRILL_DOWN, PRODUCTS, BLOCKED = range(4)
INDIA = {"add_agreements": ["IND"], "limit": 100}
WHISKY = "220830"
# The Blocked Opportunities page's Products table (`app.tables.blocked_products_frame`), with a Market column here.
BLOCKED_COLUMNS = [
    "Product",
    "Market",
    "Barrier",
    "Actual Exports",
    "Export Potential",
    "With a Trade Agreement",
    "Rise with agreement",
    "Rise %",
    "Tariff part",
    "Above Export Potential",
]


@pytest.fixture
def api_url(fixture_api_url):
    return fixture_api_url


@pytest.fixture(scope="module")
def cptpp_india_api_url(blocked_artefacts, tmp_path_factory):
    """The Blocked world with India made a CPTPP member, so the CPTPP preset adds a Trade Agreement with India."""
    features = blocked_artefacts.baseline_features
    india = pl.when(pl.col("iso3") == "IND").then(1).otherwise(pl.col("cptpp_d"))
    root = tmp_path_factory.mktemp("cptpp_india_artefacts")
    store.save(
        replace(blocked_artefacts, baseline_features=features.with_columns(india.cast(features.schema["cptpp_d"]).alias("cptpp_d"))),
        root,
    )
    with serve_api(root) as url:
        yield url


def _scenarios_page(monkeypatch, api_url) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(SCENARIOS)


def _run_cptpp(monkeypatch, api_url) -> AppTest:
    page = _scenarios_page(monkeypatch, api_url)
    page.selectbox(key="preset").set_value("cptpp").run()
    page.button(key="run").click().run()
    assert not page.exception
    return page


def _run_india(monkeypatch, api_url) -> AppTest:
    page = _scenarios_page(monkeypatch, api_url)
    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["IND"]).run()
    page.button(key="run").click().run()
    assert not page.exception
    return page


def _page_text(page: AppTest) -> str:
    return " ".join(
        str(element.value) for element in [*page.markdown, *page.caption, *page.info, *page.warning, *page.subheader]
    )


def _pair_links(page: AppTest) -> list:
    """The links a pair's box opens (to the Leaderboard), not the page's links to the Blocked Opportunities page."""
    return [link for link in page.page_link if "Leaderboard" in link.proto.label]


def _stale(page: AppTest) -> list[str]:
    return [element.value for element in [*page.info, *page.warning] if "Run Scenario" in element.value]


def test_the_summed_scenario_effect_carries_the_totals_note(monkeypatch, api_url):
    # The note that totals understate: every summed figure is read against the artefact's share of Actual Exports.
    note = httpx.get(f"{api_url}/labels").json()["totals_note"]["text"]

    page = _run_cptpp(monkeypatch, api_url)

    assert "of Actual Exports" in note
    assert any(note in element.value for element in page.caption)


def test_the_page_is_empty_until_a_scenario_is_run(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    assert not page.exception
    assert not page.dataframe and not page.metric
    assert "Change per Market" not in _page_text(page)

    page.selectbox(key="preset").set_value("cptpp").run()
    assert not page.dataframe  # choosing a Scenario does not run it

    page.button(key="run").click().run()
    assert page.dataframe and page.metric
    assert "Change per Market" in _page_text(page)


def test_the_preset_selectbox_offers_exactly_the_presets_the_api_returns(monkeypatch, api_url):
    # The fixture Baseline has no EU member Market and the UK in no Trading Bloc, so only CPTPP is offered.
    page = _scenarios_page(monkeypatch, api_url)

    assert page.selectbox(key="preset").options == ["CPTPP"]
    assert [preset["title"] for preset in httpx.get(f"{api_url}/scenarios/presets").json()] == ["CPTPP"]


def test_the_builder_has_no_sidebar_controls(monkeypatch, api_url):
    page = _run_cptpp(monkeypatch, api_url)

    sidebar = page.sidebar
    assert not [*sidebar.radio, *sidebar.selectbox, *sidebar.multiselect, *sidebar.number_input, *sidebar.button]


def test_a_preset_scenario_shows_an_associational_summary_and_markets(monkeypatch, api_url):
    page = _run_cptpp(monkeypatch, api_url)

    assert "associates" in " ".join(element.value for element in page.markdown)
    assert set(page.dataframe[MARKETS].value["Market"]) == {"Japan", "Canada"}


def test_a_line_says_when_the_controls_differ_from_the_result_shown(monkeypatch, api_url):
    page = _run_cptpp(monkeypatch, api_url)
    assert not _stale(page)

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    assert _stale(page)
    assert page.dataframe  # the earlier result stays on screen under the line

    page.radio(key="kind").set_value("Trading Bloc preset").run()
    page.selectbox(key="preset").set_value("cptpp").run()
    assert not _stale(page)  # back to the controls of the result shown

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["IND"]).run()
    page.button(key="run").click().run()
    assert not _stale(page)


def test_the_top_market_drives_the_drill_down_on_open(monkeypatch, api_url):
    top = httpx.post(f"{api_url}/scenarios", json={"preset": "cptpp", "limit": 100}).json()["markets"][0]

    page = _run_cptpp(monkeypatch, api_url)

    assert f"Pairs most affected in {top['market']}" in _page_text(page)
    assert not [box for box in page.selectbox if box.key == "drill_market"]  # no Market picker


def test_a_selected_market_drives_the_drill_down(monkeypatch, api_url):
    posted = []
    original = client.post

    def record(path, body):
        posted.append(body)
        return original(path, body)

    monkeypatch.setattr(client, "post", record)
    streamlit.cache_data.clear()  # earlier tests may have cached this Scenario's responses
    page = _run_cptpp(monkeypatch, api_url)
    page.session_state["scenario_market"] = "CAN"
    page.run()

    assert not page.exception
    assert any(body.get("market") == "CAN" for body in posted)
    assert "Pairs most affected in Canada" in _page_text(page)
    canada = httpx.post(f"{api_url}/scenarios", json={"preset": "cptpp", "market": "CAN", "limit": 50}).json()
    assert list(page.dataframe[DRILL_DOWN].value["Product"]) == [row["description"] for row in canada["top_pairs"]]


def test_drill_down_pairs_are_largest_effect_first(monkeypatch, api_url):
    page = _run_india(monkeypatch, api_url)

    pairs = page.dataframe[DRILL_DOWN].value
    assert len(pairs) == 5
    effects = list(pairs["Scenario Effect"])
    assert effects == sorted(effects, reverse=True)


def test_a_selected_pair_shows_its_box_with_a_link_to_the_leaderboard(monkeypatch, api_url):
    pair = httpx.post(f"{api_url}/scenarios", json={**INDIA, "market": "IND", "limit": 50}).json()["top_pairs"][1]

    page = _run_india(monkeypatch, api_url)
    assert not _pair_links(page)  # no pair chosen yet
    page.session_state["scenario_drill_pair"] = {"hs6": pair["hs6"], "iso3": pair["iso3"]}
    page.run()

    assert not page.exception
    [link] = _pair_links(page)
    assert "Leaderboard" in link.proto.label
    assert sorted(link.proto.query_string.split("&")) == sorted([f"product={pair['hs6']}", f"market={pair['iso3']}"])
    labels = [metric.label for metric in page.metric]
    assert pair["description"] in _page_text(page)
    assert any("Scenario Effect" in label for label in labels)
    assert {"Tariff, Baseline", "Tariff, Scenario"} <= set(labels)


def test_a_remembered_pair_not_in_the_rows_shows_no_box(monkeypatch, api_url):
    page = _run_india(monkeypatch, api_url)
    page.session_state["scenario_drill_pair"] = {"hs6": "000000", "iso3": "IND"}
    page.run()

    assert not page.exception
    assert not _pair_links(page)


def test_money_columns_are_numeric_millions_with_the_unit_stated_once(monkeypatch, api_url):
    drilled = httpx.post(f"{api_url}/scenarios", json={**INDIA, "market": "IND", "limit": 50}).json()

    page = _run_india(monkeypatch, api_url)

    pairs = page.dataframe[DRILL_DOWN].value
    assert pd.api.types.is_float_dtype(pairs["Scenario Effect"])
    assert list(pairs["Scenario Effect"]) == pytest.approx([row["effect_usd"] / 1e6 for row in drilled["top_pairs"]])
    for frame in page.dataframe:
        for column in ("Baseline", "Scenario"):
            assert pd.api.types.is_float_dtype(frame.value[column])
        assert not [heading for heading in frame.value.columns if "USD" in heading or "$" in heading]
    assert "$ millions" in _page_text(page)


def test_short_headings_carry_tooltips(monkeypatch, api_url):
    word = httpx.get(f"{api_url}/labels").json()["illustrative"]["badge"].lower()

    page = _run_india(monkeypatch, api_url)

    pairs = page.dataframe[DRILL_DOWN]
    assert pairs.value.columns[0] == "Product"
    for heading in (
        "Scenario Effect", "Effect, % of Baseline", "Baseline", "Scenario", "Tariff, Baseline %", "Tariff, Scenario %",
    ):
        assert heading in pairs.value.columns
    config = json.loads(pairs.proto.columns)
    effect = next(value for key, value in config.items() if "Scenario Effect" in key)
    assert word in effect["help"].lower()
    assert "Pairs changed" in page.dataframe[MARKETS].value.columns


# --- A Scenario's Blocked Opportunities (ADR 0012): the Blocked Opportunities page's rows, table and panel ----


def _run_tariff(monkeypatch, api_url) -> AppTest:
    page = _scenarios_page(monkeypatch, api_url)
    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["IND"]).run()
    page.button(key="run").click().run()
    assert not page.exception
    return page


def _blocked_metrics(page: AppTest) -> list[str]:
    return [metric.value for metric in page.metric if metric.label == "Blocked Opportunities"]


def _blocked_caption(page: AppTest) -> str:
    return next(element.value for element in page.caption if "the same rule and figures" in element.value)


def _blocked_heading_help(page: AppTest) -> str:
    [heading] = [element for element in page.subheader if element.value == "Blocked Opportunities"]
    return heading.proto.help


def test_adding_india_shows_the_blocked_opportunities_pages_rows_in_its_products_table(
    monkeypatch, blocked_api_url
):
    result = httpx.post(f"{blocked_api_url}/scenarios", json=INDIA).json()

    page = _run_india(monkeypatch, blocked_api_url)

    assert _blocked_metrics(page) == [f"{result['blocked_count']:,}"] == ["2"]
    assert "Blocked Opportunities" in [element.value for element in page.subheader]
    table = page.dataframe[BLOCKED].value
    assert list(table.columns) == BLOCKED_COLUMNS
    assert list(table["Product"]) == [row["description"] for row in result["blocked_opportunities"]]
    assert set(table["Market"]) == {"India"}
    assert list(table["Rise with agreement"]) == pytest.approx(
        [row["blocked_opportunity_value_usd"] / 1e6 for row in result["blocked_opportunities"]]
    )


def test_choosing_a_scenario_blocked_opportunity_opens_the_shared_before_and_after_panel(monkeypatch, blocked_api_url):
    page = _run_india(monkeypatch, blocked_api_url)
    page.session_state["scenario_blocked_pair"] = (WHISKY, "IND")
    page.run()

    assert not page.exception
    [box] = [box for box in page.info if "Whiskies → India" in box.value]
    assert "No agreement, tariff 150%" in box.value  # the panel `app.panel.blocked_opportunity` leads with
    assert any("Tariff part" in element.value for element in page.markdown)


@pytest.mark.parametrize("run", [_run_tariff, _run_cptpp], ids=["tariff", "cptpp-adds-none"])
def test_a_scenario_adding_no_trade_agreement_shows_one_neutral_line_instead(monkeypatch, blocked_api_url, run):
    # In the fixture Baseline, every CPTPP member (Japan, Canada) already has a Trade Agreement.
    page = run(monkeypatch, blocked_api_url)

    assert not _blocked_metrics(page)
    assert "Blocked Opportunities" not in [element.value for element in page.subheader]
    assert len(page.dataframe) == BLOCKED  # Change per Market, the drill-down and Top Products gaining only
    [line] = [element.value for element in page.markdown if "this Scenario adds none" in element.value]
    assert "Blocked Opportunities are measured by adding a Trade Agreement" in line
    assert line == (
        "Blocked Opportunities are measured by adding a Trade Agreement where none is in force, and this Scenario "
        "adds none."
    )  # the rule is not needed for an empty case
    [link] = [link for link in page.page_link if "Blocked Opportunities" in link.proto.label]
    assert link.proto.page.endswith(BLOCKED_PAGE) or "Blocked_Opportunities" in link.proto.page


def test_the_blocked_opportunities_caption_states_the_rule_the_hidden_tiers_and_why_the_totals_differ(
    monkeypatch, blocked_api_url
):
    hidden = [tier for tier in httpx.get(f"{blocked_api_url}/caveats").json() if tier["hidden_by_default"]]

    page = _run_india(monkeypatch, blocked_api_url)

    caption = _blocked_caption(page)
    assert "Blocked Opportunities page" in caption and "Scenario Export Potential" not in caption
    assert hidden and all(tier["plural"] in caption for tier in hidden)
    assert "covers every pair" in caption and "sums only these" in caption
    assert "preset" not in caption  # a custom addition: its Scenario Effects are the pairs' values
    assert "illustrative" in caption.lower()
    rule = _blocked_heading_help(page)  # the rule moves to the term's `?`
    assert "20%" in rule and "$1M" in rule and "associates" in rule
    [link] = [link for link in page.page_link if "hidden" in link.proto.label]
    assert {f"{tier['hide_param']}=false" for tier in hidden} == set(link.proto.query_string.split("&"))


def test_a_preset_adding_an_agreement_says_its_scenario_effect_can_differ_from_a_pairs_value(
    monkeypatch, cptpp_india_api_url
):
    page = _run_cptpp(monkeypatch, cptpp_india_api_url)

    assert _blocked_metrics(page) == ["2"]
    caption = _blocked_caption(page)
    assert "preset also changes other conditions" in caption
    assert "can differ from that pair's Blocked Opportunity value" in caption


def test_a_scenario_that_changes_nothing_says_so(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["JPN"]).run()
    page.button(key="run").click().run()

    assert not page.exception
    # Nothing is wrong, so the API's summary is a neutral box, not a warning, and there is no table.
    assert any("changes nothing" in box.value for box in page.info)
    assert not page.warning
    assert not page.dataframe


def test_a_scenario_that_is_not_valid_stays_a_warning(monkeypatch, api_url):
    # Nothing added or removed: the Advisor must change something, so this one is a warning.
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.button(key="run").click().run()

    assert not page.exception
    assert any("This Scenario is not valid" in box.value for box in page.warning)
    assert not page.dataframe


def test_the_note_is_not_shown_before_a_run(monkeypatch, api_url, fixture_artefacts):
    page = _scenarios_page(monkeypatch, api_url)

    note_text = httpx.get(f"{api_url}/labels").json()["illustrative"]["text"]

    assert not page.exception
    text = _page_text(page)
    assert fixture_artefacts.evaluation["event_backtest"]["text"] not in text
    assert note_text not in text


def test_a_run_shows_the_apis_scenario_note_in_the_blocked_opportunities_pages_expander(monkeypatch, blocked_api_url):
    # The note sits closed under the figures, in the expander the Blocked Opportunities page uses.
    note = httpx.post(f"{blocked_api_url}/scenarios", json=INDIA).json()["note"]
    page = _run_india(monkeypatch, blocked_api_url)
    # The Blocked Opportunities page's note expander, found by its label: a pair's panel, with
    # expanders of its own, is open on that page too.
    [label] = [box.label for box in run_page(BLOCKED_PAGE).expander if box.label.endswith(NOTE_EXPANDER)]

    [expander] = [box for box in page.expander if box.label == label]
    text = " ".join(str(element.value) for element in [*expander.info, *expander.markdown])
    assert note["label"] in text and note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["text"] in text
    assert note["blend"]["text"] in text


def test_no_caption_repeats_the_association_caveat_and_the_amounts_caption_names_the_clickable_tables(
    monkeypatch, api_url
):
    # The subtitle and the Illustrative note carry the caveat; only Markets and pairs can be clicked.
    page = _run_india(monkeypatch, api_url)

    captions = [element.value for element in page.caption]
    assert not [caption for caption in captions if "not causal" in caption or "learned from past trade" in caption]
    [amounts] = [caption for caption in captions if caption.startswith("Amounts in")]
    assert "Click a Market or a pair" in amounts
    assert "any value" not in amounts


def _result_order(node) -> list[str]:
    """The result's parts in the order the page renders them, an expander as one part (its contents not walked)."""
    kind = type(node).__name__
    value = str(getattr(node, "value", ""))
    if kind == "Expander":
        return ["Illustrative expander"]
    if kind == "Metric":
        return [f"metric: {node.label.split(',')[0]}"]
    if kind == "Subheader":
        return [value]
    if kind == "Markdown" and value.startswith("The model associates this Scenario"):
        return ["summary"]
    if kind == "Markdown" and "a rise for" in value:
        return ["raised and lowered"]
    if kind == "Markdown" and "this Scenario adds none" in value:
        return ["neutral line"]
    if kind == "Info" and "trained on" in value:
        return ["clipped-tariff note"]
    children = getattr(node, "children", {})
    return [part for index in sorted(children) for part in _result_order(children[index])]


def test_a_tariff_result_renders_summary_note_metrics_and_tables_in_reading_order(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)
    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["JPN"]).run()
    page.number_input(key="tariff_rate").set_value(1000.0).run()  # beyond the training range: the note applies
    page.button(key="run").click().run()

    assert not page.exception
    assert _result_order(page.main) == [
        "summary",
        "clipped-tariff note",
        "raised and lowered",
        "metric: Scenario Effect",
        "metric: Share of Baseline",
        "Illustrative expander",
        "Change per Market",
        "Pairs most affected in Japan",
        "Top Products gaining",
        "neutral line",
    ]


def test_a_result_adding_an_agreement_renders_summary_metrics_and_tables_in_reading_order(monkeypatch, blocked_api_url):
    page = _run_india(monkeypatch, blocked_api_url)

    assert _result_order(page.main) == [
        "summary",
        "metric: Scenario Effect",
        "metric: Share of Baseline",
        "metric: Blocked Opportunities",
        "Illustrative expander",
        "Change per Market",
        "Pairs most affected in India",
        "Top Products gaining",
        "Blocked Opportunities",
    ]


def test_the_scenario_effect_figure_says_illustrative(monkeypatch, api_url):
    word = httpx.get(f"{api_url}/labels").json()["illustrative"]["badge"].lower()

    page = _run_india(monkeypatch, api_url)

    effects = [metric.label for metric in page.metric if "Scenario Effect" in metric.label]
    assert effects and all(word in label for label in effects)


@pytest.mark.parametrize("kind", ["Trading Bloc preset", "Custom Trade Agreements", "Tariff change"])
def test_the_builder_has_no_threshold_inputs(monkeypatch, api_url, kind):
    # The thresholds are fixed (ADR 0012); the only number an Advisor types is a tariff rate.
    page = _scenarios_page(monkeypatch, api_url)
    page.radio(key="kind").set_value(kind).run()

    assert [box.key for box in page.number_input] == (["tariff_rate"] if kind == "Tariff change" else [])
    assert "threshold" not in " ".join(element.value for element in page.markdown).lower()


@pytest.mark.parametrize("picker", ["add", "remove", "tariff_markets"])
def test_every_market_picker_offers_exactly_the_markets_a_scenario_accepts(
    open_page, live_api, unscenario_artefacts_dir, picker
):
    with live_api(unscenario_artefacts_dir) as url:
        page = open_page(url, SCENARIOS)
        kind = "Tariff change" if picker == "tariff_markets" else "Custom Trade Agreements"
        page.radio(key="kind").set_value(kind).run()

        offered = page.multiselect(key=picker).options
        assert not page.exception

    assert sorted(offered) == ["Canada", "Hong Kong", "India", "Japan"]  # not the United Kingdom or Mayotte


def test_tariff_sectors_are_offered_by_name(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)
    page.radio(key="kind").set_value("Tariff change").run()

    offered = " ".join(page.multiselect(key="tariff_sectors").options)
    for sector in httpx.get(f"{api_url}/sectors").json():
        assert sector["name"] in offered


def test_the_tariff_mode_runs_and_says_when_the_effect_flattens_beyond_the_training_range(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["JPN"]).run()
    page.number_input(key="tariff_rate").set_value(1000.0).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert "associates" in " ".join(element.value for element in page.markdown)
    # A neutral box, information about the model rather than an error.
    clip = [element.value for element in page.info if "higher than any the model was trained on" in element.value]
    assert clip and "5" in clip[0]  # the clipped pairs, counted by the API
    assert not page.warning
    assert set(page.dataframe[MARKETS].value["Market"]) == {"Japan"}


def test_a_tariff_within_the_training_range_shows_no_clip_message(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["JPN"]).run()
    page.multiselect(key="tariff_sectors").set_value(["22"]).run()
    page.number_input(key="tariff_rate").set_value(20.0).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert "higher than any the model was trained on" not in _page_text(page)
    assert set(page.dataframe[MARKETS].value["Market"]) == {"Japan"}


def test_a_tariff_result_says_how_many_pairs_it_raises_and_lowers(monkeypatch, api_url):
    # India's Baseline tariffs: 150%, 10%, 20%, 60% and one unknown.
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["IND"]).run()
    page.number_input(key="tariff_rate").set_value(30.0).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert "This sets the Applied Tariff of 5 pairs to 30%: a rise for 2, a cut for 2." in [
        element.value for element in page.markdown
    ]


def test_a_scenario_that_is_no_tariff_change_has_no_raised_and_lowered_line(monkeypatch, api_url):
    for page in (_run_cptpp(monkeypatch, api_url), _run_india(monkeypatch, api_url)):
        assert "it raises" not in _page_text(page)


# --- Click-to-select as on the other pages, adaptive money in sentences ----------------------


def test_rows_are_chosen_by_clicking_any_value_as_on_the_leaderboard(monkeypatch, api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    leaderboard = run_page()
    clicked_any_value = list(leaderboard.dataframe[0].proto.selection_mode)

    page = _run_india(monkeypatch, api_url)

    for table in (MARKETS, DRILL_DOWN):
        assert list(page.dataframe[table].proto.selection_mode) == clicked_any_value
    assert not list(page.dataframe[PRODUCTS].proto.selection_mode)  # Top Products gaining opens nothing


def test_choosing_a_pair_keeps_the_market_and_the_other_table_unchanged(monkeypatch, api_url):
    page = _run_india(monkeypatch, api_url)
    pair = httpx.post(f"{api_url}/scenarios", json={**INDIA, "market": "IND", "limit": 50}).json()["top_pairs"][0]

    page.session_state["scenario_drill_pair"] = {"hs6": pair["hs6"], "iso3": pair["iso3"]}
    page.run()

    assert not page.exception
    assert len(_pair_links(page)) == 1  # only the drill-down opens a pair's box


def test_the_blocked_opportunities_line_states_the_fixed_rule_in_the_adaptive_form(monkeypatch, api_url):
    page = _run_india(monkeypatch, api_url)

    line = _blocked_heading_help(page)
    assert "20%" in line and "$1M" in line and "$1,000,000" not in line
    assert "no UK Trade Agreement in force" in line and "of Baseline" in line  # the full rule
    assert "Actual Exports are below the level with the agreement" in line  # a Blocked Opportunity needs room to grow


def test_the_page_says_what_a_trade_agreement_means_before_a_run(monkeypatch, api_url):
    meaning = httpx.get(f"{api_url}/labels").json()["trade_agreement_meaning"]

    page = _scenarios_page(monkeypatch, api_url)

    assert any(meaning in element.value for element in page.caption)


def test_the_subtitle_starts_with_what_the_advisor_does(monkeypatch, api_url):
    # The Advisor builds their own Scenario rather than picking one from a list.
    page = _scenarios_page(monkeypatch, api_url)

    assert page.header[0].value == "Scenario Builder"
    assert page.markdown[0].value.startswith("Build a Scenario (a Trading Bloc preset, ")


def test_a_link_with_add_prefills_a_trade_agreement_with_that_market(open_page, api_url):
    # The Blocked Opportunities page links each Market here: `?add=IND` fills the builder, the visitor runs it.
    page = open_page(api_url, SCENARIOS, query={"add": "IND"})

    assert not page.exception
    assert page.radio(key="kind").value == "Custom Trade Agreements"
    assert page.multiselect(key="add").value == ["IND"]
    assert "scenario_body" not in page.session_state  # prefilled, not run

    page.button(key="run").click().run()
    assert page.dataframe and not page.exception


def test_an_unknown_market_in_the_link_is_left_out(open_page, api_url):
    page = open_page(api_url, SCENARIOS, query={"add": "IND,XXX"})

    assert not page.exception
    assert page.multiselect(key="add").value == ["IND"]


def test_a_prefilled_builder_can_be_changed(open_page, api_url):
    # The link fills the builder once: a change the visitor makes afterwards is kept on the next run.
    page = open_page(api_url, SCENARIOS, query={"add": "IND"})

    page.radio(key="kind").set_value("Trading Bloc preset").run()

    assert page.radio(key="kind").value == "Trading Bloc preset"


def test_the_illustrative_expander_is_worded_true_for_a_scenario_with_no_blocked_opportunities(monkeypatch, api_url):
    # The expander qualifies the Scenario Effect too, often with no Blocked Opportunity value shown.
    page = _run_tariff(monkeypatch, api_url)

    [expander] = [box for box in page.expander if box.label.startswith("Illustrative")]
    assert expander.label == "Illustrative: what these figures are read against"