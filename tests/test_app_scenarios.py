"""The Streamlit Scenarios page, run with AppTest against a live API on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with India.

AppTest cannot click a table row, so a test chooses a Market or a pair by setting the page's remembered selection
(`scenario_market`, `scenario_drill_pair`, `scenario_blocked_pair`), as a click does.
"""

import json

import httpx
import pandas as pd
import pytest
import streamlit
from conftest import run_page
from streamlit.testing.v1 import AppTest

from app import client

SCENARIOS = "pages/scenarios.py"
# The page's tables, in order once a Scenario has run.
MARKETS, DRILL_DOWN, PRODUCTS, BLOCKED = range(4)
INDIA = {"add_agreements": ["IND"], "min_effect_share": 0.0, "min_effect_usd": 0.0, "limit": 100}


@pytest.fixture
def api_url(fixture_api_url):
    return fixture_api_url


def _scenarios_page(monkeypatch, api_url) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(SCENARIOS)


def _run_cptpp(monkeypatch, api_url) -> AppTest:
    page = _scenarios_page(monkeypatch, api_url)
    page.selectbox(key="preset").set_value("cptpp").run()
    page.button(key="run").click().run()
    assert not page.exception
    return page


def _run_india(monkeypatch, api_url, *, thresholds_off: bool = False) -> AppTest:
    page = _scenarios_page(monkeypatch, api_url)
    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["IND"]).run()
    if thresholds_off:
        page.number_input(key="min_share").set_value(0.0).run()
        page.number_input(key="min_usd").set_value(0.0).run()
    page.button(key="run").click().run()
    assert not page.exception
    return page


def _page_text(page: AppTest) -> str:
    return " ".join(
        str(element.value) for element in [*page.markdown, *page.caption, *page.info, *page.warning, *page.subheader]
    )


def _stale(page: AppTest) -> list[str]:
    return [element.value for element in [*page.info, *page.warning] if "Run Scenario" in element.value]


def test_the_summed_scenario_effect_carries_the_totals_note(monkeypatch, api_url):
    # Spec #113 "Totals understate": every summed figure is read against the artefact's share of Actual Exports.
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

    page.selectbox(key="preset").set_value("eu_rejoin").run()
    assert _stale(page)
    assert page.dataframe  # the earlier result stays on screen under the line

    page.selectbox(key="preset").set_value("cptpp").run()
    assert not _stale(page)  # back to the controls of the result shown

    page.number_input(key="min_share").set_value(0.0).run()
    assert _stale(page)
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
    page = _run_india(monkeypatch, api_url, thresholds_off=True)

    pairs = page.dataframe[DRILL_DOWN].value
    assert len(pairs) == 5
    effects = list(pairs["Scenario Effect"])
    assert effects == sorted(effects, reverse=True)


@pytest.mark.parametrize("remembered", ["scenario_drill_pair", "scenario_blocked_pair"])
def test_a_selected_pair_shows_its_box_with_a_link_to_the_leaderboard(monkeypatch, api_url, remembered):
    pair = httpx.post(f"{api_url}/scenarios", json=INDIA).json()["blocked_opportunities"][1]

    page = _run_india(monkeypatch, api_url, thresholds_off=True)
    assert not page.page_link  # no pair chosen yet
    page.session_state[remembered] = {"hs6": pair["hs6"], "iso3": pair["iso3"]}
    page.run()

    assert not page.exception
    [link] = page.page_link
    assert "Leaderboard" in link.proto.label
    assert sorted(link.proto.query_string.split("&")) == sorted([f"product={pair['hs6']}", f"market={pair['iso3']}"])
    labels = [metric.label for metric in page.metric]
    assert pair["description"] in _page_text(page)
    assert any("Scenario Effect" in label for label in labels)
    assert {"Tariff, Baseline", "Tariff, Scenario"} <= set(labels)


def test_a_remembered_pair_not_in_the_rows_shows_no_box(monkeypatch, api_url):
    page = _run_india(monkeypatch, api_url, thresholds_off=True)
    page.session_state["scenario_blocked_pair"] = {"hs6": "000000", "iso3": "IND"}
    page.run()

    assert not page.exception
    assert not page.page_link


def test_money_columns_are_numeric_millions_with_the_unit_stated_once(monkeypatch, api_url):
    result = httpx.post(f"{api_url}/scenarios", json=INDIA).json()

    page = _run_india(monkeypatch, api_url, thresholds_off=True)

    blocked = page.dataframe[BLOCKED].value
    assert pd.api.types.is_float_dtype(blocked["Scenario Effect"])
    assert list(blocked["Scenario Effect"]) == pytest.approx(
        [row["effect_usd"] / 1e6 for row in result["blocked_opportunities"]]
    )
    assert list(blocked["Actual Exports"]) == pytest.approx(
        [row["baseline_actual_exports_usd"] / 1e6 for row in result["blocked_opportunities"]]
    )
    for frame in page.dataframe:
        for column in ("Baseline", "Scenario"):
            assert pd.api.types.is_float_dtype(frame.value[column])
        assert not [heading for heading in frame.value.columns if "USD" in heading or "$" in heading]
    assert "$ millions" in _page_text(page)


def test_short_headings_carry_tooltips(monkeypatch, api_url):
    word = httpx.get(f"{api_url}/labels").json()["illustrative"]["badge"].lower()

    page = _run_india(monkeypatch, api_url, thresholds_off=True)

    blocked = page.dataframe[BLOCKED]
    assert list(blocked.value.columns)[:2] == ["Product", "Market"]
    for heading in (
        "Scenario Effect", "Effect, % of Baseline", "Baseline", "Scenario", "Actual Exports",
        "Tariff, Baseline %", "Tariff, Scenario %",
    ):
        assert heading in blocked.value.columns
    config = json.loads(blocked.proto.columns)
    effect = next(value for key, value in config.items() if "Scenario Effect" in key)
    assert word in effect["help"].lower()
    assert "Pairs changed" in page.dataframe[MARKETS].value.columns


def test_blocked_opportunities_are_the_apis_each_with_room_to_grow(monkeypatch, api_url):
    # Spec #133 (rewritten from "leave no pair out for its Actual Exports"): a pair needs room to grow, Baseline-year
    # Actual Exports below its Scenario Export Potential; pairs that export at all still qualify.
    result = httpx.post(f"{api_url}/scenarios", json=INDIA).json()

    page = _run_india(monkeypatch, api_url, thresholds_off=True)

    assert len(page.dataframe[BLOCKED].value) == result["blocked_count"]
    assert all(
        row["baseline_actual_exports_usd"] < row["scenario_potential_usd"] for row in result["blocked_opportunities"]
    )
    assert any(row["baseline_actual_exports_usd"] > 0 for row in result["blocked_opportunities"])


def test_a_scenario_that_changes_nothing_says_so(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["JPN"]).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert page.warning  # the API's summary, shown as a warning instead of a table
    assert not page.dataframe


def test_the_note_is_not_shown_before_a_run(monkeypatch, api_url, fixture_artefacts):
    page = _scenarios_page(monkeypatch, api_url)

    note_text = httpx.get(f"{api_url}/labels").json()["illustrative"]["text"]

    assert not page.exception
    text = _page_text(page)
    assert fixture_artefacts.evaluation["event_backtest"]["text"] not in text
    assert note_text not in text


def test_a_run_shows_the_apis_scenario_note_beside_the_scenario_effect(monkeypatch, api_url):
    note = httpx.post(f"{api_url}/scenarios", json={"add_agreements": ["IND"], "limit": 100}).json()["note"]

    text = _page_text(_run_india(monkeypatch, api_url))

    assert note["label"] in text and note["text"] in text
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["text"] in text
    assert note["blend"]["text"] in text


def test_the_scenario_effect_figure_says_illustrative(monkeypatch, api_url):
    word = httpx.get(f"{api_url}/labels").json()["illustrative"]["badge"].lower()

    page = _run_india(monkeypatch, api_url)

    effects = [metric.label for metric in page.metric if "Scenario Effect" in metric.label]
    assert effects and all(word in label for label in effects)


def test_threshold_inputs_default_to_the_apis_thresholds(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)
    defaults = client.get("/scenarios/thresholds")

    assert page.number_input(key="min_share").value == defaults["min_effect_share"] * 100
    assert page.number_input(key="min_usd").value == defaults["min_effect_usd"] / 1e6


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
    clip = [element.value for element in page.warning if "training range" in element.value]
    assert clip and "5" in clip[0]  # the clipped pairs, counted by the API
    assert set(page.dataframe[MARKETS].value["Market"]) == {"Japan"}


def test_a_tariff_within_the_training_range_shows_no_clip_message(monkeypatch, api_url):
    page = _scenarios_page(monkeypatch, api_url)

    page.radio(key="kind").set_value("Tariff change").run()
    page.multiselect(key="tariff_markets").set_value(["JPN"]).run()
    page.multiselect(key="tariff_sectors").set_value(["22"]).run()
    page.number_input(key="tariff_rate").set_value(20.0).run()
    page.button(key="run").click().run()

    assert not page.exception
    assert not [element.value for element in page.warning if "training range" in element.value]
    assert set(page.dataframe[MARKETS].value["Market"]) == {"Japan"}


# --- Final review of #103: click-to-select as on the other pages, adaptive money in sentences ----------------------


def test_rows_are_chosen_by_clicking_any_value_as_on_the_leaderboard(monkeypatch, api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    leaderboard = run_page()
    clicked_any_value = list(leaderboard.dataframe[0].proto.selection_mode)

    page = _run_india(monkeypatch, api_url, thresholds_off=True)

    for table in (MARKETS, DRILL_DOWN, BLOCKED):
        assert list(page.dataframe[table].proto.selection_mode) == clicked_any_value
    assert not list(page.dataframe[PRODUCTS].proto.selection_mode)  # Top Products gaining opens nothing


def test_choosing_a_pair_keeps_the_market_and_the_other_table_unchanged(monkeypatch, api_url):
    page = _run_india(monkeypatch, api_url, thresholds_off=True)
    pair = httpx.post(f"{api_url}/scenarios", json=INDIA).json()["blocked_opportunities"][0]

    page.session_state["scenario_blocked_pair"] = {"hs6": pair["hs6"], "iso3": pair["iso3"]}
    page.run()

    assert not page.exception
    assert len(page.page_link) == 1  # only the Blocked Opportunities table opens a pair's box


def test_the_blocked_opportunities_line_states_the_threshold_in_the_adaptive_form(monkeypatch, api_url):
    page = _run_india(monkeypatch, api_url)  # the default thresholds: 20% and $1M

    line = next(element.value for element in page.caption if "room to grow" in element.value)
    assert "$1.0m" in line and "$1,000,000" not in line
    assert "whatever their Actual Exports" not in line  # spec #133: Actual Exports must be below Scenario Export Potential
    assert "Actual Exports" in line and "Scenario Export Potential" in line


def test_the_page_says_what_a_trade_agreement_means_before_a_run(monkeypatch, api_url):
    meaning = httpx.get(f"{api_url}/labels").json()["trade_agreement_meaning"]

    page = _scenarios_page(monkeypatch, api_url)

    assert any(meaning in element.value for element in page.caption)


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
