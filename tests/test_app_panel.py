"""The Leaderboard's split view and its detail panel (spec #103, "Leaderboard" and "Detail panel"), run with AppTest
through the router against the session's live fixture API.

AppTest cannot click a table row, so a test chooses a pair as the page remembers it (`PAIR` in session state) or
through the URL's Product and Market.
"""

import httpx
import pandas as pd
import pytest
from conftest import run_page
from streamlit.testing.v1 import AppTest

PAIR = "leaderboard_pair"  # the remembered pair, (HS6, ISO3)
WHISKY = "220830"
JEWELLERY = "711319"  # the fixture's Flagged Product; Hong Kong is its Re-export Hub
MEDICINES = "300490"  # India's Applied Tariff on it is carried forward from 2021


def _home(monkeypatch, api_url: str, **query: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(query=query)


def _choose(page: AppTest, hs6: str, iso3: str) -> AppTest:
    page.session_state[PAIR] = (hs6, iso3)
    return page.run()


def _panel_heading(page: AppTest) -> str:
    [heading] = [element.value for element in page.main.subheader if "→" in element.value]
    return heading


def _texts(page: AppTest) -> str:
    kinds = ("markdown", "caption", "info", "warning", "subheader")
    return " ".join(str(element.value) for kind in kinds for element in getattr(page.main, kind))


def _top_row(api_url: str) -> dict:
    return httpx.get(f"{api_url}/opportunities/top").json()["opportunities"][0]


def test_the_top_rows_pair_is_explained_on_open(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    top = _top_row(fixture_api_url)

    assert not page.exception
    assert _panel_heading(page) == f"{top['description']} → {top['market']}"
    assert [metric.label for metric in page.main.metric] == ["Export Potential", "Actual Exports", "Export Gap"]
    assert list(page.dataframe[0].value.columns) == ["Product", "Market", "Export Gap"]


def test_the_three_figures_carry_the_apis_term_wording(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    terms = httpx.get(f"{fixture_api_url}/labels").json()["terms"]

    helps = {metric.label: metric.proto.help for metric in page.main.metric}
    assert helps == {
        "Export Potential": terms["export_potential"],
        "Actual Exports": terms["actual_exports"],
        "Export Gap": terms["export_gap"],
    }


def test_the_remembered_pair_is_explained(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "IND")

    assert not page.exception
    assert _panel_heading(page).endswith("→ India")
    assert WHISKY in _texts(page)


def test_a_link_naming_one_product_and_one_market_opens_on_that_pair(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url, product=WHISKY, market="CAN")

    assert _panel_heading(page).endswith("→ Canada")


def test_a_remembered_pair_no_longer_shown_gives_way_to_the_top_row(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "IND")

    page.sidebar.selectbox(key="market").set_value("JPN").run()

    top = httpx.get(f"{fixture_api_url}/opportunities/top", params={"market": "JPN"}).json()["opportunities"][0]
    assert _panel_heading(page) == f"{top['description']} → Japan"


def test_the_pair_stays_selected_in_the_wide_table_with_its_detail_under_it(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "IND")

    page.segmented_control(key="leaderboard_view").set_value("Wide table").run()

    assert not page.exception
    assert _panel_heading(page).endswith("→ India")
    order = list(page.dataframe[0].proto.column_order)
    assert order[:3] == ["Product", "Market", "Export Gap"] and "Applied Tariff %" in order


def test_the_panel_shows_a_pairs_caveats_and_none_for_a_plain_pair(monkeypatch, fixture_api_url):
    flagged = _choose(_home(monkeypatch, fixture_api_url), JEWELLERY, "HKG")
    plain = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "JPN")

    warnings = " ".join(warning.value for warning in flagged.main.warning)
    assert "Flagged Product" in warnings and "Re-export Hub" in warnings
    assert not any("Flagged Product" in w.value or "Re-export Hub" in w.value for w in plain.main.warning)


def test_the_panel_says_when_the_applied_tariff_is_carried_forward(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), MEDICINES, "IND")

    assert any("carried forward from 2021" in warning.value for warning in page.main.warning)


def test_the_facts_start_with_the_applied_tariff_and_explain_the_potential_range(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "IND")
    body = httpx.get(f"{fixture_api_url}/explain/{WHISKY}/IND").json()

    facts = [element for element in page.main.markdown if element.value.startswith("**") and ":**" in element.value]
    names = [element.value.split(":**")[0].strip("*") for element in facts]
    assert names[:6] == [
        "Applied Tariff, 2024",
        "UK Trade Agreement in force",
        "Blocked Opportunity value",
        "Opportunity Type",
        "Relative Export Gap",
        "Potential Range",
    ]
    assert "150.0%" in facts[0].value and "No" in facts[1].value
    assert "Expand" in facts[3].value and "already exports the Product to this Market" in facts[3].value
    assert facts[5].proto.help == body["potential_range_text"]


def test_the_barrier_shows_only_for_a_pair_with_no_uk_trade_agreement(monkeypatch, fixture_api_url):
    blocked = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "IND")
    agreed = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "JPN")

    barrier = " ".join(box.value for box in [*blocked.main.warning, *blocked.main.info] if "The Barrier" in box.value)
    assert "No UK Trade Agreement is in force with India" in barrier and "150.0%" in barrier
    assert "associates" in barrier and "Illustrative" in barrier
    assert "The Barrier" not in _texts(agreed)


def test_the_barrier_is_a_warning_on_a_blocked_opportunity(monkeypatch, fixture_api_url):
    rows = httpx.get(f"{fixture_api_url}/opportunities/top", params={"trade_agreement": False}).json()["opportunities"]
    for row in rows:
        page = _choose(_home(monkeypatch, fixture_api_url), row["hs6"], row["iso3"])
        warned = any("The Barrier" in warning.value for warning in page.main.warning)
        assert warned == row["blocked_opportunity"], row


def test_the_panel_shows_the_factors_and_both_expanders(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "JPN")
    body = httpx.get(f"{fixture_api_url}/explain/{WHISKY}/JPN").json()

    assert len(page.get("vega_lite_chart")) == 2  # the factors, and Actual Exports against Export Potential by year
    tables = [table.value for table in page.main.dataframe]
    factor_tables = [table for table in tables if "Factor" in table.columns]
    top, every = factor_tables
    assert list(top.columns) == ["Factor", "Value for this pair", "Association with Export Potential"]
    assert len(top) <= 8 and len(every) == len(body["factors"])
    assert any("Japan" in label for label in every["Factor"]) and not any("_" in label for label in every["Factor"])
    assert {expander.label for expander in page.main.expander} >= {"All factors", "Figures by year"}
    text = _texts(page)
    assert "×1.50 means" in text and f"fold model {body['fold']}" in text
    assert "associat" in text and "not causes" in text


def test_the_figures_by_year_show_each_years_values_and_applied_tariff_mark(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), MEDICINES, "IND")
    body = httpx.get(f"{fixture_api_url}/explain/{MEDICINES}/IND").json()

    [history] = [table.value for table in page.main.dataframe if "Year" in table.value.columns]
    assert list(history.columns) == ["Year", "Export Potential", "Actual Exports", "Export Gap", "Applied Tariff"]
    assert max(history["Year"]) == 2024 and list(history["Year"]) == sorted(history["Year"])
    gap_years = history[history["Year"] >= 2022]
    assert gap_years["Export Gap"].mean() * 1e6 == pytest.approx(body["export_gap_usd"])
    marks = history.set_index("Year")["Applied Tariff"]
    assert marks[2021] == "From tariff data"
    assert set(marks[[2022, 2023, 2024]]) == {"Carried forward from 2021"}
    assert pd.api.types.is_float_dtype(history["Export Potential"])


# --- The three steps --------------------------------------------------------------------------------


def _steps(page: AppTest) -> list[str]:
    return [element.value for element in page.main.markdown if element.value.startswith("**") and "·" in element.value]


def test_the_three_steps_close_and_come_back(monkeypatch, fixture_api_url):
    page = _home(monkeypatch, fixture_api_url)
    steps = _steps(page)
    assert [step.split("**")[1] for step in steps] == [
        "1 · Read the ranking.",
        "2 · Open a row.",
        "3 · Narrow it down.",
    ]
    assert "What is this?" not in [button.label for button in page.main.button]

    page.button(key="steps_close").click().run()

    assert not _steps(page)
    page.button(key="steps_open").click().run()
    assert len(_steps(page)) == 3


def test_the_panel_shows_the_markets_reach_in_one_sentence(monkeypatch, fixture_api_url):
    page = _choose(_home(monkeypatch, fixture_api_url), WHISKY, "JPN")
    reach = httpx.get(f"{fixture_api_url}/explain/{WHISKY}/JPN").json()["market_reach"]

    assert reach["text"] in _texts(page)
