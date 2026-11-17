"""The Streamlit ranking pages, run with AppTest against a live local API on fixture artefacts."""

import io
import socket
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest
import streamlit
import uvicorn
from streamlit.testing.v1 import AppTest

from api.main import create_app

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGE = APP_DIR / "pages" / "1_Product_to_Markets.py"
MARKET_PAGE = APP_DIR / "pages" / "2_Market_to_Products.py"
# Bank of England XUAAUSS, 2024 annual average: the fixture's Baseline year.
USD_PER_GBP_2024 = 1.2783


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def api_url(fixture_artefacts_dir):
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(fixture_artefacts_dir), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.05)
    yield url
    server.should_exit = True
    thread.join(timeout=5)


def _run_page(monkeypatch, api_url: str, query: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()
    page.text_input(key="product_query").input(query).run()
    return page


def test_page_shows_markets_ranked_by_export_gap(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    assert not page.exception
    table = page.dataframe[0].value
    assert list(table["Market"]) and set(table["Market"]) == {"Japan", "Canada", "India"}
    gaps = list(table["Export Gap (£)"])
    assert gaps == sorted(gaps, reverse=True)


def test_page_shows_baseline_year_banner(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    banner = " ".join(element.value for element in page.info)
    assert "2024" in banner and "provisional" in banner


def test_searching_by_everyday_words_finds_the_product(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "smartphone")

    assert not page.exception
    assert page.selectbox(key="hs6").value == "851712"
    assert any("851712" in element.value for element in page.subheader)


def test_choosing_among_several_matches_shows_that_product(monkeypatch, api_url):
    # Sector-level prefix "8": 851712 and 870323.
    page = _run_page(monkeypatch, api_url, "8")
    options = page.selectbox(key="hs6").options

    assert len(options) == 2
    page.selectbox(key="hs6").select_index(1).run()
    assert any("870323" in element.value for element in page.subheader)


def test_unknown_search_text_shows_a_message(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "zzz-not-a-product")

    assert not page.exception
    assert any("zzz-not-a-product" in element.value for element in page.warning)
    assert not page.dataframe


def test_api_down_shows_a_clear_message(monkeypatch):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", f"http://127.0.0.1:{_free_port()}")
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()

    assert not page.exception
    assert any("API" in element.value for element in page.error)


def _market_page(monkeypatch, api_url: str, iso3: str | None = "JPN") -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(MARKET_PAGE), default_timeout=30)
    page.run()
    if iso3 is not None:
        page.selectbox(key="iso3").set_value(iso3).run()
    return page


def _captions(page: AppTest) -> str:
    return " ".join(element.value for element in page.caption)


def test_market_page_ranks_products_by_export_gap_in_gbp(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")

    assert not page.exception
    assert any("Japan" in element.value for element in page.subheader)
    table = page.dataframe[0].value
    assert set(table["HS6"]) == {"030420", "220830", "300490", "851712", "870323"}
    gaps = list(table["Export Gap (£)"])
    assert gaps == sorted(gaps, reverse=True)


@pytest.mark.parametrize("make_page", [_market_page, lambda m, url: _run_page(m, url, "220830")], ids=["market", "product"])
def test_ranking_pages_state_the_gbp_rate_and_year(monkeypatch, api_url, make_page):
    page = make_page(monkeypatch, api_url)

    captions = _captions(page)
    assert "1.2783" in captions and "2024" in captions and "Bank of England" in captions


def test_values_can_be_shown_in_usd(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")
    in_gbp = page.dataframe[0].value

    page.radio(key="currency").set_value("USD").run()

    in_usd = page.dataframe[0].value
    assert list(in_usd["Export Gap (USD)"]) == pytest.approx([gap * USD_PER_GBP_2024 for gap in in_gbp["Export Gap (£)"]])


def test_market_page_shows_the_uk_trade_agreement(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "IND")

    assert set(page.dataframe[0].value["UK Trade Agreement in force"]) == {"No"}
    assert any("no UK Trade Agreement" in element.value for element in page.markdown)


def test_product_page_shows_which_markets_have_a_uk_trade_agreement(monkeypatch, api_url):
    table = _run_page(monkeypatch, api_url, "220830").dataframe[0].value

    assert dict(zip(table["Market"], table["UK Trade Agreement in force"])) == {
        "Japan": "Yes",
        "Canada": "Yes",
        "India": "No",
    }


def test_sector_filter_narrows_the_market_ranking(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")

    page.selectbox(key="sector").set_value("85").run()

    assert list(page.dataframe[0].value["HS6"]) == ["851712"]


def test_region_filter_narrows_the_product_ranking(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    page.selectbox(key="region").set_value("South Asia").run()

    assert list(page.dataframe[0].value["Market"]) == ["India"]


def test_region_filter_narrows_the_market_choice(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, iso3=None)

    page.selectbox(key="region").set_value("South Asia").run()

    assert page.selectbox(key="iso3").options == ["India"]


def test_sector_filter_narrows_the_product_search(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "8")  # 851712 and 870323

    page.selectbox(key="sector").set_value("87").run()

    assert len(page.selectbox(key="hs6").options) == 1
    assert any("870323" in element.value for element in page.subheader)


def test_min_gap_filter_drops_smaller_gaps(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")
    everything = page.dataframe[0].value

    page.number_input(key="min_gap_gbp").set_value(0).run()

    table = page.dataframe[0].value
    assert 0 < len(table) < len(everything)
    assert (table["Export Gap (£)"] >= 0).all()


def test_filters_that_match_nothing_say_so(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "JPN")

    page.number_input(key="min_gap_gbp").set_value(10**12).run()

    assert not page.exception
    assert not page.dataframe
    assert any("No Opportunities" in element.value for element in page.info)


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


@pytest.mark.parametrize("make_page", [_market_page, lambda m, url: _run_page(m, url, "220830")], ids=["market", "product"])
def test_csv_export_matches_the_ranking_on_screen(monkeypatch, api_url, downloads, make_page):
    page = make_page(monkeypatch, api_url)
    page.radio(key="currency").set_value("USD").run()

    on_screen = page.dataframe[0].value.drop(columns="Why?")  # links are navigation, not data
    exported = pd.read_csv(io.StringIO(downloads[-1]), dtype={"HS6": str, "Sector": str})
    pd.testing.assert_frame_equal(exported, on_screen.reset_index(drop=True), check_dtype=False)
    assert len(page.get("download_button")) == 1


def test_market_page_api_down_shows_a_clear_message(monkeypatch):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", f"http://127.0.0.1:{_free_port()}")
    page = AppTest.from_file(str(MARKET_PAGE), default_timeout=30)
    page.run()

    assert not page.exception
    assert any("API" in element.value for element in page.error)


def test_market_page_can_rank_by_blocked_opportunity_value(monkeypatch, api_url):
    page = _market_page(monkeypatch, api_url, "IND")

    page.radio(key="rank_by").set_value("blocked").run()

    assert not page.exception
    values = list(page.dataframe[0].value["Blocked Opportunity value (£)"])
    assert values == sorted(values, reverse=True) and values[0] > 0
    assert "Blocked Opportunity value" in _captions(page)
