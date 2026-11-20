"""The Streamlit Why? page and its links from rankings, run with AppTest against a live local API."""

import re
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import uvicorn
from conftest import run_page
from streamlit.testing.v1 import AppTest

from api.main import create_app

WHY = "pages/why.py"


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _why(monkeypatch, api_url: str, **query) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(WHY, query=query)


def test_why_page_explains_the_pair_named_in_the_link(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="220830", iso3="JPN")

    assert not page.exception
    assert any("220830" in s.value and "Japan" in s.value for s in page.subheader)
    factors = page.dataframe[0].value
    assert list(factors.columns) == ["Factor", "Value for this pair", "Association with Export Potential"]
    assert any("Japan" in label for label in factors["Factor"])
    assert not any("_" in label for label in factors["Factor"])


def test_why_page_shows_actual_exports_history_for_every_recent_year(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="220830", iso3="JPN")

    history = page.dataframe[1].value
    assert list(history["Year"]) == sorted(history["Year"])
    assert max(history["Year"]) == 2024
    assert "Actual Exports (USD, nominal)" in history.columns


def test_why_page_shows_each_gap_period_years_export_potential_and_gap(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="220830", iso3="JPN")
    body = httpx.get(f"{api_url}/explain/220830/JPN").json()

    history = page.dataframe[1].value
    assert list(history.columns) == [
        "Year",
        "Export Potential (USD, nominal)",
        "Actual Exports (USD, nominal)",
        "Export Gap (USD, nominal)",
        "Applied Tariff",
    ]
    gap_years = history[history["Year"] >= 2022]
    assert list(gap_years["Year"]) == [2022, 2023, 2024]
    assert gap_years["Export Gap (USD, nominal)"].mean() == pytest.approx(body["export_gap_usd"])
    assert history[history["Year"] < 2022]["Export Gap (USD, nominal)"].isna().all()
    text = " ".join(m.value for m in page.markdown) + " ".join(c.value for c in page.caption)
    assert "2022–2024 average" in text


def test_why_page_phrases_results_as_associations(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="220830", iso3="JPN")

    text = " ".join(m.value for m in page.markdown) + " ".join(c.value for c in page.caption)
    assert "associat" in text and "not causes" in text


def test_why_page_without_a_pair_lets_the_advisor_pick_one(monkeypatch, api_url):
    page = _why(monkeypatch, api_url)
    page.text_input(key="why_product_query").input("220830").run()
    page.selectbox(key="why_market").select("JPN").run()

    assert not page.exception
    assert any("Japan" in s.value for s in page.subheader)


def test_why_page_unknown_pair_shows_a_message(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="999999", iso3="JPN")

    assert not page.exception
    assert any("999999" in w.value for w in page.warning)


@pytest.mark.parametrize("query", [{"product": "220830"}, {"market": "JPN"}], ids=["one-product", "one-market"])
def test_leaderboard_rows_link_to_why(monkeypatch, api_url, query):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = run_page(query=query)

    table = page.dataframe[0].value
    links = {}
    for hs6, market, link in zip(table["HS6"], table["Market"], table["Why?"], strict=True):
        url = urlparse(link)
        assert url.path.endswith("Why")
        links[hs6, market] = parse_qs(url.query)
    assert links["220830", "Japan"] == {"hs6": ["220830"], "iso3": ["JPN"]}


def test_why_page_writes_values_in_dollars(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="220830", iso3="JPN")

    assert any(re.search(r"Export Potential \*\*\$[0-9,.]+(bn|m|k)?\*\*", m.value) for m in page.markdown)
    assert "Actual Exports (USD, nominal)" in page.dataframe[1].value.columns


def test_why_page_shows_the_caveats_of_a_flagged_product_in_a_re_export_hub(monkeypatch, api_url):
    flagged = _why(monkeypatch, api_url, hs6="711319", iso3="HKG")
    plain = _why(monkeypatch, api_url, hs6="220830", iso3="JPN")

    assert not flagged.exception
    caveats = " ".join(w.value for w in flagged.warning)
    assert "Flagged Product" in caveats and "Re-export Hub" in caveats
    assert not any("Flagged Product" in w.value or "Re-export Hub" in w.value for w in plain.warning)


def test_why_page_shows_the_pairs_opportunity_type(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="220830", iso3="JPN")

    assert any("Opportunity Type: **Expand**" in element.value for element in page.markdown)


def _pairs_by_uncertainty(api_url: str) -> tuple[dict, dict]:
    rows = httpx.get(f"{api_url}/opportunities/top").json()["opportunities"]
    return next(r for r in rows if r["uncertain_opportunity"]), next(r for r in rows if not r["uncertain_opportunity"])


def test_why_page_shows_the_potential_range_bar_in_plain_words(monkeypatch, api_url):
    uncertain, certain = _pairs_by_uncertainty(api_url)
    page = _why(monkeypatch, api_url, hs6=uncertain["hs6"], iso3=uncertain["iso3"])

    assert not page.exception
    assert len(page.get("vega_lite_chart")) == 1
    words = " ".join(m.value for m in page.markdown)
    assert "Potential Range" in words and "comparable pairs" in words
    assert "confidence" not in words.lower()
    # Where Actual Exports lie is said in the range's words, not as a caveat warning (spec #66)
    assert not any("Uncertain Opportunity" in w.value for w in page.warning)

    other = _why(monkeypatch, api_url, hs6=certain["hs6"], iso3=certain["iso3"])
    assert len(other.get("vega_lite_chart")) == 1
    assert "Potential Range" in " ".join(m.value for m in other.markdown)
    assert not any("Uncertain Opportunity" in w.value for w in other.warning)


def test_why_page_says_when_the_applied_tariff_is_carried_forward_from_2021(monkeypatch, api_url):
    page = _why(monkeypatch, api_url, hs6="300490", iso3="IND")

    assert not page.exception
    assert any("carried forward from 2021" in w.value for w in page.warning)
    factors = page.dataframe[0].value
    assert "The tariff India charges on this Product from the UK" in list(factors["Factor"])
    history = page.dataframe[1].value.set_index("Year")["Applied Tariff"]
    assert history[2021] == "From tariff data"
    assert set(history[[2022, 2023, 2024]]) == {"Carried forward from 2021"}
