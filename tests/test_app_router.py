"""The router (`app/Home.py`) and the frame every page shares (spec #103, "Pages and navigation" and
"Shared page furniture"), run with AppTest through the router against the session's live fixture API."""

from dataclasses import replace

import httpx
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store

MENU = ["Leaderboard", "Blocked Opportunities", "Scenarios", "Model Evaluation"]
# Every page that shows figures, by the path the router declares it with (None: the default page).
PAGES = {
    "Leaderboard": None,
    "Blocked Opportunities": "pages/blocked_opportunities.py",
    "Scenarios": "pages/scenarios.py",
    "Model Evaluation": "pages/model_evaluation.py",
    "Product → Markets": "pages/product_to_markets.py",
    "Market → Products": "pages/market_to_products.py",
    "Why?": "pages/why.py",
}


@pytest.fixture
def menu(monkeypatch) -> list:
    """The pages the router passes to `st.navigation` on its latest run."""
    declared = []
    navigation = streamlit.navigation

    def record(pages, *args, **kwargs):
        declared[:] = pages
        return navigation(pages, *args, **kwargs)

    monkeypatch.setattr(streamlit, "navigation", record)
    return declared


def test_the_menu_shows_the_four_pages_in_order_and_opens_on_the_leaderboard(open_page, fixture_api_url, menu):
    app = open_page(fixture_api_url)

    assert not app.exception
    assert isinstance(menu, list)  # a flat menu: a dict would give it group headings
    assert [page.title for page in menu if page.visibility == "visible"] == MENU
    assert [page.title for page in menu if page.url_path == ""] == ["Leaderboard"]  # the default page
    assert [title.value for title in app.title] == ["UK Exports Predictor"]
    assert [heading.value for heading in app.header][:1] == ["Leaderboard"]


def _labels(api_url: str) -> dict:
    return httpx.get(f"{api_url}/labels").json()


def _health(api_url: str) -> dict:
    return httpx.get(f"{api_url}/health").json()


@pytest.mark.parametrize("heading", list(PAGES))
def test_every_page_has_the_caution_line_under_its_heading_and_the_footer_last(open_page, fixture_api_url, heading):
    app = open_page(fixture_api_url, PAGES[heading])

    assert not app.exception
    assert app.header[0].value == heading
    captions = [caption.value for caption in app.main.caption]
    assert _labels(fixture_api_url)["caution"] in captions
    health = _health(fixture_api_url)
    footer = captions[-1]
    assert f"Baseline year {health['baseline_year']} (provisional data)." in footer
    assert health["gap_period_label"] in footer
    assert _labels(fixture_api_url)["usd_caption"] in footer
    assert f"Model version {health['artefact_version']}." in footer


@pytest.mark.parametrize("heading", list(PAGES))
def test_no_page_shows_the_old_blue_banner(open_page, fixture_api_url, heading):
    app = open_page(fixture_api_url, PAGES[heading])

    assert not [box.value for box in app.info if "provisional data" in box.value]


def test_the_footer_reads_the_model_version_from_the_api(open_page, live_api, fixture_artefacts, tmp_path):
    metadata = {**fixture_artefacts.metadata, "version": "another-version"}
    store.save(replace(fixture_artefacts, metadata=metadata), tmp_path)
    with live_api(tmp_path) as url:
        app: AppTest = open_page(url, "pages/scenarios.py")

    assert "Model version another-version." in app.main.caption[-1].value
