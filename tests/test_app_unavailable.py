"""When the API is down or not ready, every app page says so plainly and shows none of its content.

Each page is opened through the router (`app/Home.py`), which checks the API once for every page.
"""

import pytest
from conftest import free_port
from streamlit.testing.v1 import AppTest

PAGES = [
    None,  # the Leaderboard, the default page
    "pages/blocked_opportunities.py",
    "pages/scenarios.py",
    "pages/model_evaluation.py",
    "pages/product_to_markets.py",
    "pages/market_to_products.py",
    "pages/why.py",
]


def _unreachable_url() -> str:
    return f"http://127.0.0.1:{free_port()}"


def _errors(app: AppTest) -> str:
    assert not app.exception
    # None of the page's content: no heading, caution line, table, control or footer.
    assert not app.header and not app.main.caption and not app.dataframe and not app.main.selectbox
    return " ".join(element.value for element in app.error)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page or "leaderboard")
def test_api_down_shows_how_to_start_it(open_page, page):
    errors = _errors(open_page(_unreachable_url(), page))

    assert "API is unavailable" in errors
    assert "podman compose up" in errors


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page or "leaderboard")
def test_api_without_artefacts_says_they_are_not_loaded(open_page, live_api, tmp_path, page):
    with live_api(tmp_path) as url:
        errors = _errors(open_page(url, page))

    assert "artefacts are not loaded" in errors
