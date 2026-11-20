"""Every Streamlit page is in nominal US dollars only: one USD caption, no £ and no currency switch (#41).

Each page is opened through the router with AppTest against the session's live fixture API, with its
main content on screen (a ranking, an explanation, a Scenario result), so the check covers the values too.
"""

import pytest
from conftest import run_page
from streamlit.testing.v1 import AppTest

USD_CAPTION = "Values are nominal US dollars, as reported by BACI"


def _product_to_markets(page: AppTest) -> None:
    page.text_input(key="product_query").input("220830").run()


def _market_to_products(page: AppTest) -> None:
    page.selectbox(key="iso3").set_value("JPN").run()


def _scenarios(page: AppTest) -> None:
    page.selectbox(key="preset").set_value("cptpp").run()
    page.button(key="run").click().run()


def _shown(page: AppTest) -> None:
    del page  # the page's content is on screen when it opens


# Each page, the URL parameters it opens with, and how to bring its main content on screen.
SHOW = {
    "leaderboard": (None, {}, _shown),
    "product_to_markets": ("pages/product_to_markets.py", {}, _product_to_markets),
    "market_to_products": ("pages/market_to_products.py", {}, _market_to_products),
    "blocked_opportunities": ("pages/blocked_opportunities.py", {}, _shown),
    "scenarios": ("pages/scenarios.py", {}, _scenarios),
    "why": ("pages/why.py", {"hs6": "220830", "iso3": "JPN"}, _shown),
    "model_evaluation": ("pages/model_evaluation.py", {}, _shown),
}


@pytest.fixture(params=list(SHOW))
def page(request, monkeypatch, fixture_api_url) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", fixture_api_url)
    path, query, show = SHOW[request.param]
    page = run_page(path, query=query)
    show(page)
    assert not page.exception
    return page


def _everything_shown(page: AppTest) -> str:
    """The text of every element on the page, table headers and cells included."""
    texts = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "info", "success", "warning", "error"):
        texts += [str(element.value) for element in getattr(page, kind)]
    for metric in page.metric:
        texts += [str(metric.label), str(metric.value)]
    for widget in [*page.radio, *page.selectbox, *page.number_input, *page.text_input]:
        texts.append(str(widget.label))
    for table in page.dataframe:
        texts += [str(column) for column in table.value.columns]
        texts += [str(cell) for cell in table.value.astype(str).to_numpy().ravel()]
    return " ".join(texts)


def test_every_page_states_the_usd_caption_once(page):
    # In the footer, with the Baseline year, Gap Period and model version.
    assert [USD_CAPTION in c.value for c in page.caption].count(True) == 1


def test_no_page_shows_pounds(page):
    assert "£" not in _everything_shown(page)
    assert "GBP" not in _everything_shown(page)


def test_no_page_offers_a_currency_switch(page):
    assert not [radio for radio in page.radio if radio.key == "currency"]
