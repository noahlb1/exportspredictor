"""Every Streamlit page is in nominal US dollars only: one USD caption, no £ and no currency switch (#41).

Each page is run with AppTest against the session's live fixture API, with its main content
on screen (a ranking, an explanation, a Scenario result), so the check covers the values too.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = Path(__file__).resolve().parents[1] / "app"
USD_CAPTION = "Values are nominal US dollars, as reported by BACI"


def _home(page: AppTest) -> None:
    page.run()


def _product_to_markets(page: AppTest) -> None:
    page.run()
    page.text_input(key="product_query").input("220830").run()


def _market_to_products(page: AppTest) -> None:
    page.run()
    page.selectbox(key="iso3").set_value("JPN").run()


def _scenarios(page: AppTest) -> None:
    page.run()
    page.selectbox(key="preset").set_value("cptpp").run()
    page.button(key="run").click().run()


def _why(page: AppTest) -> None:
    page.query_params["hs6"] = "220830"
    page.query_params["iso3"] = "JPN"
    page.run()


SHOW = {
    "Home.py": _home,
    "pages/1_Product_to_Markets.py": _product_to_markets,
    "pages/2_Market_to_Products.py": _market_to_products,
    "pages/3_Blocked_Opportunities.py": _home,
    "pages/4_Scenarios.py": _scenarios,
    "pages/5_Why.py": _why,
    "pages/6_Model_Evaluation.py": _home,
}


@pytest.fixture(params=list(SHOW))
def page(request, monkeypatch, fixture_api_url) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", fixture_api_url)
    page = AppTest.from_file(str(PAGES / request.param), default_timeout=30)
    SHOW[request.param](page)
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
    assert [c.value for c in page.caption].count(USD_CAPTION) == 1


def test_no_page_shows_pounds(page):
    assert "£" not in _everything_shown(page)
    assert "GBP" not in _everything_shown(page)


def test_no_page_offers_a_currency_switch(page):
    assert not [radio for radio in page.radio if radio.key == "currency"]
