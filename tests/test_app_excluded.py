"""Excluded Products in the Streamlit app: the list with reasons, and what a search for one shows."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PRODUCT_PAGE = APP_DIR / "pages" / "1_Product_to_Markets.py"
MARKET_PAGE = APP_DIR / "pages" / "2_Market_to_Products.py"
CRUDE_OIL = "270900"  # the fixture world's Excluded Product


def _page(monkeypatch, api_url: str, path: Path) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(path), default_timeout=30)
    page.run()
    return page


@pytest.mark.parametrize("path", [PRODUCT_PAGE, MARKET_PAGE])
def test_the_rankings_show_the_excluded_products_list_with_reasons(monkeypatch, fixture_api_url, path):
    page = _page(monkeypatch, fixture_api_url, path)

    assert not page.exception
    [expander] = [e for e in page.expander if "Excluded Products" in e.label]
    table = expander.table[0].value
    assert list(table.index) == [CRUDE_OIL]
    assert "pipelines" in table["Reason"].iloc[0]


@pytest.mark.parametrize("query", [CRUDE_OIL, "crude"])
def test_searching_an_excluded_product_shows_its_reason(monkeypatch, fixture_api_url, query):
    page = _page(monkeypatch, fixture_api_url, PRODUCT_PAGE)
    page.text_input(key="product_query").input(query).run()

    assert not page.exception
    warning = " ".join(element.value for element in page.warning)
    assert "Excluded Product" in warning and "pipelines" in warning
    assert not page.dataframe  # no ranking is shown
