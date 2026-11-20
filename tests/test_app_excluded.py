"""Excluded Products in the Streamlit app: the list with reasons. What a search for one shows is a Leaderboard test."""

import pytest
from conftest import run_page
from streamlit.testing.v1 import AppTest

CRUDE_OIL = "270900"  # the fixture world's Excluded Product


def _page(monkeypatch, api_url: str, path: str | None) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    return run_page(path)


@pytest.mark.parametrize("path", [None, "pages/blocked_opportunities.py"], ids=["leaderboard", "blocked"])
def test_the_rankings_show_the_excluded_products_list_with_reasons(monkeypatch, fixture_api_url, path):
    page = _page(monkeypatch, fixture_api_url, path)

    assert not page.exception
    [expander] = [e for e in page.expander if "Excluded Products" in e.label]
    table = expander.table[0].value
    assert list(table.index) == [CRUDE_OIL]
    assert "pipelines" in table["Reason"].iloc[0]
