"""Excluded Products in the Streamlit app: listed on Model Evaluation only. A search for one is a Leaderboard test."""

from dataclasses import replace

import polars as pl
import pytest
from conftest import run_page
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store
from exportspredictor.exclusions import EXCLUSIONS

EVALUATION = "pages/model_evaluation.py"
FUELS, BULLION = EXCLUSIONS[0].reason, EXCLUSIONS[1].reason
EXCLUDED = pl.DataFrame(
    {
        "hs6": ["270900", "271011", "710812"],
        "description": ["Crude petroleum oils", "Light petroleum oils", "Gold, unwrought"],
        "sector": ["27", "27", "71"],
        "reason": [FUELS, FUELS, BULLION],
    }
)


def _page(monkeypatch, api_url: str, path: str | None) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = run_page(path)
    assert not page.exception
    return page


@pytest.fixture
def evaluation(monkeypatch, live_api, fixture_artefacts, tmp_path):
    """Model Evaluation over an artefact with three Excluded Products under two reasons."""
    store.save(replace(fixture_artefacts, excluded_products=EXCLUDED), tmp_path)
    with live_api(tmp_path) as url:
        yield _page(monkeypatch, url, EVALUATION)


def _section(page: AppTest):
    [section] = [e for e in page.expander if e.label == "Excluded Products"]
    return section


def test_model_evaluation_counts_the_excluded_products_under_each_reason(evaluation):
    section = _section(evaluation)
    reasons = section.dataframe[0].value

    assert "3 Products" in section.markdown[0].value
    assert list(reasons["Reason"]) == [FUELS, BULLION]
    assert list(reasons["Products"]) == [2, 1]


def test_model_evaluation_lists_every_excluded_product_with_its_sector_name_and_reason(evaluation):
    [full_list] = [e for e in _section(evaluation).expander if e.label.startswith("All ")]
    table = full_list.dataframe[0].value

    assert full_list.label == "All 3 Excluded Products"
    assert list(table.columns) == ["HS6", "Product", "Sector", "Reason"]
    assert list(table["HS6"]) == ["270900", "271011", "710812"]
    assert list(table["Sector"]) == [
        "Mineral fuels and oils", "Mineral fuels and oils", "Pearls, precious stones and metals",
    ]
    assert list(table["Reason"]) == [FUELS, FUELS, BULLION]


@pytest.mark.parametrize("path", [None, "pages/blocked_opportunities.py"], ids=["leaderboard", "blocked"])
def test_no_ranking_page_shows_the_excluded_products_list(monkeypatch, fixture_api_url, path):
    page = _page(monkeypatch, fixture_api_url, path)

    assert not [e for e in page.expander if "Excluded Products" in e.label]
