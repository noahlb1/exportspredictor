"""An artefact built without an opt-in part (spec #65, story 23) loads, is served, and renders
"not run for this version" for that part: in the API's evaluation, on the Model Evaluation page
and in the model card. An artefact with every part shows no such notice.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor import governance

PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "6_Model_Evaluation.py"
PARTS = ["Event Backtests", "Monotone-constraint comparison", "Gain ladder", "PPML benchmark"]
NOTICE = "not run for this version"


def _evaluation(root) -> dict:
    with TestClient(create_app(root)) as client:
        response = client.get("/evaluation")
    assert response.status_code == 200
    return response.json()


def test_a_default_artefact_loads(fixture_default_artefacts_dir, fixture_default_artefacts):
    assert store.load(store.latest_folder(fixture_default_artefacts_dir)).version == fixture_default_artefacts.version


def test_the_api_lists_each_part_a_default_run_skipped(fixture_default_artefacts_dir):
    not_run = _evaluation(fixture_default_artefacts_dir)["not_run"]
    assert [part["title"] for part in not_run] == PARTS
    assert all(NOTICE in part["text"] for part in not_run)


def test_the_api_lists_nothing_for_a_full_run(fixture_artefacts_dir):
    assert _evaluation(fixture_artefacts_dir)["not_run"] == []


def _page_text(monkeypatch, live_api, root) -> str:
    with live_api(root) as url:
        monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", url)
        page = AppTest.from_file(str(PAGE), default_timeout=30)
        page.run()
    assert not page.exception
    return " ".join(e.value for e in [*page.markdown, *page.caption, *page.subheader, *page.info])


def test_the_model_evaluation_page_says_which_parts_were_not_run(monkeypatch, live_api, fixture_default_artefacts_dir):
    text = _page_text(monkeypatch, live_api, fixture_default_artefacts_dir)
    for part in PARTS:
        assert f"{part}: {NOTICE}" in text


def test_the_model_evaluation_page_shows_no_notice_for_a_full_run(monkeypatch, live_api, fixture_artefacts_dir):
    assert NOTICE not in _page_text(monkeypatch, live_api, fixture_artefacts_dir)


def test_the_model_card_says_which_parts_were_not_run(fixture_default_artefacts):
    text = governance.render_model_card(governance.model_card(fixture_default_artefacts))
    for part in PARTS:
        assert f"{part}: {NOTICE}" in text
    assert "Primary metric" in text  # the v1 comparison's primary metric is still reported


@pytest.mark.parametrize("part", PARTS)
def test_the_model_card_of_a_full_run_has_no_notice(fixture_artefacts, part):
    assert f"{part}: {NOTICE}" not in governance.render_model_card(governance.model_card(fixture_artefacts))
