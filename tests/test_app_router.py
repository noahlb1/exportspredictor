"""The router (`app/Home.py`) and the frame every page shares (pages, navigation and shared page furniture), run
with AppTest through the router against the session's live fixture API."""


import httpx
import pytest

MENU = ["Leaderboard", "Blocked Opportunities", "Scenario Builder", "Model Evaluation", "How the Model Works"]
# Every page that shows figures, by the path the router declares it with (None: the default page).
PAGES = {
    "Leaderboard": None,
    "Blocked Opportunities": "pages/blocked_opportunities.py",
    "Scenario Builder": "pages/scenarios.py",
    "Model Evaluation": "pages/model_evaluation.py",
}


def test_the_menu_shows_the_five_pages_in_order_and_opens_on_the_leaderboard(open_page, fixture_api_url, menu):
    app = open_page(fixture_api_url)

    assert not app.exception
    assert isinstance(menu, list)  # a flat menu: a dict would give it group headings
    assert [page.title for page in menu if page.visibility == "visible"] == MENU
    assert [page.title for page in menu] == MENU  # the removed pages (Why? among them) are not even hidden ones
    assert [page.title for page in menu if page.url_path == ""] == ["Leaderboard"]  # the default page
    assert {page.title: page.url_path for page in menu}["Scenario Builder"] == "Scenario_Builder"
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
    assert footer.endswith(_labels(fixture_api_url)["usd_caption"].rstrip(".") + ".")  # a sentence each
    assert health["artefact_version"] not in footer  # the Artefact ID is on Model Evaluation, not on every page


@pytest.mark.parametrize("heading", list(PAGES))
def test_no_page_shows_the_old_blue_banner(open_page, fixture_api_url, heading):
    app = open_page(fixture_api_url, PAGES[heading])

    assert not [box.value for box in app.info if "provisional data" in box.value]
