"""Product surfaces show one model: no history term reaches them (ADR 0013).

The banned terms and the release-tag allowance live in `exportspredictor.product_surface`.
A ticket that cleans a surface enrols it in `SURFACES`: a name and a function returning the
surface's text as a reader or client sees it, built on the fixture.
"""

from collections.abc import Callable

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from exportspredictor import governance
from exportspredictor.product_surface import banned_terms

_requests: list[pytest.FixtureRequest] = []


@pytest.fixture(autouse=True)
def _collectors_see_fixtures(request):
    """Lets a collector take a pytest fixture (`fixture("fixture_api_url")`) and stay a plain zero-argument call."""
    _requests.append(request)
    yield
    _requests.pop()


def fixture(name: str):
    return _requests[-1].getfixturevalue(name)


def _model_card() -> str:
    return governance.render_model_card(governance.model_card(fixture("fixture_artefacts")))


def _api(path: str) -> Callable[[], str]:
    """A response of the API serving the fixture artefact, as a client receives it (keys and values)."""
    return lambda: httpx.get(f"{fixture('fixture_api_url')}{path}").text


def page_text(app: AppTest) -> str:
    """Every text a visitor sees on a page run: headings, prose, notices, metrics, expander labels and tables."""
    kinds = ("title", "header", "subheader", "markdown", "caption", "success", "warning", "info", "error")
    texts = [element.value for kind in kinds for element in getattr(app, kind)]
    texts += [f"{metric.label} {metric.value} {metric.proto.delta}" for metric in app.metric]
    texts += [expander.label for expander in app.expander]
    texts += [table.value.to_csv() for table in app.dataframe]
    return "\n".join(map(str, texts))


def _page(path: str | None) -> Callable[[], str]:
    def collect() -> str:
        app = fixture("open_page")(fixture("fixture_api_url"), path)
        assert not app.exception, app.exception
        return page_text(app)

    return collect


APP_PAGES = {
    "Leaderboard": None,
    "Blocked Opportunities": "pages/blocked_opportunities.py",
    "Scenario Builder": "pages/scenarios.py",
    "Model Evaluation": "pages/model_evaluation.py",
    "How the Model Works": "pages/how_the_model_works.py",
}

# Enrolment point: (surface name, collector returning its text).
SURFACES: list[tuple[str, Callable[[], str]]] = [
    ("model card", _model_card),
    ("API /evaluation", _api("/evaluation")),
    ("API /health", _api("/health")),
    ("API /how-it-works", _api("/how-it-works")),
    *((f"app page {name}", _page(path)) for name, path in APP_PAGES.items()),
]


def history_terms_on(surfaces: list[tuple[str, Callable[[], str]]]) -> dict[str, list[str]]:
    found = {name: banned_terms(collect()) for name, collect in surfaces}
    return {name: terms for name, terms in found.items() if terms}


def test_enrolled_product_surfaces_name_no_history():
    found = history_terms_on(SURFACES)
    assert not found, f"History terms on product surfaces; name earlier configurations by role (ADR 0013): {found}"


def test_the_guard_fails_on_a_planted_history_term_and_passes_the_release_tag():
    planted = [("planted", lambda: "The Blend beats v2 on 2023."), ("tag", lambda: "Fetch the `v1.0` release.")]
    assert history_terms_on(planted) == {"planted": ["v2"]}


def test_version_names_are_found_as_words_and_in_identifiers():
    assert banned_terms("v1 and v2's ranking; beats_v2 and v10.") == ["v1", "v2", "v2", "v10"]


def test_the_release_tag_and_ordinary_words_pass():
    assert banned_terms("release v1.0; the artefact_version, version 3, uv run, nv2x, prev2") == []


def test_data_source_release_names_pass():
    text = "BACI HS12 V202601, CEPII Gravity V202211, Open Government Licence v3.0"
    assert banned_terms(text) == []
    assert banned_terms("Licence v3.0 aside, v3.0 is a version") == ["v3.0"]


def test_a_dotted_name_that_is_not_a_release_tag_is_found():
    assert banned_terms("v1.2 and v2.0") == ["v1.2", "v2.0"]


def test_history_phrases_are_found_across_case_plurals_and_line_breaks():
    text = "Beside Earlier versions, a frozen\nversion, its first form and the Arm against the model alone."
    assert banned_terms(text) == ["Earlier versions", "frozen\nversion", "first form", "Arm against"]
