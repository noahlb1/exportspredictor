"""Product surfaces show one model: no history term reaches them (ADR 0013).

The banned terms and the release-tag allowance live in `exportspredictor.product_surface`.
A ticket that cleans a surface enrols it in `SURFACES`: a name and a function returning the
surface's text as a reader or client sees it, built on the fixture.
"""

from collections.abc import Callable

from exportspredictor.product_surface import banned_terms

# Enrolment point: (surface name, collector returning its text).
SURFACES: list[tuple[str, Callable[[], str]]] = []


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


def test_a_dotted_name_that_is_not_a_release_tag_is_found():
    assert banned_terms("v1.2 and v2.0") == ["v1.2", "v2.0"]


def test_history_phrases_are_found_across_case_plurals_and_line_breaks():
    text = "Beside Earlier versions, a frozen\nversion, its first form and the Arm against the model alone."
    assert banned_terms(text) == ["Earlier versions", "frozen\nversion", "first form", "Arm against"]
