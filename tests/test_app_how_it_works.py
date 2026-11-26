"""The How the Model Works page, as a visitor sees it through the router: last in the menu, the full outline of
ideas with a one-line summary each, and the layers beneath each written idea."""

from streamlit.testing.v1 import AppTest

PAGE = "pages/how_the_model_works.py"
TITLE = "How the Model Works"


def test_the_page_is_last_in_the_menu_and_renders_under_its_title(open_page, fixture_api_url, menu):
    app: AppTest = open_page(fixture_api_url, PAGE)

    assert not app.exception
    assert menu[-1].title == TITLE
    assert app.header[0].value == TITLE


# The outline spec #181 asks for, grouped as its tickets write it (#196, #197, #198).
OUTLINE = {
    "How the model is built": ["The data", "The features", "Gradient-boosted trees on the GPU"],
    "How the model is checked": [
        "Out-of-fold scoring",
        "The time split",
        "The test year, read once",
        "The ranking measure",
        "The blend with the Naive Baseline",
        "Monotone policy features",
    ],
    "What it produces and how it is served": [
        "The Potential Range",
        "Market Reach",
        "The gravity-model benchmark",
        "Scenarios as re-prediction",
        "From data to app",
    ],
}


def _open(open_page, api_url) -> AppTest:
    app = open_page(api_url, PAGE)
    assert not app.exception
    return app


def _summaries(app: AppTest) -> dict[str, str]:
    """Each idea's one-line summary, as the bold idea name and the sentence after it, outside any expander."""
    inside = {id(element) for expander in app.expander for element in expander.markdown}
    lines = [m.value for m in app.main.markdown if id(m) not in inside and m.value.startswith("**")]
    return {line.split("**")[1]: line.split("**", 2)[2].lstrip(": ") for line in lines}


def test_the_full_outline_shows_each_idea_with_a_one_sentence_summary(open_page, fixture_api_url):
    app = _open(open_page, fixture_api_url)

    assert [heading.value for heading in app.subheader] == list(OUTLINE)
    summaries = _summaries(app)
    assert list(summaries) == [idea for ideas in OUTLINE.values() for idea in ideas]
    for idea, summary in summaries.items():
        assert summary.endswith(".") and summary.count(". ") == 0, f"{idea}: one sentence, not {summary!r}"


LAYERS = ["What it is", "Why this project uses it", "How it is done here"]
LAYER_WORDS = 100  # a short paragraph or a few bullets, not an essay


def test_how_the_model_is_built_is_written_at_every_layer(open_page, fixture_api_url):
    app = _open(open_page, fixture_api_url)
    built = len(OUTLINE["How the model is built"])

    labels = [expander.label for expander in app.expander]
    assert labels[: built * len(LAYERS)] == LAYERS * built
    text = " ".join(m.value for expander in app.expander[: built * len(LAYERS)] for m in expander.markdown)
    for fact in ("HS6", "Zero Cells", "Leave-one-out supply", "target Market", "XGBoost", "graphics card"):
        assert fact in text, fact


def test_what_it_produces_and_how_it_is_served_is_written_at_every_layer(open_page, fixture_api_url):
    app = _open(open_page, fixture_api_url)
    served = len(OUTLINE["What it produces and how it is served"])

    labels = [expander.label for expander in app.expander]
    assert labels[-served * len(LAYERS) :] == LAYERS * served
    text = " ".join(m.value for expander in app.expander[-served * len(LAYERS) :] for m in expander.markdown)
    for fact in ("80%", "other Exporters", "year before", "PPML", "policy features", "FastAPI", "Streamlit", "uv"):
        assert fact in text, fact


def test_how_the_model_is_checked_is_written_at_every_layer(open_page, fixture_api_url):
    app = _open(open_page, fixture_api_url)
    checked = len(OUTLINE["How the model is checked"])
    start = len(OUTLINE["How the model is built"]) * len(LAYERS)
    end = start + checked * len(LAYERS)

    labels = [expander.label for expander in app.expander]
    assert labels[start:end] == LAYERS * checked
    text = " ".join(m.value for expander in app.expander[start:end] for m in expander.markdown)
    for fact in ("five", "all its years", "validation year", "Test-year Ledger", "Spearman", "ties",
                 "Naive Baseline", "monotone", "training range"):
        assert fact in text, fact


def test_every_layer_opens_and_is_short(open_page, fixture_api_url):
    app = _open(open_page, fixture_api_url)

    assert app.expander
    for expander in app.expander:
        words = sum(len(m.value.split()) for m in expander.markdown)
        assert 0 < words <= LAYER_WORDS, f"{expander.label}: {words} words"


def test_the_page_stands_alone_and_stays_associational(open_page, fixture_api_url):
    app = _open(open_page, fixture_api_url)

    text = " ".join(m.value for m in app.main.markdown).lower()
    for word in ("course", "concept notes", "adr", ".md", "causes", "forecast"):
        assert word not in text, word
