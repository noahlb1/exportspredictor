"""The frame every page shares (spec #103, "Shared page furniture"): what the router sets once, and what each page
puts around its content.

The router (`app/Home.py`) calls `setup()` (page configuration, logo, loading mark, the app's title) and runs the
chosen page only if `api_ready()`; otherwise the page shows the API-down message and none of its content. A page
wraps its content in `with page(heading, intro):`, which shows the heading, the one-sentence introduction and the
grey caution line above the content, and a divider and the grey footer (Baseline year, Gap Period, currency, model
version) below it. Every wording, year and version comes from the API.
"""

import base64
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from app import client
from app.ui import fetch, labels

APP_NAME = "UK Exports Predictor"
ASSETS = Path(__file__).parent / "assets"
LOGO = ASSETS / "logo.svg"  # the Profile container ship tile (issue #97)
LOADING_MARK = ASSETS / "loading.svg"  # the logo's ship riding the wave line
API_DOWN_HELP = (
    "Start everything with `podman compose up`, or for development run `uv run uvicorn api.main:app`. "
    "Then reload this page."
)


def setup() -> None:
    """What every page shares, set once by the router: wide layout, the logo in the sidebar and the browser tab,
    the loading mark and the app's title as text."""
    st.set_page_config(page_title=APP_NAME, page_icon=str(LOGO), layout="wide")
    st.logo(str(LOGO), size="large")
    _loading_mark()
    st.title(APP_NAME)


def _loading_mark() -> None:
    """The loading mark in the middle of the page while the app is running, after a 0.4 s delay so quick reruns do not
    flash it. Plain CSS on Streamlit's running state, not a custom component."""
    image = "data:image/svg+xml;base64," + base64.b64encode(LOADING_MARK.read_bytes()).decode()
    # `data-test-script-state` on the app element is undocumented (Streamlit 1.67): check it on any Streamlit upgrade.
    st.html(
        f"""<style>
        .stApp[data-test-script-state="running"]::after {{
            content: ""; position: fixed; left: 50%; top: 45%; width: 120px; height: 120px; margin: -60px 0 0 -60px;
            background: url("{image}") center / contain no-repeat; border-radius: 26px;
            box-shadow: 0 6px 28px rgba(0, 0, 0, 0.35); z-index: 999999; pointer-events: none;
            opacity: 0; animation: loading-mark-in 0.2s ease 0.4s forwards;
        }}
        @keyframes loading-mark-in {{ to {{ opacity: 1; }} }}
        </style>"""
    )


def api_ready() -> bool:
    """Whether the API is up with its artefacts loaded. If not, show what is wrong and how to start it."""
    try:
        client.get("/health")  # not cached: the check sees the API as it is now
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error} {API_DOWN_HELP}")
        return False
    return True


@contextmanager
def page(heading: str, intro: str, beside: Callable[[], None] | None = None) -> Iterator[None]:
    """A page's frame: its heading, introduction and caution line, then the content, then a divider and the footer.

    `beside` draws something small beside the heading (the Leaderboard's "What is this?" link)."""
    if beside is None:
        st.header(heading)
    else:
        left, right = st.columns([6, 1], vertical_alignment="bottom")
        left.header(heading)
        with right:
            beside()
    st.markdown(intro)
    st.caption(labels()["caution"])
    yield
    st.divider()
    footer()


def footer() -> None:
    """The grey line under a page's content: Baseline year, Gap Period, the API's currency caption and model version."""
    health = fetch("/health", client.api_url())
    st.caption(
        f"Baseline year {health['baseline_year']} (provisional data). "
        f"Export Gaps are {health['gap_period_label']}s. "
        f"{labels()['usd_caption'].rstrip('.')}. "
        f"Model version {health['artefact_version']}."
    )
