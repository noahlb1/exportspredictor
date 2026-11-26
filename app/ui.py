"""Shared page pieces: cached API reads, the API's wording and the caveat badges (the frame around each page is
`app.frame`, tables are `app.tables`, the ranking pages' filters `app.filters` and the detail panel `app.panel`).
"""

import json
from typing import Any

import streamlit as st

from app import client

ROWS_SHOWN = 500  # the top of a Leaderboard-style ranking shown on the page; its summary covers every filtered pair


@st.cache_data(ttl=300, show_spinner=False)
def fetch(path: str, api_url: str, **params) -> Any:
    """Cached API read. `api_url` is part of the cache key so a changed API is re-read."""
    del api_url
    return client.get(path, **params)


def labels() -> dict:
    """The API's wording and options (`/labels`): unit and caption, Opportunity Types, caveats, last tariff year,
    caution line and term definitions."""
    return fetch("/labels", client.api_url())


def read_limit() -> int:
    """The most rows a ranking read returns, as the API serves it (`/labels` `max_rows`; it refuses more): the CSV and
    Show all hold these."""
    return labels()["max_rows"]


def term(name: str) -> str:
    """What a key term means, worded once in the package (`wording.term_definitions`) and served by the API (`/labels`
    `terms`): every page's tooltip or `?` help on a key term comes from here. Those naming the Gap Period or the
    Baseline year carry the artefact's."""
    return labels()["terms"][name]


def terms_help(*terms: tuple[str, str]) -> str:
    """One `?` explaining several key terms, each (its name, its `term` key): "**Name**: explanation", a paragraph each."""
    return "\n\n".join(f"**{name}**: {term(key)}" for name, key in terms)


def usd_unit() -> str:
    """The unit in column headings, e.g. "USD, nominal"."""
    return labels()["usd_unit"]


def illustrative() -> str:
    """The word every Scenario Effect and Blocked Opportunity value heading carries, e.g. "illustrative"."""
    return labels()["illustrative"]["badge"].lower()


def scenario_note(note: dict) -> None:
    """The API's Scenario note under the figures it qualifies, closed in one expander, the same on the Blocked
    Opportunities page and the Scenario Builder: the Illustrative label, then each benchmark (or that it was not
    run) and, for a Blend, what a Scenario moves."""
    with st.expander(f"{labels()['illustrative']['badge']}: what these figures are read against"):
        st.info(f"**{note['label']}.** {note['text']}")
        parts = [note[part]["text"] for part in ("benchmark", "model_association", "event_backtest")]
        if note.get("blend"):
            parts.append(note["blend"]["text"])
        st.markdown("\n".join(f"- {text}" for text in parts))


def caveat_tiers() -> list[dict]:
    """The Caveat Tiers (ADR 0005, `/caveats`), in display order: each one's row field (`column`), `badge`, `plural`,
    `text`, `hide_param` and, if it gives a reason per row, `reason_column` and `row_text`. Every badge, caveat, hide
    filter and hidden-rows notice is drawn from these, so a tier the API declares needs no app code."""
    return fetch("/caveats", client.api_url())


def caveat_badges(row: dict) -> str:
    """The caveat badges of one ranking row or explanation, comma-separated ("" if none)."""
    return ", ".join(tier["badge"] for tier in caveat_tiers() if row[tier["column"]])


def show_caveats(row: dict) -> None:
    """One warning per caveat of a pair (its detail view), worded by the API (`caveat_texts`)."""
    for text in row.get("caveat_texts", []):
        st.warning(text)


@st.cache_data(ttl=300, show_spinner=False)
def run_scenario(body_json: str, api_url: str) -> dict:
    """Cached Scenario run; `body_json` is the request body as sorted JSON (the cache key)."""
    del api_url
    return client.post("/scenarios", json.loads(body_json))


def rank_orders() -> dict[str, dict]:
    """The ranking orders the API accepts as `rank_by`, by name: title and description."""
    return {order["name"]: order for order in fetch("/opportunities/rank-by", client.api_url())}


def blocked_rank_orders() -> dict[str, dict]:
    """The orders the Blocked Opportunities read accepts as `rank_by`, by name, the default first: title and
    description."""
    return {order["name"]: order for order in fetch("/opportunities/blocked/rank-by", client.api_url())}
