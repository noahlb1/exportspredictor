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
    """What a term means, worded by the API (`/labels` `terms`): `export_potential`, `actual_exports`,
    `export_gap` or `opportunity_type`. Those that name the Gap Period carry the artefact's label."""
    return labels()["terms"][name]


def usd_unit() -> str:
    """The unit in column headings, e.g. "USD, nominal"."""
    return labels()["usd_unit"]


def illustrative() -> str:
    """The word every Scenario Effect and Blocked Opportunity value heading carries, e.g. "illustrative"."""
    return labels()["illustrative"]["badge"].lower()


def scenario_note(note: dict) -> None:
    """The API's Scenario note beside a Scenario Effect: the Illustrative label, then each benchmark
    (or that it was not run for this version) and, for a Blend, what a Scenario moves."""
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


def caveats() -> dict[str, tuple[str, str]]:
    """Caveat badges (ADR 0005): the row field that sets each, its badge and what it means."""
    return {tier["column"]: (tier["badge"], tier["text"]) for tier in caveat_tiers()}


def caveat_badges(row: dict) -> str:
    """The caveat badges of one ranking row or explanation, comma-separated ("" if none)."""
    return ", ".join(tier["badge"] for tier in caveat_tiers() if row[tier["column"]])


def caveat_texts(row: dict) -> list[str]:
    """The caveat text of each tier one ranking row is in: the tier's text, or its row text naming the row's reason."""
    texts = []
    for tier in caveat_tiers():
        if row[tier["column"]]:
            reason = row[tier["reason_column"]] if tier["reason_column"] else None
            texts.append(tier["text"] if reason is None or not tier["row_text"] else tier["row_text"].format(reason=reason))
    return texts


def show_caveats(row: dict) -> None:
    """One warning per caveat of a pair (its detail view)."""
    for text in caveat_texts(row):
        st.warning(text)


@st.cache_data(ttl=300, show_spinner=False)
def run_scenario(body_json: str, api_url: str) -> dict:
    """Cached Scenario run; `body_json` is the request body as sorted JSON (the cache key)."""
    del api_url
    return client.post("/scenarios", json.loads(body_json))


def rank_orders() -> dict[str, dict]:
    """The ranking orders the API accepts as `rank_by`, by name: title and description."""
    return {order["name"]: order for order in fetch("/opportunities/rank-by", client.api_url())}
