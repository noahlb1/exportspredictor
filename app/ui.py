"""Shared page pieces: cached API reads, the API's wording and the caveat badges (the frame around each page is
`app.frame`, tables are `app.tables`, the ranking pages' filters `app.filters` and the detail panel `app.panel`).
"""

import json
from typing import Any

import streamlit as st

from app import client

ROWS_SHOWN = 500  # the top of a Leaderboard-style ranking shown on the page; its summary covers every filtered pair
READ_LIMIT = 6000  # the most rows a ranking read returns (the API refuses more): the CSV and Show all hold these


@st.cache_data(ttl=300, show_spinner=False)
def fetch(path: str, api_url: str, **params) -> Any:
    """Cached API read. `api_url` is part of the cache key so a changed API is re-read."""
    del api_url
    return client.get(path, **params)


def labels() -> dict:
    """The API's wording and options (`/labels`): unit and caption, Opportunity Types, caveats, last tariff year,
    caution line and term definitions."""
    return fetch("/labels", client.api_url())


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


def caveats() -> dict[str, tuple[str, str]]:
    """Caveat badges (ADR 0005): the row field that sets each, its badge and what it means."""
    return {field: (caveat["badge"], caveat["text"]) for field, caveat in labels()["caveats"].items()}


def caveat_badges(row: dict) -> str:
    """The caveat badges of one ranking row or explanation, comma-separated ("" if none)."""
    return ", ".join(badge for field, (badge, _) in caveats().items() if row[field])


def show_caveats(row: dict) -> None:
    """One warning per caveat of a pair (its detail view)."""
    for field, (_, caveat) in caveats().items():
        if row[field]:
            st.warning(caveat)


@st.cache_data(ttl=300, show_spinner=False)
def run_scenario(body_json: str, api_url: str) -> dict:
    """Cached Scenario run; `body_json` is the request body as sorted JSON (the cache key)."""
    del api_url
    return client.post("/scenarios", json.loads(body_json))


def rank_orders() -> dict[str, dict]:
    """The ranking orders the API accepts as `rank_by`, by name: title and description."""
    return {order["name"]: order for order in fetch("/opportunities/rank-by", client.api_url())}
