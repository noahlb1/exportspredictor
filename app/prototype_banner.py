"""PROTOTYPE, throwaway (wayfinder ticket #98): never merge.

What a first-time visitor sees above the table: four treatments of the banner plus today's, three
kinds of first-visit guidance, and three ways to explain the terms, on the Leaderboard's own route
over the split view settled in #95 and the logo settled in #97. Switched with `?variant=` or the bar
pinned to the bottom of the page.

    A  One quiet line          everything in one grey line under the heading; model version at the foot of the sidebar
    B  Caution at the figures  nothing at the top; the caution sits in the table's unit line, the rest in the sidebar
    C  Badges                  a row of small badges, each with a tooltip
    D  Caution up, rest down   the caution alone under the heading; the rest in a footer under the table
    today                      the blue box and currency caption

Run it (PowerShell, from this worktree; the API must be up on :8000):

    $env:EXPORTSPREDICTOR_PROTOTYPE = "1"; ./scripts/wt.ps1 streamlit run app/Home.py
"""

import streamlit as st

from app import client, prototype_logo
from app import prototype_leaderboard as pl
from app.ui import OPPORTUNITY_TYPE_HELP, labels

NAME = "UK Exports Predictor"
BANNERS = {
    "A": "One quiet line",
    "B": "Caution at the figures",
    "C": "Badges",
    "D": "Caution up, rest down",
    "today": "Today's blue box",
}
NAMES = ["Title above heading", "Small line above heading", "Heading only"]
GUIDES = ["Intro card", "Three steps", "Welcome dialog", "None"]
TERMS = ["Tooltips only", "Terms button", "Legend line"]
CAUTION = "Figures are associations the model has learned, not causes or forecasts."
CAUTION_SHORT = "Associations the model has learned, not causes"


def term_help(gap: str) -> dict[str, str]:
    return {
        "Export Potential": (
            "The value of UK exports of this Product to this Market that the model expects, given the "
            "fundamentals of the UK, the Market and the Product."
        ),
        "Actual Exports": f"What the UK actually exported of this Product to this Market ({gap}).",
        "Export Gap": (
            f"Export Potential minus Actual Exports ({gap}). Positive means the UK exports less than "
            "its fundamentals suggest: an Opportunity."
        ),
        "Opportunity Type": OPPORTUNITY_TYPE_HELP,
    }


def _init() -> None:
    asked = st.query_params
    defaults = {
        "banner": (asked.get("variant"), list(BANNERS), "A"),
        "app_name": (asked.get("name"), NAMES, NAMES[0]),
        "guide": (asked.get("guide"), GUIDES, GUIDES[0]),
        "terms": (asked.get("terms"), TERMS, TERMS[0]),
    }
    for key, (value, allowed, default) in defaults.items():
        if st.session_state.get(key) not in allowed:
            st.session_state[key] = value if value in allowed else default
    st.session_state.setdefault("guide_open", True)


def _close_guide() -> None:
    st.session_state["guide_open"] = False


def _open_guide() -> None:
    st.session_state["guide_open"] = True


WHAT = (
    "This app estimates the UK's **Export Potential** for each Product and Market: the value of exports a model "
    "trained on global goods trade expects, given the fundamentals of the UK, the Market and the Product. "
    "Where **Actual Exports** are lower, the difference is the **Export Gap**, and the pair is an **Opportunity**."
)
STEPS = [
    ("1 · Read the ranking", "Each row is one Product in one Market. The largest Export Gap is at the top."),
    (
        "2 · Open a row",
        "Click any value. The panel on the right shows which factors raised or lowered its Export Potential.",
    ),
    ("3 · Narrow it down", "Use the filters on the left to keep one Sector, Market or Product."),
]


@st.dialog("Welcome to the UK Exports Predictor", width="medium", on_dismiss=_close_guide)
def welcome() -> None:
    st.markdown(WHAT)
    for title, text in STEPS:
        st.markdown(f"**{title}.** {text}")
    st.caption(CAUTION)
    if st.button("Start exploring", type="primary"):
        _close_guide()
        st.rerun()


def guidance(kind: str) -> None:
    if kind == "None" or not st.session_state["guide_open"]:
        return
    if kind == "Welcome dialog":
        welcome()
    elif kind == "Intro card":
        with st.container(border=True):
            text, button = st.columns([9, 1], vertical_alignment="center")
            text.markdown(
                f"**New here?** {WHAT}  \n**Where to start:** the largest Opportunity is already open on the right. "
                "Click any row to see why the model rates it, and use the filters on the left to narrow to a "
                "Sector, Market or Product."
            )
            button.button("Got it", on_click=_close_guide, key="guide_close")
    else:  # Three steps
        columns = st.columns([4, 4, 4, 1], vertical_alignment="center")
        for column, (title, text) in zip(columns, STEPS, strict=False):
            with column.container(border=True):
                st.markdown(f"**{title}**  \n{text}")
        columns[3].button("Got it", on_click=_close_guide, key="guide_close")


def about(health: dict, *, everything: bool) -> list[str]:
    """The facts that leave the top of the page. `everything` adds what variants B and D move down too."""
    lines = []
    if everything:
        lines += [
            f"Baseline year {health['baseline_year']} (provisional data).",
            f"Export Gaps are {health['gap_period_label']}s.",
            labels()["usd_caption"],
        ]
    return [*lines, f"Model version {health['artefact_version']}."]


def page() -> None:
    _init()
    state = st.session_state
    tile = prototype_logo.mark_svg("Tile", dark=False)
    st.set_page_config(page_title=NAME, page_icon=prototype_logo.data_uri(tile), layout="wide")
    st.logo(tile, size="large")
    try:
        health = client.get("/health")
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        bar()
        return
    banner, gap = state["banner"], health["gap_period_label"]
    terms = term_help(gap)
    pl.TERM_HELP = terms
    pl.CAPTION_EXTRA = f" {CAUTION}" if banner == "B" else ""

    # The app's name and the page heading.
    if state["app_name"] == NAMES[0]:
        st.title(NAME)
    elif state["app_name"] == NAMES[1]:
        st.caption(NAME.upper())
    if banner == "today":
        st.info(
            f"Baseline year {health['baseline_year']} (provisional data). Export Gaps are {gap}s. "
            f"Figures are associations the model has learned, not causes. Model version {health['artefact_version']}."
        )
        st.caption(labels()["usd_caption"])
    heading, again = st.columns([8, 1], vertical_alignment="bottom")
    heading.header("Leaderboard")
    if state["guide"] != "None" and not state["guide_open"]:
        again.button("What is this?", type="tertiary", on_click=_open_guide, key="guide_again")
    st.markdown(
        "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's "
        "fundamentals suggest."
    )
    unit = labels()["usd_unit"]
    if banner == "A":
        st.caption(
            f"{health['baseline_year']} data (provisional) · Export Gaps are {gap}s · {unit} · {CAUTION_SHORT}."
        )
    elif banner == "C":
        with st.container(horizontal=True):
            st.badge(
                f"{health['baseline_year']} data, provisional",
                color="gray",
                help=f"The Baseline year is {health['baseline_year']}, the latest year of trade data. "
                "Its figures are provisional.",
            )
            st.badge(f"{gap}s", color="gray", help=f"Export Potential, Actual Exports and Export Gaps are {gap}s.")
            st.badge(unit, color="gray", help=labels()["usd_caption"])
            st.badge("Associations, not causes", color="orange", icon=":material/info:", help=CAUTION)
    elif banner == "D":
        st.caption(CAUTION)
    if state["terms"] == "Legend line":
        legend = [f"**{term}**: {text.split('. ')[0].rstrip('.')}" for term, text in list(terms.items())[:3]]
        st.caption(" · ".join(legend))
    guidance(state["guide"])

    # The split view settled in #95.
    st.sidebar.subheader("Filters")
    params, hs6, active = pl.filters_ui(st.sidebar)
    st.sidebar.button("Reset filters", on_click=pl.reset_filters, disabled=not active)
    if banner in ("A", "C"):
        st.sidebar.caption(" ".join(about(health, everything=False)))
    elif banner == "B":
        with st.sidebar.expander("About these figures"):
            for line in about(health, everything=True):
                st.caption(line)
    body = pl.read(dict(params), hs6)
    pl.active_line(active, body, reset_here=False)
    if not body["opportunities"]:
        st.info("No Opportunities match these filters.")
    else:
        view_column, terms_column = st.columns([6, 1], vertical_alignment="center")
        view = view_column.segmented_control(
            "View",
            ["Table and detail", "Wide table"],
            key="c_view",
            required=True,
            default="Table and detail",
            label_visibility="collapsed",
        )
        if state["terms"] == "Terms button":
            with terms_column.popover("Terms", icon=":material/help:", type="tertiary"):
                for term, text in terms.items():
                    st.markdown(f"**{term}.** {text}")
        if view == "Wide table":
            row = pl.ranking_table(body, columns=pl.WIDE_COLUMNS, rule="0.0", height=390, key="table_c", first=True)
            with st.container(border=True):
                pl.detail(row)
        else:
            table_side, detail_side = st.columns([5, 4], gap="large")
            with table_side:
                row = pl.ranking_table(
                    body, columns=["Product", "Market", "Export Gap"], rule="0.0", height=700, key="table_c", first=True
                )
            # PROTOTYPE (#100): a new key per pair makes the panel a new element, so it starts from the top
            with detail_side, st.container(border=True, height=780, key=f"lb_panel_{row['hs6']}_{row['iso3']}"):
                pl.detail(row)
    if banner == "D":
        st.divider()
        st.caption(" ".join(about(health, everything=True)))
    bar()


def bar() -> None:
    """The prototype bar, pinned to the bottom of the page. Not part of any design."""
    keys = list(BANNERS)

    def step(by: int) -> None:
        st.session_state["banner"] = keys[(keys.index(st.session_state["banner"]) + by) % len(keys)]

    with st.bottom, st.container(border=True):
        back, pick, forward, name, guide, terms = st.columns([1, 6, 1, 4, 4, 4], vertical_alignment="bottom")
        back.button("←", on_click=step, args=(-1,), key="banner_back")
        pick.selectbox(
            "PROTOTYPE: the banner",
            keys,
            key="banner",
            format_func=lambda k: BANNERS[k] if k == "today" else f"{k} · {BANNERS[k]}",
        )
        forward.button("→", on_click=step, args=(1,), key="banner_forward")
        name.selectbox("App name", NAMES, key="app_name")
        guide.selectbox("First-visit guidance", GUIDES, key="guide", on_change=_open_guide)
        terms.selectbox("Terms explained by", TERMS, key="terms")
    state = {
        "variant": st.session_state["banner"],
        "name": st.session_state["app_name"],
        "guide": st.session_state["guide"],
        "terms": st.session_state["terms"],
    }
    st.query_params.update(state)
    with st.expander("PROTOTYPE state"):
        st.json(state | {"guide_open": st.session_state["guide_open"]})
