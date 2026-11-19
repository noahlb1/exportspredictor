"""PROTOTYPE, throwaway (wayfinder ticket #97): never merge.

Four original marks for the app, on the Leaderboard's own route over the split view settled in #95,
switched with `?variant=` or the bar pinned to the bottom of the page. The bar also switches the
questions that cut across marks: tile or bare, where the mark sits, and the accent colour.

    A  Gap bars     two bars level to the same height, the taller one's extra in a lighter tint: the Export Gap
    B  Route        a line leaving a dot and arriving at an arrowhead: an export route
    C  Matrix       a 3 x 3 grid with one cell lit: one Product-Market pair picked out of the grid
    D  Globe        a globe with one point marked: a Market

Run it (PowerShell, from this worktree; the API must be up on :8000):

    $env:EXPORTSPREDICTOR_PROTOTYPE = "1"; ./scripts/wt.ps1 streamlit run app/Home.py

The accent is `primaryColor` under `[theme.light]` and `[theme.dark]` in `.streamlit/config.toml`;
the bar's Accent control overwrites both at run time, for this server only.
"""

import base64

import streamlit as st
from streamlit import config

NAME = "UK Exports Predictor"
MARKS = {"A": "Gap bars", "B": "Route", "C": "Matrix", "D": "Globe"}
STYLES = ["Tile", "Bare"]
PLACEMENTS = {
    "sidebar-mark": "Mark in the sidebar, title as text",
    "sidebar-wordmark": "Mark and name in the sidebar, no page title",
    "title": "Mark beside the title",
}
ACCENTS = {"Blue": "#2a78d6", "Teal": "#0f766e", "Indigo": "#4f46e5", "Streamlit red": "#ff4b4b"}
LIGHT, DARK = "#ffffff", "#171614"  # Streamlit 1.67 page backgrounds, light and dark


def glyph(mark: str, ink: str) -> str:
    """The mark's shapes on a 64 x 64 canvas, in one ink, with a fainter tint of it for the quiet parts."""
    soft = f'fill="{ink}" fill-opacity="0.45"'
    soft_line = f'stroke="{ink}" stroke-opacity="0.45"'
    if mark == "A":
        return (
            f'<rect x="14" y="36" width="14" height="16" rx="2" fill="{ink}"/>'
            f'<rect x="36" y="36" width="14" height="16" fill="{ink}"/>'
            f'<path d="M36 36V14a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v22z" {soft}/>'
        )
    if mark == "B":
        return (
            f'<path d="M16 48C18 30 28 20 44 18" fill="none" stroke="{ink}" stroke-width="6" stroke-linecap="round"/>'
            f'<path d="M38 10l12 7-10 9" fill="none" stroke="{ink}" stroke-width="6" stroke-linecap="round" '
            'stroke-linejoin="round"/>'
            f'<circle cx="16" cy="49" r="6" fill="{ink}"/>'
        )
    if mark == "C":
        cells = ""
        for row, y in enumerate((9, 26, 43)):
            for col, x in enumerate((9, 26, 43)):
                lit = (row, col) == (0, 2)
                cells += f'<rect x="{x}" y="{y}" width="12" height="12" rx="2.5" ' + (
                    f'fill="{ink}"/>' if lit else f"{soft}/>"
                )
        return cells
    return (
        f'<circle cx="30" cy="34" r="19" fill="none" {soft_line} stroke-width="5"/>'
        f'<ellipse cx="30" cy="34" rx="8" ry="19" fill="none" {soft_line} stroke-width="4"/>'
        f'<path d="M11 34h38" {soft_line} stroke-width="4"/>'
        f'<circle cx="46" cy="18" r="8" fill="{ink}"/>'
    )


def _inner(mark: str, style: str, accent: str) -> str:
    """A white glyph on an accent tile, or a bare accent glyph."""
    if style == "Tile":
        return f'<rect width="64" height="64" rx="14" fill="{accent}"/>{glyph(mark, "#ffffff")}'
    return glyph(mark, accent)


def mark_svg(mark: str, style: str, accent: str) -> str:
    """One mark as a square SVG."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
        f"{_inner(mark, style, accent)}</svg>"
    )


def wordmark_svg(mark: str, style: str, accent: str) -> str:
    """The mark with the app's name. One image serves both themes, so the name's colour follows the
    browser's light or dark setting (not Streamlit's own theme menu)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="430" height="64" viewBox="0 0 430 64">'
        "<style>text{fill:#1f2430}@media(prefers-color-scheme:dark){text{fill:#f4f5f7}}</style>"
        f"{_inner(mark, style, accent)}"
        '<text x="80" y="44" font-family="Source Sans Pro,Segoe UI,Arial,sans-serif" font-size="36" '
        f'font-weight="600">{NAME}</text></svg>'
    )


def data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def contact_sheet(accent: str) -> str:
    """Every mark in both styles at 16, 24, 32 and 64 px, on Streamlit's light and dark backgrounds,
    with a mock browser tab for the 16 px icon."""
    rows = ""
    for mark, name in MARKS.items():
        for style in STYLES:
            uri = data_uri(mark_svg(mark, style, accent))
            cells = ""
            for background, text in ((LIGHT, "#31333f"), (DARK, "#fafafa")):
                images = "".join(
                    f'<img src="{uri}" width="{px}" height="{px}" style="margin-right:14px;vertical-align:middle">'
                    for px in (16, 24, 32, 64)
                )
                tab = (
                    '<span style="display:inline-block;border:1px solid #8884;border-radius:8px 8px 0 0;'
                    f'padding:5px 12px;font:13px Segoe UI,Arial,sans-serif;color:{text};vertical-align:middle">'
                    f'<img src="{uri}" width="16" height="16" style="vertical-align:-3px;margin-right:7px">'
                    f"{NAME}</span>"
                )
                cells += (
                    f'<td style="background:{background};padding:14px 18px;border:1px solid #8883">{images}{tab}</td>'
                )
            rows += (
                '<tr><td style="padding:6px 14px 6px 0;font:14px Segoe UI,Arial,sans-serif;white-space:nowrap">'
                f"{mark} · {name}, {style.lower()}</td>{cells}</tr>"
            )
    return f'<table style="border-collapse:collapse">{rows}</table>'


def _init() -> None:
    """Defaults for the bar's controls, taking `?variant=`, `?style=`, `?placement=` and `?accent=` from the URL once."""
    asked = st.query_params
    defaults = {
        "logo_mark": (asked.get("variant"), list(MARKS), "A"),
        "logo_style": (asked.get("style"), STYLES, "Tile"),
        "logo_placement": (asked.get("placement"), list(PLACEMENTS), "sidebar-mark"),
        "logo_accent": (asked.get("accent"), list(ACCENTS), "Blue"),
    }
    for key, (value, allowed, default) in defaults.items():
        if st.session_state.get(key) not in allowed:
            st.session_state[key] = value if value in allowed else default


def _set_accent() -> None:
    colour = ACCENTS[st.session_state["logo_accent"]]
    config.set_option("theme.light.primaryColor", colour)
    config.set_option("theme.dark.primaryColor", colour)


def page_setup() -> None:
    """The browser-tab icon, the sidebar logo and the title, as the bar's controls choose them."""
    _init()
    mark, style = st.session_state["logo_mark"], st.session_state["logo_style"]
    placement, accent = st.session_state["logo_placement"], ACCENTS[st.session_state["logo_accent"]]
    if config.get_option("theme.light.primaryColor") != accent:  # an accent asked for in the URL
        _set_accent()
        st.rerun()  # the browser takes the theme at the start of a run
    icon = mark_svg(mark, style, accent)
    st.set_page_config(page_title=NAME, page_icon=data_uri(icon), layout="wide")
    if placement == "sidebar-mark":
        st.logo(icon, size="large")
        st.title(NAME)
    elif placement == "sidebar-wordmark":
        # `icon_image` is what shows at the top left while the sidebar is collapsed.
        st.logo(wordmark_svg(mark, style, accent), size="large", icon_image=icon)
    else:
        image, title = st.columns([1, 24], vertical_alignment="center")
        image.image(icon, width=56)
        title.title(NAME)
    with st.expander("PROTOTYPE: every mark at 16, 24, 32 and 64 px, on the light and the dark background"):
        st.html(contact_sheet(accent))


def bar() -> None:
    """The prototype bar, pinned to the bottom of the page. Not part of any design."""
    keys = list(MARKS)

    def step(by: int) -> None:
        st.session_state["logo_mark"] = keys[(keys.index(st.session_state["logo_mark"]) + by) % len(keys)]

    with st.bottom, st.container(border=True):
        back, pick, forward, style, place, accent = st.columns([1, 8, 1, 3, 6, 3], vertical_alignment="bottom")
        back.button("←", on_click=step, args=(-1,), key="logo_back")
        pick.segmented_control(
            "PROTOTYPE: mark", keys, format_func=lambda k: f"{k} · {MARKS[k]}", key="logo_mark", required=True
        )
        forward.button("→", on_click=step, args=(1,), key="logo_forward")
        style.segmented_control("Style", STYLES, key="logo_style", required=True)
        place.selectbox("Where it sits", list(PLACEMENTS), format_func=PLACEMENTS.get, key="logo_placement")
        accent.selectbox("Accent", list(ACCENTS), key="logo_accent", on_change=_set_accent)
    state = {
        "variant": st.session_state["logo_mark"],
        "style": st.session_state["logo_style"],
        "placement": st.session_state["logo_placement"],
        "accent": st.session_state["logo_accent"],
    }
    st.query_params.update(state)
    with st.expander("PROTOTYPE state"):
        st.json(state | {"primaryColor": ACCENTS[state["accent"]]})
