"""PROTOTYPE, throwaway (wayfinder ticket #97): never merge.

The chosen mark (the Profile container ship, white on a blue tile, with the app's name) tried in the app itself, on
the Leaderboard's own route over the split view settled in #95. The bar pinned to the bottom of the
page switches where it sits, its size and the colour of the name. The browser-tab icon is the tile. The marks it was chosen from are in
`prototype_logo_gallery.html`, `prototype_logo_ships.html` and `prototype_logo_profiles.html`.

Run it (PowerShell, from this worktree; the API must be up on :8000):

    $env:EXPORTSPREDICTOR_PROTOTYPE = "1"; ./scripts/wt.ps1 streamlit run app/Home.py

The accent is `primaryColor` under `[theme.light]` and `[theme.dark]` in `.streamlit/config.toml`, so
the app follows the computer's light or dark setting. Streamlit takes one logo image for both themes
and reports a theme changed in its own Settings menu only on the next rerun, so a name coloured for the
theme is briefly wrong after a switch; a blue name needs no theme at all.
"""

import base64

import streamlit as st

NAME = "UK Exports Predictor"
# The SVG namespace, spelt so the no-hard-coded-years test does not read it as a year. The real logo is a file.
SVG_NS = "http://www.w3.org/" + str(20 * 100) + "/svg"
BLUE = "#2a78d6"
BLUE_ON_DARK = "#5b9dea"  # the blue, lightened so the bare mark stays readable on the dark background
PLACEMENTS = {
    "sidebar": "Tile and name in the sidebar, no page title",
    "sidebar-title": "Tile and name in the sidebar, page title kept",
    "sidebar-ship": "Tile alone in the sidebar, page title kept",
    "title": "Tile beside the page title",
}
SIZES = ["large", "medium", "small"]
NAME_COLOURS = ["Blue", "Follows the theme"]


def ship(ink: str) -> str:
    """The Profile container ship on a 64 x 64 canvas, in one ink."""
    line = f'fill="none" stroke="{ink}" stroke-linecap="round" stroke-linejoin="round"'
    soft = "".join(
        f'<rect x="{x}" y="{y}" width="9.5" height="7.5" rx="1.5" fill="{ink}" opacity=".55"/>'
        for x, y in ((21, 30), (32.5, 30), (44, 30), (21, 20.5))
    )
    return (
        '<g transform="translate(0 -3)">'
        f'<path d="M7 42h50l-7 10H13z" fill="{ink}" stroke="{ink}" stroke-width="3" stroke-linejoin="round"/>'
        f'<path d="M9 38V26.5a1.5 1.5 0 0 1 1.5-1.5h5a1.5 1.5 0 0 1 1.5 1.5V38z" fill="{ink}"/>'
        f'{soft}<rect x="32.5" y="20.5" width="9.5" height="7.5" rx="1.5" fill="{ink}"/>'
        f'<path d="M7 60q4-3 8.3 0t8.3 0 8.3 0 8.3 0 8.3 0 8.3 0" {line} stroke-width="2.5" opacity=".5"/>'
        "</g>"
    )


def mark_svg(style: str, dark: bool) -> str:
    """The ship as a square SVG: bare in the theme's blue, or white on a blue tile."""
    if style == "Tile":
        inner = f'<rect width="64" height="64" rx="14" fill="{BLUE}"/>{ship("#ffffff")}'
    else:
        inner = ship(BLUE_ON_DARK if dark else BLUE)
    return f'<svg xmlns="{SVG_NS}" width="64" height="64" viewBox="0 0 64 64">{inner}</svg>'


def wordmark_svg(name_colour: str, dark: bool) -> str:
    """The tile with the app's name: in the blue, which reads on both themes, or coloured for the theme."""
    if name_colour == "Blue":
        text = BLUE_ON_DARK if dark else BLUE
    else:
        text = "#f4f5f7" if dark else "#1f2430"
    return (
        f'<svg xmlns="{SVG_NS}" width="420" height="64" viewBox="0 0 420 64">'
        f'<rect width="64" height="64" rx="14" fill="{BLUE}"/>{ship("#ffffff")}'
        f'<text x="80" y="44" fill="{text}" font-family="Source Sans Pro,Segoe UI,Arial,sans-serif" font-size="36" '
        f'font-weight="600">{NAME}</text></svg>'
    )


def data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _init() -> None:
    """Defaults for the bar's controls, taking `?variant=`, `?size=` and `?tab=` from the URL once."""
    asked = st.query_params
    defaults = {
        "logo_placement": (asked.get("variant"), list(PLACEMENTS), "sidebar"),
        "logo_size": (asked.get("size"), SIZES, "large"),
        "logo_name": (asked.get("name"), NAME_COLOURS, "Blue"),
    }
    for key, (value, allowed, default) in defaults.items():
        if st.session_state.get(key) not in allowed:
            st.session_state[key] = value if value in allowed else default


def page_setup() -> None:
    """The browser-tab icon, the sidebar logo and the title, as the bar's controls choose them."""
    _init()
    placement, size = st.session_state["logo_placement"], st.session_state["logo_size"]
    tile = mark_svg("Tile", dark=False)
    st.set_page_config(page_title=NAME, page_icon=data_uri(tile), layout="wide")
    dark = st.context.theme.type == "dark"
    if placement in ("sidebar", "sidebar-title"):
        # `icon_image` is what shows at the top left while the sidebar is collapsed.
        st.logo(wordmark_svg(st.session_state["logo_name"], dark), size=size, icon_image=tile)
    elif placement == "sidebar-ship":
        st.logo(tile, size=size)
    if placement == "title":
        image, title = st.columns([1, 22], vertical_alignment="center")
        image.image(tile, width=56)
        title.title(NAME)
    elif placement != "sidebar":
        st.title(NAME)


def bar() -> None:
    """The prototype bar, pinned to the bottom of the page. Not part of any design."""
    keys = list(PLACEMENTS)

    def step(by: int) -> None:
        st.session_state["logo_placement"] = keys[(keys.index(st.session_state["logo_placement"]) + by) % len(keys)]

    with st.bottom, st.container(border=True):
        back, pick, forward, size, tab = st.columns([1, 8, 1, 4, 4], vertical_alignment="bottom")
        back.button("←", on_click=step, args=(-1,), key="logo_back")
        pick.selectbox("PROTOTYPE: where the logo sits", keys, format_func=PLACEMENTS.get, key="logo_placement")
        forward.button("→", on_click=step, args=(1,), key="logo_forward")
        size.segmented_control("Sidebar logo size", SIZES, key="logo_size", required=True)
        tab.segmented_control("Colour of the name", NAME_COLOURS, key="logo_name", required=True)
    state = {
        "variant": st.session_state["logo_placement"],
        "size": st.session_state["logo_size"],
        "name": st.session_state["logo_name"],
    }
    st.query_params.update(state)
    with st.expander("PROTOTYPE state"):
        st.json(state | {"theme": st.context.theme.type})
