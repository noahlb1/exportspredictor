"""PROTOTYPE, throwaway (wayfinder ticket #97): never merge.

The chosen mark (the Profile container ship, bare, with the app's name) tried in the app itself, on
the Leaderboard's own route over the split view settled in #95. The bar pinned to the bottom of the
page switches where it sits, its size and the browser-tab icon. The marks it was chosen from are in
`prototype_logo_gallery.html`, `prototype_logo_ships.html` and `prototype_logo_profiles.html`.

Run it (PowerShell, from this worktree; the API must be up on :8000):

    $env:EXPORTSPREDICTOR_PROTOTYPE = "1"; ./scripts/wt.ps1 streamlit run app/Home.py

The accent is `primaryColor` under `[theme.light]` and `[theme.dark]` in `.streamlit/config.toml`, so
the app follows the computer's light or dark setting. The logo's colours follow `st.context.theme.type`.
"""

import base64

import streamlit as st

NAME = "UK Exports Predictor"
BLUE = "#2a78d6"
BLUE_ON_DARK = "#5b9dea"  # the blue, lightened so the bare mark stays readable on the dark background
PLACEMENTS = {
    "sidebar": "Ship and name in the sidebar, no page title",
    "sidebar-title": "Ship and name in the sidebar, page title kept",
    "sidebar-ship": "Ship alone in the sidebar, page title kept",
    "title": "Ship beside the page title",
}
SIZES = ["large", "medium", "small"]
TAB_ICONS = ["Tile", "Bare"]


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
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">{inner}</svg>'


def wordmark_svg(dark: bool) -> str:
    """The bare ship with the app's name, coloured for the light or the dark theme."""
    text = "#f4f5f7" if dark else "#1f2430"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="412" height="64" viewBox="0 0 412 64">'
        f"{ship(BLUE_ON_DARK if dark else BLUE)}"
        f'<text x="72" y="44" fill="{text}" font-family="Source Sans Pro,Segoe UI,Arial,sans-serif" font-size="36" '
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
        "logo_tab": (asked.get("tab"), TAB_ICONS, "Tile"),
    }
    for key, (value, allowed, default) in defaults.items():
        if st.session_state.get(key) not in allowed:
            st.session_state[key] = value if value in allowed else default


def page_setup() -> None:
    """The browser-tab icon, the sidebar logo and the title, as the bar's controls choose them."""
    _init()
    placement, size = st.session_state["logo_placement"], st.session_state["logo_size"]
    st.set_page_config(
        page_title=NAME, page_icon=data_uri(mark_svg(st.session_state["logo_tab"], dark=False)), layout="wide"
    )
    dark = st.context.theme.type == "dark"
    bare = mark_svg("Bare", dark)
    if placement in ("sidebar", "sidebar-title"):
        # `icon_image` is what shows at the top left while the sidebar is collapsed.
        st.logo(wordmark_svg(dark), size=size, icon_image=bare)
    elif placement == "sidebar-ship":
        st.logo(bare, size=size)
    if placement == "title":
        image, title = st.columns([1, 22], vertical_alignment="center")
        image.image(bare, width=64)
        title.title(NAME)
    elif placement != "sidebar":
        st.title(NAME)


def bar() -> None:
    """The prototype bar, pinned to the bottom of the page. Not part of any design."""
    keys = list(PLACEMENTS)

    def step(by: int) -> None:
        st.session_state["logo_placement"] = keys[(keys.index(st.session_state["logo_placement"]) + by) % len(keys)]

    st.sidebar.divider()
    st.sidebar.caption("Light or dark: the app follows your computer. To change it here, open the ⋮ menu at the "
                       "top right, then Settings.")
    with st.bottom, st.container(border=True):
        back, pick, forward, size, tab = st.columns([1, 9, 1, 4, 3], vertical_alignment="bottom")
        back.button("←", on_click=step, args=(-1,), key="logo_back")
        pick.selectbox("PROTOTYPE: where the logo sits", keys, format_func=PLACEMENTS.get, key="logo_placement")
        forward.button("→", on_click=step, args=(1,), key="logo_forward")
        size.segmented_control("Sidebar logo size", SIZES, key="logo_size", required=True)
        tab.segmented_control("Browser-tab icon", TAB_ICONS, key="logo_tab", required=True)
    state = {
        "variant": st.session_state["logo_placement"],
        "size": st.session_state["logo_size"],
        "tab": st.session_state["logo_tab"],
    }
    st.query_params.update(state)
    with st.expander("PROTOTYPE state"):
        st.json(state | {"theme": st.context.theme.type})
