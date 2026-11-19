"""PROTOTYPE, throwaway (wayfinder ticket #101): never merge.

Which colours the app uses in light mode, and the dark colours that go with them. Four themes plus
Streamlit's own, on the four settled pages, switched with `?theme=` or the bar pinned to the bottom.

    A  Crisp white           white page, pale cool-grey sidebar, near-black text
    B  Grey canvas           pale grey page, white sidebar, white inputs and tables
    C  Navy sidebar          white page, a navy sidebar in both modes
    D  Warm paper, darker    Streamlit's warm look with the contrast turned up
    today                    Streamlit 1.67's default, with only the accent set

Each theme sets a light and a dark half; the app still follows the computer's mode. The bar writes the
chosen theme to `.streamlit/config.toml`, which Streamlit reloads while running, so the choice applies
to every open tab of this server. The page layouts are locked to the ones already settled.

Run it (PowerShell, from this worktree; the API must be up on :8000):

    $env:EXPORTSPREDICTOR_PROTOTYPE = "1"; ./scripts/wt.ps1 streamlit run app/Home.py
"""

import time
from pathlib import Path

import streamlit as st

CONFIG = Path(__file__).parent.parent / ".streamlit" / "config.toml"
ACCENT = "#2a78d6"
CHARTS_LIGHT = ["#2a78d6", "#d9730d", "#0f9d8a", "#7c4ddb", "#d6336c"]
CHARTS_DARK = ["#5b9dea", "#f0a04b", "#3fc1ac", "#a78bfa", "#f06595"]

# The named colours behind st.info, st.warning, st.error and badges, shared by themes A to D.
NAMED_LIGHT = {
    "blueBackgroundColor": "#e3eefc", "blueTextColor": "#17467f",
    "yellowBackgroundColor": "#fff0c7", "yellowTextColor": "#6b4a00",
    "redBackgroundColor": "#fde4e2", "redTextColor": "#8c1d18",
    "greenBackgroundColor": "#dff3e6", "greenTextColor": "#14532d",
    "grayBackgroundColor": "#e9edf2", "grayTextColor": "#3b4554",
}
NAMED_DARK = {
    "blueBackgroundColor": "#16304f", "blueTextColor": "#bcd8fb",
    "yellowBackgroundColor": "#4a3a10", "yellowTextColor": "#ffe29a",
    "redBackgroundColor": "#4d1c1a", "redTextColor": "#ffc9c5",
    "greenBackgroundColor": "#173b27", "greenTextColor": "#bfeccd",
    "grayBackgroundColor": "#2a303b", "grayTextColor": "#c3cad6",
}

THEMES: dict[str, dict] = {
    "A": {
        "name": "Crisp white",
        "light": {
            "backgroundColor": "#ffffff", "secondaryBackgroundColor": "#f1f4f8", "textColor": "#14181f",
            "borderColor": "#cfd6e0", "linkColor": "#1d5fb4",
            "dataframeHeaderBackgroundColor": "#eef2f7", "dataframeHeaderTextColor": "#3b4554",
            "dataframeBorderColor": "#dde2ea",
            "sidebar": {"backgroundColor": "#f4f6f9", "secondaryBackgroundColor": "#ffffff", "borderColor": "#cfd6e0"},
        },
        "dark": {
            "backgroundColor": "#0e1117", "secondaryBackgroundColor": "#1b212c", "textColor": "#f2f4f8",
            "borderColor": "#363f4e", "linkColor": "#6eabf2",
            "dataframeHeaderBackgroundColor": "#1b212c", "dataframeHeaderTextColor": "#b6bfcc",
            "dataframeBorderColor": "#2a313d",
            "sidebar": {"backgroundColor": "#161b24", "secondaryBackgroundColor": "#0e1117", "borderColor": "#363f4e"},
        },
    },
    "B": {
        "name": "Grey canvas",
        "light": {
            "backgroundColor": "#eef1f5", "secondaryBackgroundColor": "#ffffff", "textColor": "#14181f",
            "borderColor": "#c6ced9", "linkColor": "#1d5fb4",
            "dataframeHeaderBackgroundColor": "#dfe5ed", "dataframeHeaderTextColor": "#2c3542",
            "dataframeBorderColor": "#d3dae3",
            "sidebar": {"backgroundColor": "#ffffff", "secondaryBackgroundColor": "#eef1f5", "borderColor": "#c6ced9"},
        },
        "dark": {
            "backgroundColor": "#181c24", "secondaryBackgroundColor": "#252b37", "textColor": "#f2f4f8",
            "borderColor": "#3d4656", "linkColor": "#6eabf2",
            "dataframeHeaderBackgroundColor": "#2b3240", "dataframeHeaderTextColor": "#c3cad6",
            "dataframeBorderColor": "#333b49",
            "sidebar": {"backgroundColor": "#0f1218", "secondaryBackgroundColor": "#1c212b", "borderColor": "#3d4656"},
        },
    },
    "C": {
        "name": "Navy sidebar",
        "light": {
            "backgroundColor": "#ffffff", "secondaryBackgroundColor": "#eef2f8", "textColor": "#101828",
            "borderColor": "#cbd3df", "linkColor": "#1d5fb4",
            "dataframeHeaderBackgroundColor": "#e8eef7", "dataframeHeaderTextColor": "#22304a",
            "dataframeBorderColor": "#d9e0ea",
            "sidebar": {
                "backgroundColor": "#10294d", "secondaryBackgroundColor": "#1d3b69", "textColor": "#eef3fa",
                "borderColor": "#3d5a87", "primaryColor": "#7db4f5", "linkColor": "#a9cdf9",
            },
        },
        "dark": {
            "backgroundColor": "#0b1220", "secondaryBackgroundColor": "#18233a", "textColor": "#eef2f8",
            "borderColor": "#34425c", "linkColor": "#6eabf2",
            "dataframeHeaderBackgroundColor": "#18233a", "dataframeHeaderTextColor": "#b9c4d6",
            "dataframeBorderColor": "#263249",
            "sidebar": {
                "backgroundColor": "#10294d", "secondaryBackgroundColor": "#1d3b69", "textColor": "#eef3fa",
                "borderColor": "#3d5a87", "primaryColor": "#7db4f5", "linkColor": "#a9cdf9",
            },
        },
    },
    "D": {
        "name": "Warm paper, darker",
        "light": {
            "backgroundColor": "#fbf9f4", "secondaryBackgroundColor": "#efebe1", "textColor": "#1b1915",
            "borderColor": "#cdc6b6", "linkColor": "#1d5fb4",
            "dataframeHeaderBackgroundColor": "#ece7db", "dataframeHeaderTextColor": "#3d382e",
            "dataframeBorderColor": "#ddd7c9",
            "sidebar": {"backgroundColor": "#f1ede3", "secondaryBackgroundColor": "#fbf9f4", "borderColor": "#cdc6b6"},
        },
        "dark": {
            "backgroundColor": "#1a1917", "secondaryBackgroundColor": "#2a2825", "textColor": "#f5f2ec",
            "borderColor": "#48443d", "linkColor": "#6eabf2",
            "dataframeHeaderBackgroundColor": "#2a2825", "dataframeHeaderTextColor": "#c9c3b8",
            "dataframeBorderColor": "#37342f",
            "sidebar": {"backgroundColor": "#232220", "secondaryBackgroundColor": "#1a1917", "borderColor": "#48443d"},
        },
    },
    "today": {"name": "Streamlit's default (today)", "light": {}, "dark": {}},
}

TINTS = {
    "28% blue, bold (now)": "background-color: rgba(42, 120, 214, 0.28); font-weight: 600",
    "16% blue, bold": "background-color: rgba(42, 120, 214, 0.16); font-weight: 600",
    "40% blue, bold": "background-color: rgba(42, 120, 214, 0.40); font-weight: 600",
}
SETTLED = {"Blocked Opportunities": "C", "Scenarios": "B", "Model Evaluation": "C"}
SETTLED_EXTRAS = {"Scenario caution": "Page line and Scenario line"}


def _value(value) -> str:
    return "[" + ", ".join(f'"{v}"' for v in value) + "]" if isinstance(value, list) else f'"{value}"'


def toml(key: str) -> str:
    """The whole config file for one theme: colours only under [theme.light] and [theme.dark] (#97)."""
    lines = [f"# PROTOTYPE (#101) theme: {key}", "# Written by app/prototype_theme.py; never merge."]
    for mode, charts, named in (("light", CHARTS_LIGHT, NAMED_LIGHT), ("dark", CHARTS_DARK, NAMED_DARK)):
        options = {"primaryColor": ACCENT} | THEMES[key][mode]
        if key != "today":
            options |= {"chartCategoricalColors": charts} | named
        side = options.pop("sidebar", {})
        lines += ["", f"[theme.{mode}]"] + [f"{k} = {_value(v)}" for k, v in options.items()]
        if side:
            lines += ["", f"[theme.{mode}.sidebar]"] + [f"{k} = {_value(v)}" for k, v in side.items()]
    return "\n".join(lines) + "\n"


def on_disk() -> str:
    first = CONFIG.read_text(encoding="utf-8").splitlines()[0] if CONFIG.exists() else ""
    key = first.rpartition("theme: ")[2]
    return key if key in THEMES else "today"


def bar() -> tuple[str, str]:
    """The prototype bar, pinned to the bottom of every page. Not part of any design."""
    keys = list(THEMES)
    asked = st.query_params.get("theme")
    if st.session_state.get("theme") not in keys:
        st.session_state["theme"] = st.session_state.get("_theme_kept") or (asked if asked in keys else on_disk())
    if st.session_state.get("tint") not in TINTS:
        st.session_state["tint"] = st.session_state.get("_tint_kept") or next(iter(TINTS))

    def step(by: int) -> None:
        st.session_state["theme"] = keys[(keys.index(st.session_state["theme"]) + by) % len(keys)]

    with st.bottom, st.container(border=True):
        columns = st.columns([1, 7, 1, 5, 6], vertical_alignment="bottom")
        columns[0].button("←", on_click=step, args=(-1,), key="theme_back")
        key = columns[1].selectbox(
            "PROTOTYPE: colours", keys, key="theme",
            format_func=lambda k: THEMES[k]["name"] if k == "today" else f"{k} · {THEMES[k]['name']}",
        )
        columns[2].button("→", on_click=step, args=(1,), key="theme_forward")
        tint = columns[3].selectbox("Selected row", list(TINTS), key="tint")
        columns[4].caption(
            f"Showing the **{st.context.theme.type}** half. Switch your computer (or Streamlit's menu, "
            "Settings) between light and dark to see the other."
        )
    st.session_state.update(_theme_kept=key, _tint_kept=tint)
    st.query_params["theme"] = key
    wanted = toml(key)
    if not CONFIG.exists() or CONFIG.read_text(encoding="utf-8") != wanted:
        CONFIG.write_text(wanted, encoding="utf-8")
        time.sleep(0.7)  # Streamlit polls the config file every 200ms
        st.rerun()
    return key, TINTS[tint]


def state_box() -> None:
    key = st.session_state["theme"]
    with st.expander("PROTOTYPE state"):
        st.json({"theme": key, "mode": st.context.theme.type, "selected row": st.session_state["tint"]})
        st.code(toml(key), language="toml")
