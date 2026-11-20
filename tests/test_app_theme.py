"""The app's theme (spec #103, "Theme and logo"): colours set per mode, so the app follows the computer's light or dark mode."""

import tomllib
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
CONFIG = APP_DIR / ".streamlit" / "config.toml"
ACCENT = "#2a78d6"


def _theme() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))["theme"]


def test_the_top_level_theme_section_sets_no_colour():
    # A colour under [theme] makes a custom theme that stops following the computer's mode.
    top_level = {name: value for name, value in _theme().items() if not isinstance(value, dict)}

    assert not [name for name in top_level if "color" in name.lower()], top_level


def test_the_accent_is_set_in_both_mode_sections():
    theme = _theme()

    assert theme["light"]["primaryColor"] == ACCENT
    assert theme["dark"]["primaryColor"] == ACCENT


def test_each_mode_gives_the_page_and_the_sidebar_their_own_backgrounds():
    theme = _theme()

    for mode in ("light", "dark"):
        assert theme[mode]["backgroundColor"] != theme[mode]["sidebar"]["backgroundColor"]
