"""The app page where an Advisor builds a Scenario is the Scenario Builder.

Its old name read as a list of ready-made Scenarios. No tracked text file uses it, in any case or across a line break,
except the past spec run logs under `docs/runs/`, which are history. The pattern is built in pieces so this file does
not match itself.
"""

import re

from test_encoding import _tracked_text

OLD_NAME = re.compile(r"\bscenario" + r"s(?:\s|#|\\n|\")*pages?\b", re.IGNORECASE)
HISTORY = "docs/runs/"


def test_no_tracked_file_uses_the_old_page_name():
    found = [
        (name, text[max(match.start() - 40, 0) : match.end() + 10])
        for name, text in _tracked_text().items()
        if not name.startswith(HISTORY)
        for match in OLD_NAME.finditer(text)
    ]
    assert not found, f"Call the page the Scenario Builder: {found}"


# The page once called "Why?" became each pair's detail panel; the model card and the notebook still named it.
# Research notes, like the run logs, are dated history.
WHY_PAGE = re.compile(r"\bWhy\?" + r"(?:\s|\\n)*page\b", re.IGNORECASE)


def test_no_tracked_file_names_the_why_page_that_became_the_detail_panel():
    found = [
        (name, text[max(match.start() - 40, 0) : match.end() + 10])
        for name, text in _tracked_text().items()
        if not name.startswith((HISTORY, "docs/research/"))
        for match in WHY_PAGE.finditer(text)
    ]
    assert not found, f"Call it the pair's detail panel: {found}"
