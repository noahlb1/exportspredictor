"""The app page where an Advisor builds a Scenario is the Scenario Builder (#150, finding F13 on #139).

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
