"""The terms that keep the project's history off its product surfaces (ADR 0013).

Product text names earlier configurations by role ("the model alone", "the starting
configuration"), never by version name. `banned_terms` finds the history terms in a text:

- a version name, `v` and a number (`v1`, `v2`, `v1.2`), also inside an identifier such as
  `beats_v2`, but not inside a word (`prev2`); only the release tags in `RELEASE_TAGS` and data sources'
  own release names (`V202601`, "Licence v3.0") pass;
- a phrase in `HISTORY_PHRASES`, in any case, plural, or across a line break.
"""

import re

from exportspredictor.release import TAG

# The first release, and the release the README and the fetch point at.
RELEASE_TAGS = frozenset({"v1.0", TAG})
HISTORY_PHRASES = (
    "earlier version",
    "previous version",
    "older version",
    "frozen version",
    "frozen result",
    "first form",
    "Arm against",
    "gain ladder",
)

# Data sources' own release names: CEPII's dated releases (`V202601`) and the Open Government Licence's `v3.0`.
_SOURCE_RELEASE = r"(?<![A-Za-z0-9])v\d{6}(?![A-Za-z0-9])|Licence\s+v\d+(?:\.\d+)*"
_VERSION = rf"(?:{_SOURCE_RELEASE})|(?<![A-Za-z0-9])v\d+(?:\.\d+)*(?![A-Za-z0-9])"
_PHRASES = (r"\b" + r"s?\s+".join(map(re.escape, phrase.split())) + r"s?\b" for phrase in HISTORY_PHRASES)
_PATTERN = re.compile("|".join([_VERSION, *_PHRASES]), re.IGNORECASE)
_SOURCE = re.compile(_SOURCE_RELEASE, re.IGNORECASE)


def banned_terms(text: str) -> list[str]:
    """The history terms in `text`, as written and in order; empty when it is clean."""
    return [
        m.group() for m in _PATTERN.finditer(text)
        if m.group().lower() not in RELEASE_TAGS and not _SOURCE.fullmatch(m.group())
    ]
