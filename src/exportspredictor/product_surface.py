"""The terms that keep the project's history off its product surfaces (ADR 0013).

Product text names earlier configurations by role ("the model alone", "the starting
configuration"), never by version name. `banned_terms` finds the history terms in a text:

- a version name, `v` and a number (`v1`, `v2`, `v1.2`), also inside an identifier such as
  `beats_v2`, but not inside a word (`prev2`); only the release tags in `RELEASE_TAGS` and data sources'
  own release names (`V202601`, "Licence v3.0") pass;
- a phrase in `HISTORY_PHRASES`, in any case, plural, or across a line break.

`notebook_text` is a notebook as its reader sees it: the prose and what the outputs show. `readme_text` is the
README without its reproduction section, which is part of the record.
"""

import re
from collections.abc import Mapping
from html import unescape

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
    "this version",
    "that version",
    "old measure",
)

# The README's `##` section that is record, not product.
README_RECORD_SECTION = "Rebuild the data and artefacts"
_SECTION_BREAK = "\n## "

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


def notebook_text(notebook: Mapping) -> str:
    """What a reader of a saved notebook (its JSON) sees outside the code: every markdown cell, and each output's
    printed text, table (tags stripped), Markdown and the labels of its chart. Code, and what an SVG carries
    unseen (its metadata names the drawing library and its version), are left out."""
    texts = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            texts.append(_joined(cell["source"]))
        for output in cell.get("outputs", []):
            texts.append(_joined(output.get("text", "")))
            for kind, value in output.get("data", {}).items():
                value = _joined(value)
                if kind == "image/svg+xml":
                    texts += re.findall(r"<text[^>]*>(.*?)</text>", value, flags=re.DOTALL)
                elif kind == "text/html":
                    texts.append(re.sub(r"<style.*?</style>|<[^>]+>", " ", value, flags=re.DOTALL))
                elif kind.startswith("text/"):
                    texts.append(value)
    return unescape("\n".join(text for text in texts if text))


def _joined(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def readme_text(readme: str) -> str:
    """The README outside its reproduction section (`README_RECORD_SECTION`, with its `###` subsections)."""
    return _SECTION_BREAK.join(
        section for section in readme.split(_SECTION_BREAK) if section.splitlines()[0] != README_RECORD_SECTION
    )
