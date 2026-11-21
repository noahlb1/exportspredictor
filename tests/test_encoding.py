"""No tracked text file holds mojibake: UTF-8 read as cp1252 and saved again.

On this machine a tool that opens a file without naming an encoding reads cp1252, so an en
dash comes back as three characters starting U+00E2 U+20AC, a minus sign as U+00E2 U+02C6, "£"
as U+00C2 U+00A3 and "é" as U+00C3 U+00A9. Spec #66's notebook refresh garbled the prose this
way and every other check passed; spec #113 left garbled minus signs the first pattern missed.
`scripts/wt.ps1` sets PYTHONUTF8; this test catches a tool that runs without it. The patterns
are written as escapes so this file does not match itself.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# A UTF-8 continuation byte (0x80-0xBF) as cp1252 shows it, or as a C1 control where cp1252 has no character.
_CONTINUATION = "".join(bytes([b]).decode("cp1252", errors="ignore") or chr(b) for b in range(0x80, 0xC0))
# Lead byte 0xE2 (punctuation and symbols U+2000-U+2FFF, three bytes), or 0xC2 / 0xC3 (Latin-1, two bytes).
MOJIBAKE = re.compile(f"[\u00e2\u00c2\u00c3][{re.escape(_CONTINUATION)}]")
# Garbled text that is meant to be there, by file, as the exact text allowed.
ALLOWED = {
    # BACI's raw country file is mis-encoded; `baci` repairs it, and its docstring and test quote it.
    "src/exportspredictor/baci.py": {"C\u00c3\u00b4te"},
    "tests/test_baci.py": {"C\u00c3\u00b4te"},
    # A known data defect in data/derived/country_map.parquet, printed by the gravity cell;
    # rebuilding the gravity step fixes it, and then this entry goes.
    "notebooks/exports_predictor.ipynb": {"Barth\u00c3\u00a9lemy"},
}


def _tracked_text() -> dict[str, str]:
    names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    texts = {}
    for name in names.decode().split("\0"):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue  # binary
        try:
            texts[name] = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return texts


def test_no_tracked_text_file_holds_mojibake():
    found = []
    for name, text in _tracked_text().items():
        for match in MOJIBAKE.finditer(text):
            around = text[max(match.start() - 12, 0) : match.end() + 12]
            if not any(allowed in around for allowed in ALLOWED.get(name, ())):
                found.append((name, around))
    assert not found, f"UTF-8 saved in the wrong encoding; restore these from git: {found}"


def test_the_check_finds_a_garbled_en_dash_and_accent():
    assert MOJIBAKE.search("2012 \u00e2\u20ac\u201c 2024")
    assert MOJIBAKE.search("Saint Barth\u00c3\u00a9lemy")
    assert not MOJIBAKE.search("2012 \u2013 2024, Saint Barth\u00e9lemy, \u00a3")


def test_the_check_finds_every_garbled_punctuation_mark_and_symbol():
    for text in "\u2212\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u2022\u00a3\u00a0\u00b0\u00e9\u2264\u2265\u2192":
        garbled = text.encode("utf-8").decode("cp1252", errors="replace")
        assert MOJIBAKE.search(garbled), repr(text)
        assert not MOJIBAKE.search(text), repr(text)
