"""The app and the API take every year from the package (CODING_STANDARDS.md, "One definition per constant").

A year written into a page or an endpoint (`"Carried forward from 2021"`) goes stale when the
data moves on. The package defines it once (`tariffs.LAST_TRAINS_YEAR`, the artefact's
`gap_years` and `baseline_year`), and the app reads it from the API.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted([*(ROOT / "app").rglob("*.py"), *(ROOT / "api").rglob("*.py")])
YEAR = re.compile(r"(?<!\d)20\d\d(?!\d)")
# Names that contain a year but never go stale: the HS nomenclature is called after its edition.
NAMES = ["2012 edition"]


def _years(path: Path) -> list[str]:
    """Year-like literals in code and strings (docstrings and comments are prose, and are not checked)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in docstrings:
            continue
        if isinstance(node.value, int) and not isinstance(node.value, bool) and 2000 <= node.value <= 2099:
            found.append(f"line {node.lineno}: {node.value}")
        elif isinstance(node.value, str) and YEAR.search(re.sub("|".join(NAMES), "", node.value)):
            found.append(f"line {node.lineno}: {node.value!r}")
    return found


@pytest.mark.parametrize("path", FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_no_year_is_written_into_the_app_or_the_api(path):
    assert _years(path) == [], f"{path.name} hard-codes a year; read it from the package or the API"
