"""The app talks to the model only through the API over HTTP (CLAUDE.md "Shape"): no app file imports the package."""

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP_FILES = sorted(APP_DIR.rglob("*.py"))


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


@pytest.mark.parametrize("path", APP_FILES, ids=lambda path: str(path.relative_to(APP_DIR)))
def test_no_app_file_imports_the_package(path):
    imported = [m for m in _imported_modules(path) if m.split(".")[0] in {"exportspredictor", "api"}]

    assert imported == [], f"{path.name} imports {imported}; read it from the API instead"


def test_every_page_has_its_own_sidebar_number():
    numbers = [page.name.split("_", 1)[0] for page in sorted((APP_DIR / "pages").glob("*.py"))]

    assert len(numbers) == len(set(numbers)), numbers
