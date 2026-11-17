"""The README's manual UI checklist keeps up with the app: every page, a Scenario run and a CSV export."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def _checklist() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Manual UI checklist", 1)[1]
    return section.split("\n## ", 1)[0]


def _page_titles() -> list[str]:
    pages = [REPO / "app" / "Home.py", *sorted((REPO / "app" / "pages").glob("*.py"))]
    return [re.search(r'st\.title\("([^"]+)"\)', page.read_text(encoding="utf-8")).group(1) for page in pages]


def test_checklist_names_every_app_page():
    checklist = _checklist()
    missing = [title for title in _page_titles() if title not in checklist]
    assert not missing, f"Add these pages to the README's manual UI checklist: {missing}"


def test_checklist_covers_a_scenario_run_and_a_csv_export():
    checklist = _checklist()
    assert "Scenario" in checklist
    assert "CSV" in checklist


def test_readme_documents_the_one_command_run_and_the_rebuild_script():
    text = README.read_text(encoding="utf-8")
    assert "podman compose up" in text
    assert "scripts/rebuild.py" in text
    assert "uv sync" in text


def test_checklist_checks_values_are_in_usd_with_no_pounds():
    checklist = _checklist()
    assert "Values are nominal US dollars, as reported by BACI" in checklist
    assert "£" not in checklist
