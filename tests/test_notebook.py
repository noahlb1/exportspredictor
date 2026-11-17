"""Notebook smoke run: the write-up runs end to end in fixture mode and only imports logic."""

import ast
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from exportspredictor import artefacts as store

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "exports_predictor.ipynb"


def _code_cells():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_defines_no_reusable_functions_or_classes():
    for source in _code_cells():
        tree = ast.parse(source)
        defined = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        ]
        assert not defined, f"Notebook cell defines logic; move it into the package:\n{source}"


def test_fixture_mode_is_the_first_parameter_and_defaults_to_on():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    first = next(cell for cell in notebook.cells if cell.cell_type == "code")

    assert "parameters" in first.metadata.get("tags", [])
    assert "FIXTURE_MODE = True" in first.source


def _sections():
    """Top-level sections: (heading, cells up to the next heading). The title and parameters are not sections."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    sections = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            sections.append((cell.source.splitlines()[0], [cell]))
        elif sections:
            sections[-1][1].append(cell)
    return [(heading, cells) for heading, cells in sections if heading != "## Parameters"]


def test_every_section_is_introduced_and_concluded():
    for heading, cells in _sections():
        intro = cells[0].source.split("\n", 1)[-1].strip()
        assert len(intro) > 100, f"{heading} has no introduction"
        last = cells[-1]
        assert last.cell_type == "markdown" and last.source.startswith("**Conclusion.**"), f"{heading} has no conclusion"


def test_the_write_up_covers_governance():
    headings = " ".join(heading for heading, _ in _sections())
    text = " ".join(cell.source for _, cells in _sections() for cell in cells)

    for topic in ("Explaining", "Governance", "Conclusions"):
        assert topic in headings, topic
    for topic in ("### Model card", "### Data lineage", "### Bias review", "EU AI Act", "### Drift", "HMRC"):
        assert topic in text, topic


def test_notebook_imports_from_the_package():
    assert any("from exportspredictor" in source for source in _code_cells())


def test_notebook_runs_in_fixture_mode_and_saves_artefacts(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(tmp_path))
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()

    loaded = store.load_latest(tmp_path)
    assert loaded.metadata["source"] == "fixture"
    assert loaded.baseline.height == 15
