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


def test_notebook_imports_from_the_package():
    assert any("from exportspredictor" in source for source in _code_cells())


def test_notebook_runs_in_fixture_mode_and_saves_artefacts(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(tmp_path))
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()

    loaded = store.load_latest(tmp_path)
    assert loaded.metadata["source"] == "fixture"
    assert loaded.baseline.height == 15
