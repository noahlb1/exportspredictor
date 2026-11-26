"""Notebook smoke run: the write-up and the Model History run end to end in fixture mode and only import logic."""

import ast
import hashlib
import json
import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from exportspredictor import artefacts as store
from exportspredictor import (
    illustrative,
    pipeline,
    realisation,
    realisation_wording,
    reporting,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "exports_predictor.ipynb"
HISTORY = NOTEBOOK.parent / "model_history.ipynb"
# Both notebooks follow the same rules: fixture mode by default, no logic of their own, nothing built in real mode.
NOTEBOOKS = pytest.mark.parametrize("path", [NOTEBOOK, HISTORY], ids=["write-up", "Model History"])


def _code_cells(path: Path = NOTEBOOK):
    notebook = nbformat.read(path, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


@NOTEBOOKS
def test_notebook_defines_no_reusable_functions_or_classes(path):
    for source in _code_cells(path):
        tree = ast.parse(source)
        defined = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        ]
        assert not defined, f"Notebook cell defines logic; move it into the package:\n{source}"


@NOTEBOOKS
def test_fixture_mode_is_the_first_parameter_and_defaults_to_on(path):
    notebook = nbformat.read(path, as_version=4)
    first = next(cell for cell in notebook.cells if cell.cell_type == "code")

    assert "parameters" in first.metadata.get("tags", [])
    assert "FIXTURE_MODE = True" in first.source


def _sections(path: Path = NOTEBOOK):
    """Top-level sections: (heading, cells up to the next heading). The title and parameters are not sections."""
    notebook = nbformat.read(path, as_version=4)
    sections = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            sections.append((cell.source.splitlines()[0], [cell]))
        elif sections:
            sections[-1][1].append(cell)
    return [(heading, cells) for heading, cells in sections if heading != "## Parameters"]


# The argument's sections, in order: the whole notebook after the parameters.
SECTIONS = [
    "## 1. The question and the headline",
    "## 2. The data: Flows, Zero Cells and policy features",
    "## 3. Features and the leakage rules",
    "## 4. The model and how it is scored out-of-fold",
    "## 5. What v2 showed",
    "## 6. Re-choosing the model on the Selection Measure",
    "## 7. The 2023 result and what it cost",
    "## 8. Scenarios and Blocked Opportunities as Illustrative figures",
    "## 9. Explaining Export Potential",
    "## 10. Limits and governance",
    "## 11. Conclusions",
    "## Appendix: frozen results",
]
# No prose cell runs over about this many words; longer detail is linked.
WORD_LIMIT = 300
# The whole notebook's prose comes to about this many words, within 10%.
TOTAL_WORDS = 10_200  # 10,000 at first, plus the ITC comparison in §7
# The parts that no longer decide anything, each one appendix entry.
FROZEN_PARTS = (
    "hyperparameter search", "feature ablation", "learning curve", "v1 comparison", "gain ladder",
    "constraint comparison",
)


def _words(source: str) -> int:
    """Words of prose: whitespace-separated tokens with a letter or digit, so table pipes and rules do not count."""
    return sum(any(character.isalnum() for character in token) for token in source.split())


def _prose() -> list[str]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]



def test_the_sections_are_in_the_argument_s_order():
    assert [heading for heading, _ in _sections()] == SECTIONS


def test_no_prose_cell_runs_over_the_word_limit():
    for source in _prose():
        assert _words(source) <= WORD_LIMIT, f"{_words(source)} words: {source[:200]}"


def test_the_prose_comes_to_about_ten_thousand_words():
    total = sum(_words(source) for source in _prose())

    assert 0.9 * TOTAL_WORDS <= total <= 1.1 * TOTAL_WORDS, total


def test_each_frozen_part_is_one_appendix_entry():
    appendix = next(cells for heading, cells in _sections() if heading.startswith("## Appendix"))
    entries = [line.lower() for cell in appendix if cell.cell_type == "markdown" for line in cell.source.splitlines() if line.startswith("### ")]

    for part in FROZEN_PARTS:
        assert sum(part in entry for entry in entries) == 1, part


def test_the_word_count_reads_whitespace_separated_words():
    assert _words("**Conclusion.** The model\nbeats the | Naive Baseline |") == 7


def test_each_rewritten_section_links_its_choices_to_the_course():
    for heading, cells in _sections():
        if heading in SECTIONS:
            prose = " ".join(cell.source for cell in cells if cell.cell_type == "markdown")
            assert "Course:" in prose or "course:" in prose, f"{heading} names no course concept"


def test_every_section_is_introduced_and_concluded():
    for heading, cells in _sections():
        intro = cells[0].source.split("\n", 1)[-1].strip()
        assert len(intro) > 100, f"{heading} has no introduction"
        last = cells[-1]
        assert last.cell_type == "markdown" and last.source.startswith("**Conclusion.**"), f"{heading} has no conclusion"


def _subheadings(cells) -> str:
    return " ".join(
        line for cell in cells if cell.cell_type == "markdown" for line in cell.source.splitlines() if line.startswith("### ")
    )


def test_the_write_up_covers_governance():
    sections = _sections()
    headings = " ".join(heading for heading, _ in sections)

    for topic in ("Explaining", "governance", "Conclusions"):
        assert topic in headings, topic
    # One entry per course topic, found by its subheading, not by its wording.
    governance = next(cells for heading, cells in sections if "governance" in heading)
    subheadings = _subheadings(governance)
    for topic in ("Model card", "Data lineage", "Bias review", "Regulation", "Drift"):
        assert topic in subheadings, topic


def test_the_write_up_names_the_gap_period():
    text = " ".join(cell.source for _, cells in _sections() for cell in cells if cell.cell_type == "markdown")

    assert "Gap Period" in text


# Calls that train, fit or rebuild a derived table. In real mode the notebook reads what these
# wrote, so each may appear only where it runs in fixture mode.
BUILDS = {
    "baci.prepare", "gravity.build", "hmrc.build", "policy.build", "features.build", "features.build_features",
    "pipeline.build_artefacts", "experiments.run", "ppml.fit_ppml", "ppml.real_sector_flows", "ppml.sector_flows",
    "ppml.add_benchmark", "artefacts.save", "governance.write_model_card", "realisation.realisation_report",
    "realisation.add_to_latest", "realisation.with_itc_comparison",
}


def _builds_outside_fixture_mode(tree: ast.AST) -> list[str]:
    """Calls named in BUILDS that are not inside the fixture-only branch of an `if FIXTURE_MODE` or
    `... if FIXTURE_MODE else ...`."""
    found = []

    def visit(node: ast.AST, fixture_only: bool) -> None:
        if isinstance(node, (ast.If, ast.IfExp)) and isinstance(node.test, ast.Name) and node.test.id == "FIXTURE_MODE":
            visit(node.test, fixture_only)
            for child in node.body if isinstance(node.body, list) else [node.body]:
                visit(child, True)
            for child in node.orelse if isinstance(node.orelse, list) else [node.orelse]:
                visit(child, fixture_only)
            return
        if isinstance(node, ast.Call) and not fixture_only and ast.unparse(node.func) in BUILDS:
            found.append(ast.unparse(node))
        for child in ast.iter_child_nodes(node):
            visit(child, fixture_only)

    visit(tree, False)
    return found


@NOTEBOOKS
def test_in_real_mode_the_notebook_trains_fits_and_builds_nothing(path):
    for source in _code_cells(path):
        assert not _builds_outside_fixture_mode(ast.parse(source)), f"Real mode would build here:\n{source}"


def test_the_build_check_finds_a_build_outside_fixture_mode():
    source = "if FIXTURE_MODE:\n    a = pipeline.build_artefacts(w)\nelse:\n    b = features.build()\nc = ppml.fit_ppml(p) if FIXTURE_MODE else None"

    assert _builds_outside_fixture_mode(ast.parse(source)) == ["features.build()"]


def test_the_notebook_shows_each_main_result_from_the_reporting_module():
    source = "\n".join(_code_cells())

    for result in (
        "provenance", "experiments", "lift_by_tenth", "event_backtests", "ppml_benchmark", "blocked_opportunity_split",
        "agreement_part_variation", "agreement_part_variation_chart", "itc_method", "itc_coefficients",
        "itc_direct_comparison",
    ):
        assert f"reporting.{result}(bundle)" in source, result
    # The history it still shows until #214 comes from the Model History's module.
    for result in ("ranking_by_version", "experiments_chart", "confirmation", "round_count", "gain_ladder"):
        assert f"history.{result}(bundle)" in source, result
    # The preset table reads the presets the notebook runs.
    assert "reporting.preset_scenarios(bundle, preset_results)" in source


@NOTEBOOKS
def test_notebook_imports_from_the_package(path):
    assert any("from exportspredictor" in source for source in _code_cells(path))


def test_notebook_runs_in_fixture_mode_and_saves_artefacts(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(tmp_path))
    committed = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()

    # The copy ran in memory: the committed file is untouched.
    assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == committed

    loaded = store.load_latest(tmp_path)
    assert loaded.metadata["source"] == pipeline.FIXTURE_SOURCE
    assert loaded.baseline.height == 20
    # The feature ablation and the learning curve travel with the artefact, measured on 2022 only.
    experiments = loaded.metadata["experiments"]
    assert experiments["ablation"][0]["dropped"] is None
    assert len(experiments["learning_curve"]) == 9
    assert {row["validation_year"] for row in experiments["ablation"] + experiments["learning_curve"]} == {2022}
    # The Potential Range's coverage is reported on the fixture's held-out half, near its 80% target
    # (14 held-out rows, so one row moves it by 7 points).
    coverage = loaded.evaluation["potential_range"]
    assert coverage["coverage"] == pytest.approx(0.8, abs=0.2)
    printed = " ".join(
        output.get("text", "") for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    )
    assert f"{coverage['coverage']:.0%}" in printed  # the coverage is printed from the artefact
    # The Uncertain Opportunity share is reported as a finding, read from the evaluation report.
    assert f"({coverage['uncertain_opportunity_share']:.0%})" in printed
    # The Scenarios section shows the package's Scenario note, every part of it.
    note = illustrative.scenario_note(loaded)
    for part in ("benchmark", "model_association", "event_backtest"):
        assert note[part]["text"] in printed, part
    assert note["text"] in printed
    # The ITC comparison runs on a synthetic base-year panel and says why part (c) was dropped.
    assert realisation_wording.direct_comparison_text(loaded.evaluation[realisation.REPORT_KEY]) in printed
    # Its coefficient table shows both levels, ITC's figures on HS2 only, and the Naive Baseline column.
    [coefficients] = [cell for cell in notebook.cells if cell.source.strip() == "reporting.itc_coefficients(bundle)"]
    table = " ".join(o.get("data", {}).get("text/html", "") for o in coefficients.outputs)
    for shown in ("HS2", "HS6", "not compared", "Naive Baseline", "Extensive margin", "New-flow size"):
        assert shown in table, shown
    # The pre-registered v1 comparison section runs on the fixture's synthetic frozen v1.
    assert loaded.evaluation["v1_comparison"]["verdict"]["text"] in printed
    # The outputs name the artefact version they come from.
    assert (loaded.version, pipeline.FIXTURE_SOURCE) in reporting.provenance_versions(printed)
    # Charts are static images: SVG, never an interactive chart that GitHub cannot render.
    kinds = {kind for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs for kind in o.get("data", {})}
    assert "image/svg+xml" in kinds
    assert not any("vega" in kind or "javascript" in kind for kind in kinds), kinds


def _provenance_versions(notebook) -> list[tuple[str, str]]:
    """(version, source) of every provenance line in the saved outputs (`reporting.provenance`)."""
    printed = " ".join(
        output.get("text", "") for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    )
    return reporting.provenance_versions(printed)


@NOTEBOOKS
def test_the_committed_notebook_carries_outputs_from_a_real_artefact(path):
    # Read as saved; never executed here. `scripts/rebuild.py --only notebook` (or `--only history`) refreshes them.
    notebook = nbformat.read(path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]

    imports_only = [all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.parse(cell.source).body) for cell in code]
    silent = [cell.source[:60] for cell, quiet in zip(code, imports_only, strict=True) if not cell.outputs and not quiet]
    assert not silent, f"Code cells without outputs; refresh with scripts/rebuild.py --only notebook: {silent}"
    assert not [o for cell in code for o in cell.outputs if o.output_type == "error"]
    kinds = {kind for cell in code for o in cell.outputs for kind in o.get("data", {})}
    assert "image/svg+xml" in kinds
    provenance = _provenance_versions(notebook)
    assert provenance, "the outputs must name the artefact version they come from"
    for version, source in provenance:
        assert re.fullmatch(r"\d{8}T\d{6}Z", version), version
        assert source != pipeline.FIXTURE_SOURCE, "the committed outputs come from the fixture, not real data"


# "Outputs stay small": the largest printed output or table stays under this; a chart
# image under `reporting.CHART_LIMIT_BYTES`.
OUTPUT_LIMIT_BYTES = 15_000


@NOTEBOOKS
def test_no_committed_output_is_large(path):
    notebook = nbformat.read(path, as_version=4)

    sizes = [
        (len(json.dumps(output)), "image/svg+xml" in output.get("data", {}), cell.source[:60])
        for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    ]
    too_large = [
        size for size in sizes if size[0] > (reporting.CHART_LIMIT_BYTES if size[1] else OUTPUT_LIMIT_BYTES)
    ]
    assert not too_large, too_large


@NOTEBOOKS
def test_the_notebook_shows_the_test_year_ledger_from_the_history_module(path):
    assert "history.test_year_ledger()" in "\n".join(_code_cells(path))


# --- The Model History (ADR 0013) --------------------------------------------------------------------------------

# Its chapters, in the order things happened.
HISTORY_CHAPTERS = [
    "## 1. v1",
    "## 2. v2",
    "## 3. The experiment loop and every Arm",
    "## 4. The Blend",
    "## 5. Market Reach's first form and its correction",
    "## 6. The final round's dropped ideas",
    "## 7. Every read of 2023",
    "## 8. Reproduce this record",
]


def test_the_model_history_tells_its_chapters_in_order():
    assert [heading for heading, _ in _sections(HISTORY)] == HISTORY_CHAPTERS


def test_each_model_history_chapter_links_its_adr_or_research_note():
    for heading, cells in _sections(HISTORY):
        prose = " ".join(cell.source for cell in cells if cell.cell_type == "markdown")
        assert "docs/adr/" in prose or "docs/research/" in prose, heading


def test_the_model_history_shows_everything_the_product_no_longer_does():
    source = "\n".join(_code_cells(HISTORY))
    prose = " ".join(cell.source for cell in nbformat.read(HISTORY, as_version=4).cells if cell.cell_type == "markdown")

    for shown in (
        "history.ranking_by_version(bundle)", "history.large_markets_by_version(bundle)", "history.gain_ladder(bundle)",
        "history.confirmation(bundle)", "history.round_count(bundle)", "history.test_year_ledger()",
        "history.final_round(", "history.experiments_chart(bundle)", "history.traded_away(bundle)",
        "history.zero_cells_against_naive(bundle)",
    ):
        assert shown in source, shown
    for told in ("### Ties cannot win", "### The round count", "### The frozen results"):
        assert told in prose, told


def test_reproduce_this_record_lists_the_history_only_commands():
    [cells] = [cells for heading, cells in _sections(HISTORY) if heading == "## 8. Reproduce this record"]
    text = " ".join(cell.source for cell in cells)

    for command in (
        "python -m exportspredictor.pipeline compare",  # the v1 comparison
        "--gain-ladder",
        "docs/research/final-round-2022.md",  # records each command of the final round
        "scripts/rebuild.py --only history",
    ):
        assert command in text, command


# What the Model History may read in real mode: the newest artefact, the committed Test-year Ledger and the saved
# experiment-loop results (with the final round's saved probe). Never an older artefact folder.
OTHER_SOURCES = ("artefacts.load(", "read_evaluation(", "copy_lineage(", "experiments.load(", "load_version(")


def test_the_model_history_reads_only_the_newest_artefact_the_ledger_and_the_loop_s_results():
    source = "\n".join(_code_cells(HISTORY))

    assert "artefacts.load_latest(artefacts_dir())" in source
    assert "loop.load()" in source
    assert not [name for name in OTHER_SOURCES if name in source]
    # Its provenance line names a copy's source without opening that folder.
    assert "reporting.provenance(bundle, lineage=False)" in source


def test_the_model_history_runs_in_fixture_mode(monkeypatch, tmp_path):
    # About 11 s: a kernel start and the fixture build, the least that runs the notebook end to end.
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(tmp_path))
    committed = hashlib.sha256(HISTORY.read_bytes()).hexdigest()
    notebook = nbformat.read(HISTORY, as_version=4)

    NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()

    assert hashlib.sha256(HISTORY.read_bytes()).hexdigest() == committed
    assert not list(tmp_path.iterdir())  # it saves nothing
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    printed = " ".join(o.get("text", "") for o in outputs)
    assert [source for _, source in reporting.provenance_versions(printed)] == [pipeline.FIXTURE_SOURCE]
    kinds = {kind for o in outputs for kind in o.get("data", {})}
    assert "image/svg+xml" in kinds
