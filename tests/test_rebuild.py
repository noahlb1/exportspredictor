"""The full data rebuild script: an ordered list of download and build steps, run in order."""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rebuild.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("rebuild", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_prints_the_steps_in_order_without_running_them():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list"], capture_output=True, text=True, timeout=60, cwd=REPO, check=False
    )

    assert result.returncode == 0
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("1. ")
    assert "notebook" in lines[-1].lower()


def test_run_executes_steps_in_order_and_stops_at_the_first_failure(tmp_path):
    rebuild = _load_script()
    log = tmp_path / "log.txt"

    def append(word: str) -> list[str]:
        return [sys.executable, "-c", f"open(r'{log}', 'a').write('{word} ')"]

    steps = [
        rebuild.Step("first", append("first")),
        rebuild.Step("second", append("second")),
        rebuild.Step("broken", [sys.executable, "-c", "raise SystemExit(3)"]),
        rebuild.Step("never", append("never")),
    ]

    assert rebuild.run(steps) == 3
    assert log.read_text().split() == ["first", "second"]


def test_a_step_runs_with_its_own_environment_variables(tmp_path):
    rebuild = _load_script()
    log = tmp_path / "log.txt"
    write_env = f"import os; open(r'{log}', 'w').write(os.environ['EXPORTSPREDICTOR_FIXTURE_MODE'])"

    assert rebuild.run([rebuild.Step("env", [sys.executable, "-c", write_env], env={"EXPORTSPREDICTOR_FIXTURE_MODE": "0"})]) == 0
    assert log.read_text() == "0"


def test_the_notebook_step_runs_on_real_data():
    rebuild = _load_script()

    assert rebuild.STEPS[-1].env == {"EXPORTSPREDICTOR_FIXTURE_MODE": "0"}


def test_the_real_build_trains_once_before_the_notebook_reads_it():
    # The notebook reads the saved artefact on real data (#58), so the pipeline runs as its own step,
    # after the ablation, the learning curve and the search it reads.
    rebuild = _load_script()
    modules = [step.command[2] for step in rebuild.STEPS if step.command[1:2] == ["-m"]]

    order = ["exportspredictor.features", "exportspredictor.experiments", "exportspredictor.search",
             "exportspredictor.pipeline", "jupyter"]
    assert all(module in modules for module in order)
    assert [module for module in modules if module in order] == sorted(
        (m for m in modules if m in order), key=order.index
    )
    assert modules.index("exportspredictor.pipeline") == len(modules) - 2
