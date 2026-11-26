"""Full data rebuild: download every source and rebuild the derived data and artefacts from scratch.

Run from the repo root:
    uv run python scripts/rebuild.py          # run every step in order
    uv run python scripts/rebuild.py --list   # show the steps without running them
    uv run python scripts/rebuild.py --only notebook   # refresh the committed notebook's outputs only
    uv run python scripts/rebuild.py --only history    # refresh the Model History notebook's outputs only

Data lands in `EXPORTSPREDICTOR_DATA_DIR` (default `<repo>/data`) and artefacts in
`EXPORTSPREDICTOR_ARTEFACTS_DIR` (default `<repo>/artefacts`). Steps run in order and the
script stops at the first one that fails. `--only TEXT` runs only the steps whose name contains
TEXT (case-insensitive); `--only notebook` executes the notebook on the real artefact and saves its
outputs into the committed file, training nothing. `--only history` does the same for the Model History notebook.

To add a step, add ONE line to `STEPS` in its place in the order: downloads first, then
derived tables, then the pipeline, which trains, evaluates and saves the artefacts, then the
notebook, which reads them and trains nothing.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
NOTEBOOK = "notebooks/exports_predictor.ipynb"
NOTEBOOK_STEP = "The notebook (reads the saved artefact)"
HISTORY_NOTEBOOK = "notebooks/model_history.ipynb"
# PYTHONUTF8: `jupyter execute` opens a notebook in the locale's encoding (cp1252 on Windows) and saves it as UTF-8,
# which garbles every non-ASCII character of the prose.
REAL_NOTEBOOK_ENV = {"EXPORTSPREDICTOR_FIXTURE_MODE": "0", "PYTHONUTF8": "1"}


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)  # extra environment variables for this step


STEPS = [
    # Downloads and derived tables (BACI Flows and lookups, gravity and WDI, Trade Agreements and
    # Trading Blocs, the feature table) go here, one Step per line, in this order.
    Step("BACI Flows and lookups (download, convert to Parquet)", [PY, "-m", "exportspredictor.baci"]),
    Step("Gravity and WDI features (download CEPII Gravity, WDI, IMF WEO; build Parquet)", [PY, "-m", "exportspredictor.gravity"]),
    Step("Trade Agreement and Trading Bloc features (WTO RTA-IS download, build)", [PY, "-m", "exportspredictor.policy"]),
    Step("Applied Tariffs (TRAINS download via WITS, HS17/HS07 -> HS12, EU schedule, gap fill)", [PY, "-m", "exportspredictor.tariffs"]),
    Step("Feature table (training rows with sampled zeros, UK grid; per year)", [PY, "-m", "exportspredictor.features"]),
    Step("HMRC OTS 2023 exports for the BACI cross-check (download, HS 2022 -> 2012)", [PY, "-m", "exportspredictor.hmrc"]),
    # Every choice is made on 2022: the feature ablation and learning curve, then the hyperparameter search.
    Step("Experiment rows (Sector subset, 2012–2022)", [PY, "-m", "exportspredictor.experiments", "build"]),
    Step("Feature ablation on 2022 (GPU)", [PY, "-m", "exportspredictor.experiments", "ablation"]),
    Step("Learning curve on 2022 (GPU)", [PY, "-m", "exportspredictor.experiments", "curve"]),
    Step("Hyperparameter search on 2022 (GPU, resumable, about 3 h)", [PY, "-m", "exportspredictor.search"]),
    # The experiment loop chooses the model form on the Selection Measure, on 2022 only;
    # the pipeline serves the form it chose and records its results. Each step resumes from its cache.
    *(
        Step(f"Experiment loop on 2022: {command} (GPU, resumable)", [PY, "-m", "exportspredictor.loop", command])
        for command in ("noise", "naive", "select", "two-part", "scope", "product-detail", "constraints")
    ),
    Step("Five-fold confirmation of the loop's best form on 2022 (GPU, resumable, about 1 h)",
         [PY, "-m", "exportspredictor.loop", "confirm"]),
    # Trains the fold models, scores the Gap Period, evaluates on 2023 (read here for the first
    # time) and saves a new artefact version. Fold and test-year models are cached, so a re-run resumes.
    # The default run: the Event Backtests, the constraint comparison and the gain ladder are opt-in
    # (`--every-part` for all three, about two more hours).
    Step("Train the fold and test-year models, evaluate and save the artefacts (GPU, about 2.5 h)", [PY, "-m", "exportspredictor.pipeline"]),
    # The notebooks run on real data here (each defaults to fixture mode, as the tests run it), read
    # the newest artefact and save their outputs into the committed files: the one way to refresh them.
    # The Model History also reads the Test-year Ledger and the experiment loop's saved results.
    Step(
        "The Model History (reads the saved artefact, the Test-year Ledger and the loop's results)",
        [PY, "-m", "jupyter", "execute", "--inplace", HISTORY_NOTEBOOK],
        env=REAL_NOTEBOOK_ENV,
    ),
    Step(NOTEBOOK_STEP, [PY, "-m", "jupyter", "execute", "--inplace", NOTEBOOK], env=REAL_NOTEBOOK_ENV),
]


def run(steps: list[Step]) -> int:
    """Run the steps in order from the repo root. Returns 0, or the first failing exit code."""
    for number, step in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] {step.name}", flush=True)
        code = subprocess.run(step.command, cwd=REPO, env={**os.environ, **step.env}, check=False).returncode
        if code != 0:
            print(f"Step {number} failed with exit code {code}: {step.name}", file=sys.stderr)
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the steps without running them")
    parser.add_argument("--only", metavar="TEXT", help="run only the steps whose name contains TEXT (case-insensitive)")
    args = parser.parse_args()
    steps = [step for step in STEPS if args.only is None or args.only.lower() in step.name.lower()]
    if not steps:
        parser.error(f"no step name contains {args.only!r}; see --list")
    if args.list:
        for number, step in enumerate(steps, start=1):
            print(f"{number}. {step.name}")
        return 0
    return run(steps)


if __name__ == "__main__":
    sys.exit(main())
