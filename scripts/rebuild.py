"""Full data rebuild: download every source and rebuild the derived data and artefacts from scratch.

Run from the repo root:
    uv run python scripts/rebuild.py          # run every step in order
    uv run python scripts/rebuild.py --list   # show the steps without running them

Data lands in `EXPORTSPREDICTOR_DATA_DIR` (default `<repo>/data`) and artefacts in
`EXPORTSPREDICTOR_ARTEFACTS_DIR` (default `<repo>/artefacts`). Steps run in order and the
script stops at the first one that fails.

To add a step, add ONE line to `STEPS` in its place in the order: downloads first, then
derived tables, then the notebook, which trains, evaluates and saves the artefacts.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)  # extra environment variables for this step


STEPS = [
    # Downloads and derived tables (#12 BACI Flows and lookups, #14 gravity and WDI, #13
    # Trade Agreements and Trading Blocs, #15 the feature table) go here, one Step per line, in this order.
    Step("BACI Flows and lookups (download, convert to Parquet)", [PY, "-m", "exportspredictor.baci"]),
    Step("Gravity and WDI features (download CEPII Gravity, WDI, IMF WEO; build Parquet)", [PY, "-m", "exportspredictor.gravity"]),
    Step("Trade Agreement and Trading Bloc features (WTO RTA-IS download, build)", [PY, "-m", "exportspredictor.policy"]),
    Step("Applied Tariffs (TRAINS download via WITS, HS17/HS07 -> HS12, EU schedule, gap fill)", [PY, "-m", "exportspredictor.tariffs"]),
    Step("Feature table (training rows with sampled zeros, UK grid; per year)", [PY, "-m", "exportspredictor.features"]),
    Step("HMRC OTS 2023 exports for the BACI cross-check (download, HS 2022 -> 2012)", [PY, "-m", "exportspredictor.hmrc"]),
    # Every choice is made on 2022: the feature ablation and learning curve (#53), then the search (#54).
    Step("Experiment rows (Sector subset, 2012–2022)", [PY, "-m", "exportspredictor.experiments", "build"]),
    Step("Feature ablation on 2022 (GPU)", [PY, "-m", "exportspredictor.experiments", "ablation"]),
    Step("Learning curve on 2022 (GPU)", [PY, "-m", "exportspredictor.experiments", "curve"]),
    Step("Hyperparameter search on 2022 (GPU, resumable, about 3 h)", [PY, "-m", "exportspredictor.search"]),
    # Trains the fold models, scores the Gap Period, evaluates on 2023 (read here for the first
    # time) and saves a new artefact version. Fold and test-year models are cached, so a re-run resumes.
    # The default run: the Event Backtests, the constraint comparison and the gain ladder are opt-in
    # (`--every-part` for all three, about two more hours).
    Step("Train the fold and test-year models, evaluate and save the artefacts (GPU, about 2.5 h)", [PY, "-m", "exportspredictor.pipeline"]),
    # The notebook runs on real data here (it defaults to fixture mode, as the tests run it) and
    # reads the artefact the step before saved.
    Step(
        "The notebook (reads the saved artefact)",
        [PY, "-m", "jupyter", "execute", "notebooks/exports_predictor.ipynb"],
        env={"EXPORTSPREDICTOR_FIXTURE_MODE": "0"},
    ),
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
    args = parser.parse_args()
    if args.list:
        for number, step in enumerate(STEPS, start=1):
            print(f"{number}. {step.name}")
        return 0
    return run(STEPS)


if __name__ == "__main__":
    sys.exit(main())
