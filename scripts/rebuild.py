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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]


STEPS = [
    # Downloads and derived tables (#12 BACI Flows and lookups, #13 Trade Agreements and
    # Trading Blocs, #14 gravity and WDI) go here, one Step per line, in this order.
    Step("BACI Flows and lookups (download, convert to Parquet)", [PY, "-m", "exportspredictor.baci"]),
    Step("Gravity and WDI features (download CEPII Gravity, WDI, IMF WEO; build Parquet)", [PY, "-m", "exportspredictor.gravity"]),
    Step("Train, evaluate and save artefacts (notebook)", [PY, "-m", "jupyter", "execute", "notebooks/exports_predictor.ipynb"]),
]


def run(steps: list[Step]) -> int:
    """Run the steps in order from the repo root. Returns 0, or the first failing exit code."""
    for number, step in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] {step.name}", flush=True)
        code = subprocess.run(step.command, cwd=REPO).returncode
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
