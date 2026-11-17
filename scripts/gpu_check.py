"""The GPU check behind the row budget (ticket #52, ADR 0004): peak VRAM and seconds per round.

Two steps, run through `scripts/wt.ps1` (or `uv run`) from the repo root:

    python scripts/gpu_check.py prepare <out>   CPU: v2 training rows on the Sector subset
    python scripts/gpu_check.py measure <out>   GPU: two short training runs, measured

`prepare` builds the v2 training rows (top-100 Exporter scope, every UK Zero Cell, other
Exporters' Zero Cells at `features.ZERO_SAMPLING_RATE`) for 2012–2021, keeps the
`training.SECTOR_SUBSET` rows and adds their features from every Flow. It also counts the
full-scope rows of 2012–2023, UK and other, which the real build's derived rate depends on.

`measure` trains a few rounds with v1's configuration on a device-built `QuantileDMatrix`
of the subset's rows fed one to four times (about 27 M to 107 M rows), each in its own
process, while `nvidia-smi` is polled for the peak memory in use while building and while
training. A line through the peaks above the idle level, against rows, gives the fixed
and per-row VRAM cost that `training.row_budget` uses.
"""

import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor import baci, features, training
from exportspredictor.paths import artefacts_dir
from exportspredictor.scope import exporter_scope

ROUNDS = 40
COUNT_YEARS = range(2012, 2024)  # the Baseline fold models' training years
SUBSET_YEARS = range(2012, 2022)  # the tuning years
COPIES = (1, 2, 3, 4)  # the subset's rows fed 1-4 times: about 27 M to 107 M rows
V1_ARTEFACT = "20261117T022550Z"


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sources, to_iso3 = features.load_sources()
    scope = exporter_scope(sources.country_year, uk=baci.UK)
    counts = {}
    for year in COUNT_YEARS:
        started = time.perf_counter()
        flows = features.load_year_flows(year, to_iso3)
        rows = features.training_rows(flows, rate=features.ZERO_SAMPLING_RATE, uk=baci.UK, exporters=scope)
        is_uk = pl.col("exporter") == baci.UK
        counts[year] = {"uk": rows.filter(is_uk).height, "non_uk": rows.filter(~is_uk).height}
        if year in SUBSET_YEARS:
            subset = rows.filter(pl.col("hs6").cast(pl.String).str.slice(0, 2).is_in(training.SECTOR_SUBSET))
            counts[year]["subset"] = subset.height
            features.build_features(subset, flows, sources).write_parquet(out / f"training_{year}.parquet")
        print(year, counts[year], f"{time.perf_counter() - started:.0f} s", flush=True)
        del flows, rows
    settings = {
        "exporter_scope": scope,
        "zero_sampling_rate": features.ZERO_SAMPLING_RATE,
        "sector_subset": list(training.SECTOR_SUBSET),
        "rows": counts,
    }
    (out / "rows.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _v1_params() -> dict:
    return json.loads((artefacts_dir() / V1_ARTEFACT / "metadata.json").read_text(encoding="utf-8"))["params"]


def _memory_used_mib() -> int:
    query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    return int(subprocess.run(query, capture_output=True, text=True, check=True).stdout.split()[0])


class _Peaks:
    """Polls the GPU's memory in use and keeps the peak of each named phase."""

    def __init__(self):
        self.idle = statistics.median(_memory_used_mib() for _ in range(5))
        self.phase, self.peaks, self._done = "start", {}, threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._done.is_set():
            used, phase = _memory_used_mib(), self.phase
            self.peaks[phase] = max(self.peaks.get(phase, 0), used)
            time.sleep(0.05)

    def stop(self) -> dict:
        time.sleep(0.5)
        self._done.set()
        self._thread.join()
        return {phase: peak - self.idle for phase, peak in self.peaks.items() if phase != "start"}


def run(out: Path, copies: int) -> None:
    """One measured training run, in its own process so that the GPU starts empty.

    The Sector subset's 2012–2021 rows are fed `copies` times, so that the matrix reaches
    the size of a full real build without a full real feature build.
    """
    years = list(SUBSET_YEARS)
    chunks = [out / f"training_{y}.parquet" for y in years] * copies
    transform = features.FeatureTransform.fit(training.scan_chunks(chunks[: len(years)]), years)
    peaks = _Peaks()
    peaks.phase = "build"
    started = time.perf_counter()
    matrix = training.training_matrix(chunks, training.preparer(uk=baci.UK, transform=transform), device="cuda")
    build_seconds = time.perf_counter() - started
    stamps = []

    class Clock(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            stamps.append(time.perf_counter())
            return False

    peaks.phase = "train"
    started = time.perf_counter()
    xgb.train(
        training.model_params(_v1_params(), device="cuda"), matrix, num_boost_round=ROUNDS, callbacks=[Clock()]
    )
    above_idle = peaks.stop()
    rounds = [b - a for a, b in zip([started, *stamps], stamps, strict=False)]
    print(json.dumps({
        "rows": matrix.num_row(),
        "features": matrix.num_col(),
        "idle_mib": peaks.idle,
        "build_peak_mib": above_idle["build"],
        "train_peak_mib": above_idle["train"],
        "peak_vram_mib": max(above_idle.values()),
        "build_seconds": round(build_seconds, 2),
        "first_round_seconds": round(rounds[0], 3),
        "seconds_per_round": round(statistics.median(rounds[1:]), 4),
    }))


def measure(out: Path) -> None:
    results = {}
    for copies in COPIES:
        child = subprocess.run(
            [sys.executable, __file__, "run", str(out), str(copies)], capture_output=True, text=True, check=False
        )
        if child.returncode:
            raise RuntimeError(child.stderr)
        results[copies] = json.loads(child.stdout.strip().splitlines()[-1])
        print(copies, results[copies], flush=True)
    rows = [r["rows"] for r in results.values()]
    peak_bytes = [r["peak_vram_mib"] * 2**20 for r in results.values()]
    per_row, fixed = np.polyfit(rows, peak_bytes, 1)  # least squares: peak = fixed + rows x per_row
    summary = {
        "runs": list(results.values()),
        "fit": {"bytes_per_row": round(float(per_row), 2), "fixed_bytes": round(float(fixed))},
        "params": _v1_params(),
        "rounds": ROUNDS,
        "xgboost_version": xgb.__version__,
    }
    print(summary["fit"], flush=True)
    (out / "gpu_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    command, out = sys.argv[1], Path(sys.argv[2])
    if command == "prepare":
        prepare(out)
    elif command == "measure":
        measure(out)
    else:
        run(out, int(sys.argv[3]))
