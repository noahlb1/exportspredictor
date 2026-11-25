"""Freezing v1: v1's predictions saved once, as fixed tables that v2 joins to (spec #39).

v2 changes the features, so v1's saved models stop loading against the code once v2 lands.
Before that, this command reads v1's models one last time and writes, into v1's artefact
folder, a `v1_predictions/` set of Parquet tables holding every prediction the v1/v2
comparison needs:

- `test_rows`: v1's own test-year (2023) feature rows, all Exporters (every positive Flow
  plus v1's seeded weighted sample of Zero Cells, weights kept), with Export Potential from
  the test-year fold models: UK rows out-of-fold, other rows by the mean of the fold models,
  exactly as v1's evaluation scored them.
- `uk_grid`: every UK Product × Market of the test year, each pair scored by the test-year
  fold model that held it out (ADR 0001), with its `fold`.
- `lift` and `lift_held_out_exporters`: the Lift model's Export Potential for the held-out
  Exporters' pairs in 2015–2017, and the held-out Exporter list itself.
- `event_backtest`: the EU–Japan Event Backtest model's Export Potential with the Trade
  Agreement on and off, for each direction's treated pairs in the post years (2019–2021).

The naive baseline is not frozen: v2 computes it itself on the common rows.

`manifest.json` records the `v1` tag, the commit, the source model identifiers, and each
table's row count and SHA-256. Every table is built before anything is written, the files go
to `v1_predictions.part/`, and the folder is renamed only when complete, so a failed freeze
leaves nothing behind. Re-running is a no-op once a manifest exists (it is verified instead).

`uv run python -m exportspredictor.freeze` freezes the real v1 artefact on the CPU.
"""

import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from exportspredictor import baci, backtests, downloads
from exportspredictor.artefacts import METADATA_FILE
from exportspredictor.evaluation import TestModels, score_rows, test_models_dir
from exportspredictor.features import features_dir
from exportspredictor.folds import score_out_of_fold
from exportspredictor.paths import REPO_ROOT, artefacts_dir, data_dir
from exportspredictor.training import Chunk, read_chunk, read_rows

FOLDER = "v1_predictions"
MANIFEST_FILE = "manifest.json"
TABLES = {
    "test_rows": "test_rows.parquet",
    "uk_grid": "uk_grid.parquet",
    "lift": "lift.parquet",
    "lift_held_out_exporters": "lift_held_out_exporters.parquet",
    "event_backtest": "event_backtest.parquet",
}
TAG = "v1"
_ROW_KEYS = ["exporter", "importer", "hs6"]


@dataclass(frozen=True)
class V1:
    """What the freeze reads: v1's feature rows and its saved models."""

    rows: Mapping[int, Chunk]  # v1's training rows with features, per year
    test_grid: Chunk  # the test year's UK Product × Market grid with features
    test_models: TestModels
    lift_model: backtests.BacktestModel
    event_model: backtests.BacktestModel
    event: backtests.Event
    lift_years: tuple[int, ...]
    model_ids: dict  # source model identifiers recorded in the manifest


def tables(v1: V1, *, uk: str) -> dict[str, pl.DataFrame]:
    """Every frozen table, built in memory; nothing is written."""
    lift = _lift(v1)
    return {
        "test_rows": _test_rows(v1, uk),
        "uk_grid": _uk_grid(v1, uk),
        "lift": lift,
        "lift_held_out_exporters": pl.DataFrame(
            {"exporter": sorted(v1.lift_model.held_out_exporters)}, schema={"exporter": pl.String}
        ),
        "event_backtest": _event_backtest(v1),
    }


def _test_rows(v1: V1, uk: str) -> pl.DataFrame:
    models = v1.test_models
    rows = read_chunk(v1.rows[models.test_year])
    scored = score_rows(rows, models.models, models.transform, uk)
    return _keyed(scored).select(*_ROW_KEYS, "year", "value_usd", "weight", "export_potential_usd").sort(_ROW_KEYS)


def _uk_grid(v1: V1, uk: str) -> pl.DataFrame:
    models = v1.test_models
    grid = read_chunk(v1.test_grid)
    scored = score_out_of_fold(models.transform.apply(grid), list(models.models), uk=uk)
    table = scored.select(
        pl.col("hs6").cast(pl.String),
        iso3=pl.col("importer").cast(pl.String),
        year="year",
        actual_exports_usd="value_usd",
        fold="fold",
        export_potential_usd="export_potential_usd",
    ).sort("hs6", "iso3")
    if table.height != grid.height or table.select("hs6", "iso3").is_duplicated().any():
        raise ValueError("The UK grid must hold exactly one row per UK Product–Market pair of the test year")
    return table


def _lift(v1: V1) -> pl.DataFrame:
    model = v1.lift_model
    is_held = pl.col("exporter").cast(pl.String).is_in(model.held_out_exporters)
    frames = []
    for year in v1.lift_years:
        rows = read_rows(v1.rows[year], is_held)
        frames.append(
            _keyed(rows.with_columns(export_potential_usd=pl.Series(model.predict(rows), dtype=pl.Float64)))
            .select(*_ROW_KEYS, "year", "value_usd", "weight", "export_potential_usd")
        )
    return pl.concat(frames).sort("year", *_ROW_KEYS)


def _event_backtest(v1: V1) -> pl.DataFrame:
    event, model = v1.event, v1.event_model
    if max(model.training_years) >= event.event_year:
        raise backtests.LeakageError(f"The {event.name} model trained on {model.training_years}")
    frames = []
    for d in event.directions:
        condition = pl.col("exporter").cast(pl.String).is_in(list(d.exporters)) & pl.col("importer").cast(
            pl.String
        ).is_in(list(d.importers))
        for year in event.post_years:
            rows = read_rows(v1.rows[year], condition)
            flag = rows.schema["trade_agreement"]
            on = model.predict(rows.with_columns(trade_agreement=pl.lit(1, flag)))
            off = model.predict(rows.with_columns(trade_agreement=pl.lit(0, flag)))
            frames.append(
                _keyed(rows)
                .select(*_ROW_KEYS, "year", "value_usd", "weight")
                .with_columns(
                    export_potential_on_usd=pl.Series(on, dtype=pl.Float64),
                    export_potential_off_usd=pl.Series(off, dtype=pl.Float64),
                )
                .select(pl.lit(d.label).alias("direction"), pl.all())
            )
    return pl.concat(frames).sort("direction", "year", *_ROW_KEYS)


def _keyed(frame: pl.DataFrame) -> pl.DataFrame:
    """Keys as plain strings, so the tables do not depend on v1's category lists."""
    return frame.with_columns(pl.col(_ROW_KEYS).cast(pl.String), pl.col("year").cast(pl.Int16))


# --- Writing and checking -------------------------------------------------------------------------


def freeze(artefact_folder: Path, v1: V1, *, uk: str, provenance: dict) -> Path:
    """Write `v1_predictions/` into `artefact_folder`, or verify it if it is already there.

    `provenance` holds the `tag` and `commit`. Returns the folder.
    """
    target = Path(artefact_folder) / FOLDER
    if (target / MANIFEST_FILE).exists():
        verify(target)
        return target
    started = time.perf_counter()
    built = tables(v1, uk=uk)
    fields = {
        **provenance,
        "source_models": v1.model_ids,
        "test_year": v1.test_models.test_year,
        "lift_years": list(v1.lift_years),
        "event": v1.event.name,
        "event_post_years": list(v1.event.post_years),
    }
    return write(artefact_folder, built, fields, started=started)


def write(artefact_folder: Path, built: Mapping[str, pl.DataFrame], fields: dict, *, started: float | None = None) -> Path:
    """Write the frozen tables `built` (one per `TABLES` entry) and their manifest into `artefact_folder`.

    `fields` (provenance, source models, test year, Lift years, event) head the manifest.
    The files go to a staging folder renamed only when complete, so a failed write leaves
    nothing. The fixture's synthetic frozen v1 is written the same way. Returns the folder.
    """
    started = time.perf_counter() if started is None else started
    target = Path(artefact_folder) / FOLDER
    staging = Path(artefact_folder) / f"{FOLDER}.part"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        entries = {}
        for name, frame in built.items():
            path = downloads.write_parquet(frame, staging / TABLES[name])
            entries[name] = {"file": TABLES[name], "rows": frame.height, "sha256": downloads.sha256(path)}
        manifest = {
            **fields,
            "tables": entries,
            "frozen_at": datetime.now(UTC).isoformat(),
            "seconds": round(time.perf_counter() - started, 2),
        }
        with downloads.writing(staging / MANIFEST_FILE) as part:
            part.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.rmtree(target, ignore_errors=True)  # a folder without a manifest is incomplete
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def verify(folder: Path) -> dict:
    """The manifest of a frozen folder, after checking each file's row count and checksum."""
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    for name, entry in manifest["tables"].items():
        path = folder / entry["file"]
        if downloads.sha256(path) != entry["sha256"] or pl.scan_parquet(path).select(pl.len()).collect().item() != entry["rows"]:
            raise ValueError(f"The frozen table {name} does not match its manifest")
    return manifest


def lift_held_out_exporters(folder: Path) -> list[str]:
    """v1's held-out Lift Exporters, read from a frozen folder after checking that table against the manifest.

    v2's Lift model holds out the same Exporters (within v2's scope, `backtests.lift_exporters`).
    """
    folder = Path(folder)
    entry = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))["tables"]["lift_held_out_exporters"]
    path = folder / entry["file"]
    table = pl.read_parquet(path)
    if downloads.sha256(path) != entry["sha256"] or table.height != entry["rows"]:
        raise ValueError("The frozen table lift_held_out_exporters does not match its manifest")
    return sorted(table["exporter"].cast(pl.String).to_list())


def v1_folder(artefacts_root: Path | None = None) -> Path:
    """The frozen v1 tables of the real v1 artefact."""
    return Path(artefacts_root or artefacts_dir()) / V1_IDS["artefact"] / FOLDER


# --- The real v1 --------------------------------------------------------------------------------

# v1's run (spec #39): its artefact, its cached 2023 test-year fold models and its cached
# Lift and EU–Japan Event Backtest models, by folder name.
V1_IDS = {
    "artefact": "20261117T022550Z",
    "test_models": "2023_20e7c6bf9547",
    "lift": "lift_fbc42f09c602",
    "event_backtest": "eu_japan_034ce6fa2915",
}
V1_LIFT_YEARS = (2015, 2016, 2017)


def load_v1(
    root: Path | None = None,
    *,
    ids: dict = V1_IDS,
    event: backtests.Event = backtests.EU_JAPAN,
    lift_years: tuple[int, ...] = V1_LIFT_YEARS,
) -> V1:
    """v1's feature files and cached models under the data root `root`, by the folder names in `ids`."""
    folder = features_dir(root)
    years = json.loads((folder / "features.json").read_text(encoding="utf-8"))["years"]
    test_models = TestModels.load(folder.parent / "test_models" / ids["test_models"])
    lift_model = backtests.BacktestModel.load(backtests.cache_dir(root) / ids["lift"])
    event_model = backtests.BacktestModel.load(backtests.cache_dir(root) / ids["event_backtest"])
    if event_model.name != event.name or event_model.training_years != list(event.training_years):
        raise ValueError(f"{ids['event_backtest']} is not the {event.name} Event Backtest model")
    if not lift_model.held_out_exporters or max(lift_years) > max(lift_model.training_years):
        raise ValueError(f"{ids['lift']} is not a Lift model covering {list(lift_years)}")
    return V1(
        rows={y: folder / f"training_{y}.parquet" for y in years},
        test_grid=folder / f"uk_grid_{test_models.test_year}.parquet",
        test_models=test_models,
        lift_model=lift_model,
        event_model=event_model,
        event=event,
        lift_years=tuple(lift_years),
        model_ids=dict(ids),
    )


def git_provenance(repo: Path = REPO_ROOT) -> dict:
    """The checked-out commit, whether the `v1` tag points at it, and whether tracked files differ from it."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    tags = git("tag", "--points-at", "HEAD").split()
    return {
        "tag": TAG if TAG in tags else None,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def main() -> None:
    """Freeze the real v1 artefact (CPU; v1's models are only read), from a clean checkout of the `v1` tag."""
    provenance = git_provenance()
    if provenance["tag"] != TAG or provenance["dirty"]:
        raise SystemExit(f"Run the freeze from a clean checkout of the `{TAG}` tag, not {provenance}")
    root = data_dir()
    artefact = artefacts_dir() / V1_IDS["artefact"]
    metadata = json.loads((artefact / METADATA_FILE).read_text(encoding="utf-8"))
    if test_models_dir(root, metadata).name != V1_IDS["test_models"]:
        raise SystemExit(f"{V1_IDS['test_models']} is not the test-year model cache of {V1_IDS['artefact']}")
    started = time.perf_counter()
    folder = freeze(artefact, load_v1(root), uk=baci.UK, provenance=provenance)
    manifest = verify(folder)
    rows = {name: entry["rows"] for name, entry in manifest["tables"].items()}
    print(json.dumps({"folder": str(folder), "rows": rows, "seconds": round(time.perf_counter() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
