"""Versioned artefact folders: what the notebook saves and the API serves.

Layout of `<root>/<version>/`:

    metadata.json      version, baseline_year, gbp_rate, n_folds, features, ... (free-form beyond those)
    models/fold_<k>.ubj
    baseline.parquet   one row per UK Product–Market pair in the Baseline year
    baseline_features.parquet  the same pairs' model inputs and fold (for Scenarios, explanations)
    history.parquet    UK Actual Exports per pair and year, non-zero only, last six years
    products.parquet   Product lookup (hs6, description, sector)
    markets.parquet    Market lookup (iso3, name, region)
    evaluation.json    accuracy on the test year, stability and later checks (optional)

Version names sort chronologically (default: a UTC timestamp), so the newest folder is
the lexically greatest one that holds a metadata file.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import xgboost as xgb

from exportspredictor import currency

METADATA_FILE = "metadata.json"
EVALUATION_FILE = "evaluation.json"
_TABLES = ("baseline", "baseline_features", "history", "products", "markets")


class ArtefactsNotFound(FileNotFoundError):
    """No artefact folder with a metadata file exists under the given root."""


@dataclass(frozen=True)
class Artefacts:
    metadata: dict
    fold_models: list[xgb.Booster]
    baseline: pl.DataFrame  # hs6, iso3, export_potential_usd, actual_exports_usd, export_gap_usd
    baseline_features: pl.DataFrame  # hs6, iso3, fold, FEATURES (transformed, as the models saw them)
    history: pl.DataFrame  # hs6, iso3, year, actual_exports_usd
    products: pl.DataFrame
    markets: pl.DataFrame
    # The evaluation report (`exportspredictor.evaluation`), JSON-ready; None if not evaluated.
    evaluation: dict | None = None

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def baseline_year(self) -> int:
        return self.metadata["baseline_year"]

    @property
    def gbp_rate(self) -> dict:
        """The USD → GBP rate for £ values: usd_per_gbp, year, source.

        Folders saved before the rate was recorded fall back to the Baseline year's rate.
        """
        return self.metadata.get("gbp_rate") or currency.gbp_rate(self.baseline_year)


def new_version() -> str:
    """A version name that sorts after every earlier one."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def save(artefacts: Artefacts, root: Path) -> Path:
    """Write the artefacts to `<root>/<version>/` and return that folder."""
    folder = Path(root) / artefacts.version
    (folder / "models").mkdir(parents=True, exist_ok=False)
    for k, model in enumerate(artefacts.fold_models):
        model.save_model(folder / "models" / f"fold_{k}.ubj")
    for name in _TABLES:
        getattr(artefacts, name).write_parquet(folder / f"{name}.parquet")
    if artefacts.evaluation is not None:
        (folder / EVALUATION_FILE).write_text(json.dumps(artefacts.evaluation, indent=2), encoding="utf-8")
    # Metadata goes last: its presence marks the folder as complete.
    (folder / METADATA_FILE).write_text(json.dumps(artefacts.metadata, indent=2), encoding="utf-8")
    return folder


def load(folder: Path) -> Artefacts:
    """Load one artefact folder."""
    folder = Path(folder)
    metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    models = []
    for k in range(metadata["n_folds"]):
        model = xgb.Booster()
        model.load_model(folder / "models" / f"fold_{k}.ubj")
        models.append(model)
    tables = {name: pl.read_parquet(folder / f"{name}.parquet") for name in _TABLES}
    return Artefacts(metadata=metadata, fold_models=models, **tables, evaluation=_read_json(folder / EVALUATION_FILE))


def latest_folder(root: Path) -> Path:
    """The newest complete artefact folder under `root`."""
    root = Path(root)
    folders = sorted(p.parent for p in root.glob(f"*/{METADATA_FILE}")) if root.is_dir() else []
    if not folders:
        raise ArtefactsNotFound(f"No artefact folder with {METADATA_FILE} under {root}")
    return folders[-1]


def load_latest(root: Path) -> Artefacts:
    """Load the newest complete artefact folder under `root`."""
    return load(latest_folder(root))


def copy_version(source: Path, root: Path, *, metadata: dict, evaluation: dict | None = None) -> Path:
    """Copy an artefact folder, file for file, to a new version `<root>/<metadata["version"]>/`.

    Models and tables are copied byte for byte, including any this code does not know
    about, so a step that adds one result does not drop another step's tables. The
    metadata is replaced and, if given, the evaluation report too. Returns the new folder.
    """
    target = Path(root) / metadata["version"]
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(METADATA_FILE, EVALUATION_FILE))
    report = evaluation if evaluation is not None else _read_json(Path(source) / EVALUATION_FILE)
    if report is not None:
        (target / EVALUATION_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Metadata goes last: its presence marks the folder as complete.
    (target / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
