"""Versioned artefact folders: what the notebook saves and the API serves.

Layout of `<root>/<version>/`:

    metadata.json      version, baseline_year, n_folds, features, ... (free-form beyond those)
    models/fold_<k>.ubj
    baseline.parquet   one row per UK Product–Market pair in the Baseline year
    baseline_features.parquet  the same pairs' model inputs and fold (for Scenarios, explanations)
    history.parquet    UK Actual Exports per pair and year, non-zero only, last six years
    products.parquet   Product lookup (hs6, description, sector)
    markets.parquet    Market lookup (iso3, name, region)

Version names sort chronologically (default: a UTC timestamp), so the newest folder is
the lexically greatest one that holds a metadata file.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import xgboost as xgb

METADATA_FILE = "metadata.json"
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

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def baseline_year(self) -> int:
        return self.metadata["baseline_year"]


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
    return Artefacts(metadata=metadata, fold_models=models, **tables)


def load_latest(root: Path) -> Artefacts:
    """Load the newest complete artefact folder under `root`."""
    root = Path(root)
    folders = sorted(p.parent for p in root.glob(f"*/{METADATA_FILE}")) if root.is_dir() else []
    if not folders:
        raise ArtefactsNotFound(f"No artefact folder with {METADATA_FILE} under {root}")
    return load(folders[-1])
