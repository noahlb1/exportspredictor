"""Versioned artefact folders: what the notebook saves and the API serves.

Layout of `<root>/<version>/`:

    metadata.json      artefact_format, version, baseline_year, gap_years, n_folds, features,
                       model_form (`prediction.ModelForm`), ranking_experiments (the selection
                       measure, the noise floor and the experiment results that chose the form), ...
                       (free-form beyond those)
    models/fold_<k>.ubj
    baseline.parquet   one row per UK Product–Market pair: Gap Period means of Export Potential,
                       Actual Exports and Export Gap; the Baseline year's Export Potential;
                       the Blocked Opportunity value; its Flagged Product and Re-export Hub
                       flags (ADR 0005); its Potential Range and Uncertain Opportunity flag (ADR 0007);
                       whether its Baseline-year Applied Tariff is carried forward from 2021 (ADR 0006)
    baseline_features.parquet  the same pairs' Baseline-year model inputs, fold and naive baseline
                       (`naive_usd`), for Scenarios and explanations
    history.parquet    per pair and year, last six years: UK Actual Exports (non-zero only before
                       the Gap Period) and, in Gap Period years, Export Potential and Export Gap;
                       every year marks a carried-forward Applied Tariff (tariff_imputed)
    products.parquet   Product lookup (hs6, description, sector)
    markets.parquet    Market lookup (iso3, name, region)
    excluded_products.parquet  Excluded Products (hs6, description, sector, reason), ADR 0005
    potential_range_bands.parquet  the Potential Range's band offsets (ADR 0007, `conformal.fit_bands`)
    evaluation.json    accuracy on the test year, stability and later checks (optional)
    ppml_benchmark.json  the PPML benchmark (`ppml.benchmark`): the fit's effect and interval, the
                       published range and the model's Trade Agreement association on the served
                       model form (optional: a version without it loads and serves)

`artefact_format` is 3: format 2 served the model alone and recorded no model form (spec
#65 made the served form a choice). `load` refuses any other format, including v1 folders
(which carry none), and a folder missing any table above, so the API never serves a bundle
it cannot read.

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

from exportspredictor.prediction import ModelForm

METADATA_FILE = "metadata.json"
EVALUATION_FILE = "evaluation.json"
PPML_BENCHMARK_FILE = "ppml_benchmark.json"
ARTEFACT_FORMAT = 3
_TABLES = (
    "baseline",
    "baseline_features",
    "history",
    "products",
    "markets",
    "excluded_products",
    "potential_range_bands",
)
# Baseline columns renamed after the first format-2 folders were written (code review of
# spec #39): `load` reads an older folder under the current names.
_RENAMED_BASELINE_COLUMNS = {
    "blocked_effect_usd": "blocked_opportunity_value_usd",
    "uncertain": "uncertain_opportunity",
}


class ArtefactsNotFound(FileNotFoundError):
    """No artefact folder with a metadata file exists under the given root."""


class UnreadableArtefacts(ValueError):
    """An artefact folder of another format, or one missing a table this code reads."""


@dataclass(frozen=True)
class Artefacts:
    metadata: dict
    fold_models: list[xgb.Booster]
    # hs6, iso3, export_potential_usd, actual_exports_usd, export_gap_usd (Gap Period means),
    # baseline_export_potential_usd (the Baseline year's), flagged_product, reexport_hub, blocked_opportunity_value_usd,
    # potential_range_low_usd, potential_range_high_usd, uncertain_opportunity, tariff_imputed
    baseline: pl.DataFrame
    # hs6, iso3, fold, FEATURES (transformed, as the models saw them), mfn_tariff (percent, the 2021 MFN rate),
    # naive_usd (the naive baseline a blend takes)
    baseline_features: pl.DataFrame
    history: pl.DataFrame  # hs6, iso3, year, actual_exports_usd, export_potential_usd, export_gap_usd, tariff_imputed
    products: pl.DataFrame  # every Product, Excluded Products included (search marks them)
    markets: pl.DataFrame
    excluded_products: pl.DataFrame  # hs6, description, sector, reason (`exclusions.excluded_products`)
    # band, min_export_potential_usd, rows, low_offset_usd, high_offset_usd (`conformal.fit_bands`)
    potential_range_bands: pl.DataFrame
    # The evaluation report (`exportspredictor.evaluation`), JSON-ready; None if not evaluated.
    evaluation: dict | None = None
    # The PPML benchmark (`ppml.benchmark`), JSON-ready; None where it was not run for this version.
    ppml_benchmark: dict | None = None

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def model_form(self) -> ModelForm:
        """How Export Potential is made from the fold models: pass it to every `prediction` call."""
        return ModelForm.from_dict(self.metadata["model_form"])

    @property
    def baseline_year(self) -> int:
        return self.metadata["baseline_year"]

    @property
    def gap_years(self) -> tuple[int, int]:
        """The first and last year of the Gap Period the Export Gaps average over."""
        first, last = self.metadata["gap_years"]
        return first, last


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
    if artefacts.ppml_benchmark is not None:
        (folder / PPML_BENCHMARK_FILE).write_text(json.dumps(artefacts.ppml_benchmark, indent=2), encoding="utf-8")
    # Metadata goes last: its presence marks the folder as complete.
    (folder / METADATA_FILE).write_text(json.dumps(artefacts.metadata, indent=2), encoding="utf-8")
    return folder


def load(folder: Path) -> Artefacts:
    """Load one artefact folder; `UnreadableArtefacts` if it is not a complete format-3 folder."""
    folder = Path(folder)
    metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    found = metadata.get("artefact_format")
    if found != ARTEFACT_FORMAT:
        named = "no artefact_format" if found is None else f"artefact format {found}"
        raise UnreadableArtefacts(
            f"{folder} has {named}; this code reads only artefact format {ARTEFACT_FORMAT}, which records the "
            "model form it serves. Rebuild it with `python -m exportspredictor.pipeline`"
        )
    missing = [name for name in _TABLES if not (folder / f"{name}.parquet").exists()]
    if missing:
        raise UnreadableArtefacts(f"{folder} is missing the table(s) {', '.join(missing)}")
    models = []
    for k in range(metadata["n_folds"]):
        model = xgb.Booster()
        model.load_model(folder / "models" / f"fold_{k}.ubj")
        models.append(model)
    tables = {name: pl.read_parquet(folder / f"{name}.parquet") for name in _TABLES}
    tables["baseline"] = tables["baseline"].rename(_RENAMED_BASELINE_COLUMNS, strict=False)
    return Artefacts(
        metadata=metadata,
        fold_models=models,
        **tables,
        evaluation=_read_json(folder / EVALUATION_FILE),
        ppml_benchmark=_read_json(folder / PPML_BENCHMARK_FILE),
    )


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


def copy_version(
    source: Path, root: Path, *, metadata: dict, evaluation: dict | None = None, ppml_benchmark: dict | None = None
) -> Path:
    """Copy an artefact folder, file for file, to a new version `<root>/<metadata["version"]>/`.

    Models and tables are copied byte for byte, including any this code does not know
    about, so a step that adds one result does not drop another step's tables. The
    metadata is replaced and, if given, the evaluation report and the PPML benchmark too.
    Returns the new folder.
    """
    target = Path(root) / metadata["version"]
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(METADATA_FILE, EVALUATION_FILE))
    report = evaluation if evaluation is not None else _read_json(Path(source) / EVALUATION_FILE)
    if report is not None:
        (target / EVALUATION_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if ppml_benchmark is not None:
        (target / PPML_BENCHMARK_FILE).write_text(json.dumps(ppml_benchmark, indent=2), encoding="utf-8")
    # Metadata goes last: its presence marks the folder as complete.
    (target / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
