"""Versioned artefact folders: what the notebook saves and the API serves.

Layout of `<root>/<version>/`:

    metadata.json      artefact_format, version, baseline_year, gap_years, n_folds, features,
                       model_form (`prediction.ModelForm`), ranking_experiments (the selection
                       measure, the noise floor and the experiment results that chose the form),
                       potential_share_of_actual_exports (summed Export Potential over summed Actual
                       Exports, Gap Period means; versions without it compute it from the Baseline), ...
                       (free-form beyond those)
    models/fold_<k>.ubj
    baseline.parquet   one row per UK Product–Market pair: Gap Period means of Export Potential,
                       Actual Exports and Export Gap; the Baseline year's Export Potential;
                       the Blocked Opportunity value and its Tariff part (ADR 0011); its Caveat Tier columns, such as the
                       Flagged Product and Re-export Hub flags (`caveats.TIERS`, ADR 0005); its Potential Range and Uncertain Opportunity flag (ADR 0007);
                       whether its Baseline-year Applied Tariff is carried forward from 2021 (ADR 0006)
    baseline_features.parquet  the same pairs' Baseline-year model inputs, fold and Naive Baseline
                       (`naive_usd`), for Scenarios and explanations
    history.parquet    per pair and year, last six years: UK Actual Exports (non-zero only before
                       the Gap Period) and, in Gap Period years, Export Potential and Export Gap;
                       every year marks a carried-forward Applied Tariff (tariff_imputed)
    products.parquet   Product lookup (hs6, description, sector)
    markets.parquet    Market lookup (iso3, name, region)
    excluded_products.parquet  Excluded Products (hs6, description, sector, reason), ADR 0005
    potential_range_bands.parquet  the Potential Range's band offsets (ADR 0007, `conformal.fit_bands`)
    market_reach.parquet  the Baseline year's Market Reach table, one row per fold and Market (ADR 0010,
                       `market_reach.TABLE_COLUMNS`); the served Model Form applies it
    evaluation.json    accuracy on the test year, stability and later checks (optional)
    ppml_benchmark.json  the PPML benchmark (`ppml.benchmark`): the fit's effect and interval, the
                       published range and the model's Trade Agreement association on the served
                       Model Form (optional: a version without it loads and serves)

`artefact_format` is 4: format 3 had no Market Reach (spec #113, ADR 0010), and format 2
served the model alone and recorded no Model Form (spec #65 made the served form a choice).
`load` refuses any other format, including v1 folders (which carry none), and a folder missing
any table above, so the API never serves a bundle it cannot read. Only `pipeline.rereach`, which
turns a format-3 folder into a format-4 one from its own fold models, reads format 3
(`load_format_3`).

Version names sort chronologically (default: a UTC timestamp), so the newest folder is
the lexically greatest one that holds a metadata file.
"""

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import xgboost as xgb

from exportspredictor.gap_period import potential_share_of_actual_exports
from exportspredictor.prediction import ModelForm

METADATA_FILE = "metadata.json"
EVALUATION_FILE = "evaluation.json"
PPML_BENCHMARK_FILE = "ppml_benchmark.json"
# The optional JSON results, by `Artefacts` field: each is None where its file is absent.
_OPTIONAL_JSON = {"evaluation": EVALUATION_FILE, "ppml_benchmark": PPML_BENCHMARK_FILE}
ARTEFACT_FORMAT = 4
_TABLES = (
    "baseline",
    "baseline_features",
    "history",
    "products",
    "markets",
    "excluded_products",
    "potential_range_bands",
    "market_reach",
)
_FORMAT_3_TABLES = tuple(name for name in _TABLES if name != "market_reach")
# Baseline columns renamed after the first format-2 folders were written (code review of
# spec #39): `load` reads an older folder under the current names.
_RENAMED_BASELINE_COLUMNS = {
    "blocked_effect_usd": "blocked_opportunity_value_usd",
    "uncertain": "uncertain_opportunity",
}
# The Baseline's Blocked Opportunity columns (`scenarios.blocked_opportunity_values`): the value and its Tariff part
# (ADR 0011). A version saved before the Tariff part loads only for `scenarios.rebuild_blocked_values`.
BLOCKED_VALUE_COLUMNS = ("blocked_opportunity_value_usd", "tariff_part_usd")


class ArtefactsNotFound(FileNotFoundError):
    """No artefact folder with a metadata file exists under the given root."""


class UnreadableArtefacts(ValueError):
    """An artefact folder of another format, or one missing a table this code reads."""


@dataclass(frozen=True)
class Artefacts:
    metadata: dict
    fold_models: list[xgb.Booster]
    # hs6, iso3, export_potential_usd, actual_exports_usd, export_gap_usd (Gap Period means),
    # baseline_export_potential_usd (the Baseline year's), each Caveat Tier's columns (`caveats.TIERS`), blocked_opportunity_value_usd,
    # tariff_part_usd, potential_range_low_usd, potential_range_high_usd, uncertain_opportunity, tariff_imputed
    baseline: pl.DataFrame
    # hs6, iso3, fold, FEATURES (transformed, as the models saw them), mfn_tariff (percent, the 2021 MFN rate),
    # naive_usd (the Naive Baseline a Blend takes)
    baseline_features: pl.DataFrame
    history: pl.DataFrame  # hs6, iso3, year, actual_exports_usd, export_potential_usd, export_gap_usd, tariff_imputed
    products: pl.DataFrame  # every Product, Excluded Products included (search marks them)
    markets: pl.DataFrame
    excluded_products: pl.DataFrame  # hs6, description, sector, reason (`exclusions.excluded_products`)
    # band, min_export_potential_usd, rows, low_offset_log, high_offset_log (`conformal.fit_bands`;
    # versions before ADR 0007's log-scale amendment hold low_offset_usd, high_offset_usd)
    potential_range_bands: pl.DataFrame
    # fold, importer, products_sent, products_imported, products_expected, offset: the Baseline year's Market Reach
    # (`market_reach.reach_table`, ADR 0010); None only in a format-3 folder read by `load_format_3`.
    market_reach: pl.DataFrame | None = None
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
        return ModelForm.from_dict(self.metadata["model_form"], self.market_reach)

    @property
    def baseline_year(self) -> int:
        return self.metadata["baseline_year"]

    @property
    def potential_share_of_actual_exports(self) -> float | None:
        """Summed Export Potential as a share of summed Actual Exports over the Baseline's pairs (Gap Period means):
        the metadata's `potential_share_of_actual_exports`, or, for a version saved before it was stored, the same
        figure from the Baseline (`gap_period.potential_share_of_actual_exports`)."""
        if "potential_share_of_actual_exports" in self.metadata:
            return self.metadata["potential_share_of_actual_exports"]
        return potential_share_of_actual_exports(self.baseline)

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
        table = getattr(artefacts, name)
        if table is None:
            raise ValueError(f"Artefact format {ARTEFACT_FORMAT} needs the table {name}")
        table.write_parquet(folder / f"{name}.parquet")
    _write_optional_json(folder, {field: getattr(artefacts, field) for field in _OPTIONAL_JSON})
    # Metadata goes last: its presence marks the folder as complete.
    (folder / METADATA_FILE).write_text(json.dumps(artefacts.metadata, indent=2), encoding="utf-8")
    return folder


def load(folder: Path, *, tiers: Sequence | None = None, blocked_values: bool = True) -> Artefacts:
    """Load one artefact folder; `UnreadableArtefacts` if it is not a complete format-4 folder, or if its Baseline
    lacks a column of one of `tiers` (default: every declared Caveat Tier, `caveats.TIERS`) or, with
    `blocked_values`, a Blocked Opportunity column (`BLOCKED_VALUE_COLUMNS`: the value and its Tariff part, ADR 0011).
    `recaveat`, which marks the Baseline again, reads it with `tiers=()`; `scenarios.rebuild_blocked_values`, which
    computes the Blocked Opportunity columns again, with `blocked_values=False`."""
    bundle = _load(folder, ARTEFACT_FORMAT, _TABLES)
    lacking = [column for column in BLOCKED_VALUE_COLUMNS if column not in bundle.baseline.columns]
    if blocked_values and lacking:
        raise UnreadableArtefacts(
            f"{folder}'s Baseline lacks {', '.join(lacking)} (a version saved before the Tariff part, ADR 0011); "
            "write a version with them from its saved fold models with `python -m exportspredictor.scenarios`"
        )
    if tiers is None:
        # Imported here: the caveat tables are not needed to load a bundle.
        from exportspredictor.caveats import TIERS

        tiers = TIERS
    missing = [column for tier in tiers for column in tier.columns if column not in bundle.baseline.columns]
    if missing:
        raise UnreadableArtefacts(
            f"{folder}'s Baseline lacks the Caveat Tier column(s) {', '.join(missing)}; mark it with the declared "
            "Caveat Tiers with `python -m exportspredictor.pipeline recaveat`"
        )
    return bundle


def load_format_3(folder: Path) -> Artefacts:
    """Load a complete format-3 folder (no Market Reach), only for `pipeline.rereach` to upgrade it."""
    return _load(folder, 3, _FORMAT_3_TABLES)


def _load(folder: Path, artefact_format: int, names: tuple[str, ...]) -> Artefacts:
    folder = Path(folder)
    metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    found = metadata.get("artefact_format")
    if found != artefact_format:
        named = "no artefact_format" if found is None else f"artefact format {found}"
        raise UnreadableArtefacts(
            f"{folder} has {named}; this code reads only artefact format {artefact_format}, which records the "
            "model form it serves and its Market Reach. Rebuild it with `python -m exportspredictor.pipeline`, or "
            "upgrade a format-3 folder with `python -m exportspredictor.pipeline rereach`"
        )
    missing = [name for name in names if not (folder / f"{name}.parquet").exists()]
    if missing:
        raise UnreadableArtefacts(f"{folder} is missing the table(s) {', '.join(missing)}")
    models = []
    for k in range(metadata["n_folds"]):
        model = xgb.Booster()
        model.load_model(folder / "models" / f"fold_{k}.ubj")
        models.append(model)
    tables = {name: pl.read_parquet(folder / f"{name}.parquet") for name in names}
    tables["baseline"] = tables["baseline"].rename(_RENAMED_BASELINE_COLUMNS, strict=False)
    return Artefacts(
        metadata=metadata,
        fold_models=models,
        **tables,
        **{field: _read_json(folder / name) for field, name in _OPTIONAL_JSON.items()},
    )


def latest_folder(root: Path) -> Path:
    """The newest complete artefact folder under `root`."""
    root = Path(root)
    folders = sorted(p.parent for p in root.glob(f"*/{METADATA_FILE}")) if root.is_dir() else []
    if not folders:
        raise ArtefactsNotFound(f"No artefact folder with {METADATA_FILE} under {root}")
    return folders[-1]


def load_latest(root: Path, *, tiers: Sequence | None = None, blocked_values: bool = True) -> Artefacts:
    """Load the newest complete artefact folder under `root` (`load`, with its `tiers` and `blocked_values`)."""
    return load(latest_folder(root), tiers=tiers, blocked_values=blocked_values)


def copy_version(
    source: Path, root: Path, *, metadata: dict, evaluation: dict | None = None, ppml_benchmark: dict | None = None
) -> Path:
    """Copy an artefact folder, file for file, to a new version `<root>/<metadata["version"]>/`.

    Models and tables are copied byte for byte, including any this code does not know
    about, so a step that adds one result does not drop another step's tables. The
    metadata is replaced and, if given, the evaluation report and the PPML benchmark too;
    an optional result not given is the source's. Returns the new folder.
    """
    target = Path(root) / metadata["version"]
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(METADATA_FILE, *_OPTIONAL_JSON.values()))
    given = {"evaluation": evaluation, "ppml_benchmark": ppml_benchmark}
    _write_optional_json(target, {
        field: given[field] if given[field] is not None else _read_json(Path(source) / name)
        for field, name in _OPTIONAL_JSON.items()
    })
    # Metadata goes last: its presence marks the folder as complete.
    (target / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def _write_optional_json(folder: Path, results: dict) -> None:
    """Write each optional JSON result that is not None (`_OPTIONAL_JSON`)."""
    for field, value in results.items():
        if value is not None:
            (folder / _OPTIONAL_JSON[field]).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
