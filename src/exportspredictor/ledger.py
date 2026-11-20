"""The test-year ledger: every read of the test year, and the guard that allows one per configuration (ADR 0009).

The test year (2023) is read once per frozen version. Every choice is made on the validation year
(2022) and fixed before the test year is opened, and no test-year result feeds back into that
version. A version is identified by its **configuration fingerprint** (`fingerprint`): a hash of
the metadata that decides what the test-year models are and how they are scored, so rebuilding
the same configuration under a new version name is still the same version.

The ledger (`LEDGER_PATH`, a CSV in the repo) has one row per read: date, version, fingerprint,
what was read, the command, whether the read was an override, the record it comes from, its
`kind` and its `status`. Its first three rows are the reads made before the ledger existed (v1,
v2 and the Blend), taken from their records. A ledger written before `kind` and `status` existed
reads as completed evaluations, and gains the two columns the next time a row is written.

Every command that reads the test year calls `record_read` first, which refuses a second read of
a fingerprint already in the ledger unless the `Ledger` says `override`, and logs the override like
any other read. The row is written as `started`; the command marks it `completed` (`complete_read`)
once the read's figures are saved. A run that crashed before saving anything left its row
`started`, so a re-run of the same fingerprint takes that row over and completes it, with no
override and no second row.

Two kinds of read exist (`kind`): an `evaluation` of one configuration, and a `re-score`
(`record_rescore`): earlier versions' saved test-year scores measured again under a new measure.
A re-score trains, predicts and selects nothing. Its fingerprint (`rescore_fingerprint`) hashes the
measure and the fingerprints of the versions re-scored, so it is recorded once and refused a
second time like any read. The notebook shows the ledger through `read_ledger`, which reads only
this file.
"""

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from exportspredictor.paths import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "docs" / "test-year-ledger.csv"
COLUMNS = ("date", "version", "fingerprint", "read", "command", "override", "record", "kind", "status")
EVALUATION, RESCORE = "evaluation", "re-score"  # the kinds of read
STARTED, COMPLETED = "started", "completed"  # a read's status
# What a ledger written before `kind` and `status` existed holds in them: completed evaluations.
_ADDED_COLUMNS = {"kind": EVALUATION, "status": COMPLETED}
# The command-line switch that allows a second read of a configuration already in the ledger.
OVERRIDE_FLAG = "--read-test-year-again"
# The metadata that decides the test-year models and how they are scored. A new configuration
# setting (a feature, a model form, an offset) must be added here, or two configurations that
# differ only in it share a fingerprint and the second is refused (the safe way to be wrong).
FINGERPRINT_KEYS = (
    "source",
    "test_year",
    "features",
    "monotone_constraints",
    "model_form",
    "params",
    "rounds",
    "n_folds",
    "non_uk_sampling_rate",
    "exporter_scope",
    "zero_sampling_rate",
    "uk_zero_sampling_rate",
    "zero_sampling_seed",
    "feature_set",
    "market_reach",  # Market Reach's form, β, pseudo-count and lag (`market_reach.settings`, ADR 0010)
)
# Keys added after versions were read: hashed only when set, so an earlier version keeps its fingerprint.
LATER_KEYS = ("feature_set", "market_reach")


class TestYearAlreadyRead(RuntimeError):
    """The ledger already holds a read of this configuration, and no override was given."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class Ledger:
    """Where a command records its read of the test year, as which command, and whether it overrides a refusal."""

    path: Path = LEDGER_PATH
    command: str = "python -m exportspredictor.pipeline"
    override: bool = False

    @classmethod
    def for_command(cls, command: str, argv: Sequence[str], path: Path = LEDGER_PATH) -> "Ledger":
        """The repo's ledger for a command line: `OVERRIDE_FLAG` among `argv` allows a second read."""
        return cls(path=path, command=" ".join([command, *argv]), override=OVERRIDE_FLAG in argv)


def fingerprint(metadata: Mapping) -> str:
    """A short hash of the configuration in `metadata` (`FINGERPRINT_KEYS`); version names and timings are left out."""
    seed = (metadata.get("model_params") or {}).get("seed")
    config = {
        key: metadata.get(key) for key in FINGERPRINT_KEYS if key not in LATER_KEYS or metadata.get(key)
    } | {"seed": seed}
    # The Model Form's fitted Market Reach (its reference year and rescale constant) is computed from the data and
    # the models, so a rebuild of the same configuration could differ in the last digits; its settings are hashed
    # through `market_reach` instead.
    if isinstance(config.get("model_form"), Mapping):
        config["model_form"] = {k: v for k, v in config["model_form"].items() if k != "market_reach"}
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def rescore_fingerprint(measure: str, fingerprints: Iterable[str]) -> str:
    """The fingerprint of re-scoring the saved test-year scores of the versions with `fingerprints` under `measure`."""
    config = {"kind": RESCORE, "measure": measure, "versions": sorted(set(fingerprints))}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def read_ledger(path: Path = LEDGER_PATH) -> pl.DataFrame:
    """The ledger's rows, oldest first; empty (with its columns) if the file does not exist yet."""
    schema = {column: pl.String for column in COLUMNS} | {"override": pl.Boolean}
    rows = [{**row, "override": row["override"].strip().lower() == "true"} for row in _rows(Path(path))]
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)


def record_read(ledger: Ledger, metadata: Mapping, *, read: str, record: str = "") -> dict:
    """Record a read of the test year of the configuration in `metadata`, before anything reads it.

    Raises `TestYearAlreadyRead` if the ledger already holds a completed read of this fingerprint
    and `ledger.override` is false. An override is recorded like any other read, with `override`
    true. A started read of the fingerprint (a run that crashed before it saved anything) is taken
    over: its row gets this run's date, version, read and command, and no row is added. The row
    stays `started` until `complete_read`. Returns the row.
    """
    return _record(
        ledger, fingerprint(metadata), kind=EVALUATION, version=str(metadata.get("version", "")), read=read,
        record=record, refused=f"The test year {metadata.get('test_year')} was already read for configuration",
    )


def record_rescore(
    ledger: Ledger, *, measure: str, fingerprints: Iterable[str], version: str, read: str = "", record: str = ""
) -> dict:
    """Record re-scoring earlier versions' saved test-year scores under `measure`, before any is read.

    `fingerprints` are the configuration fingerprints of the versions whose saved scores are
    re-scored (their own reads are already in the ledger); `version` is the version whose report
    holds the result. Nothing is trained, predicted or selected, so it is its own kind of read
    (`RESCORE`), started and completed like any other and refused a second time unless
    `ledger.override`. Returns the row.
    """
    keys = sorted(set(fingerprints))
    return _record(
        ledger, rescore_fingerprint(measure, keys), kind=RESCORE, version=version,
        read=read or f"re-scoring the saved test-year scores of {', '.join(keys)} under the {measure} measure",
        record=record, refused="The earlier versions' saved test-year scores were already re-scored under this measure as",
    )


def complete_read(ledger: Ledger, key: str) -> dict:
    """Mark the started read of fingerprint `key` completed, in place, once its figures are saved. Returns the row.

    Raises ValueError if the ledger holds no started read of `key`.
    """
    path = Path(ledger.path)
    rows = _rows(path)
    started = [row for row in rows if row["fingerprint"] == key and row["status"] == STARTED]
    if not started:
        raise ValueError(f"The ledger {path} holds no started read of {key} to complete")
    started[-1]["status"] = COMPLETED
    _write(path, rows)
    return dict(started[-1])


def _record(ledger: Ledger, key: str, *, kind: str, version: str, read: str, record: str, refused: str) -> dict:
    path = Path(ledger.path)
    rows = _rows(path)
    earlier = [row for row in rows if row["fingerprint"] == key]
    started = [row for row in earlier if row["status"] == STARTED]
    run = {"date": datetime.now(UTC).date().isoformat(), "version": version, "read": read, "command": ledger.command}
    if started:  # a run that saved nothing: the same read, which this run completes
        started[-1].update(run)
        _write(path, rows)
        return dict(started[-1])
    if earlier and not ledger.override:
        first = earlier[0]
        raise TestYearAlreadyRead(
            f"{refused} {key} (version {first['version']}, {first['date']}, {first['read']}; ledger {ledger.path}). "
            f"ADR 0009 allows one read per frozen version; pass {OVERRIDE_FLAG} to read it again, which is logged."
        )
    row = run | {
        "fingerprint": key, "override": "true" if earlier else "false", "record": record, "kind": kind, "status": STARTED,
    }
    _write(path, [*rows, row])
    return row


def _rows(path: Path) -> list[dict]:
    """The ledger's rows as written, each with every column of `COLUMNS` (`_ADDED_COLUMNS` where the file has none)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return [{**_ADDED_COLUMNS, **{k: v for k, v in row.items() if v is not None}} for row in csv.DictReader(file)]


def _write(path: Path, rows: list[dict]) -> None:
    """Write every row with every column of `COLUMNS`; earlier rows keep their values as written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
