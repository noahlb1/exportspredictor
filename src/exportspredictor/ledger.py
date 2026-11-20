"""The test-year ledger: every read of the test year, and the guard that allows one per configuration (ADR 0009).

The test year (2023) is read once per frozen version. Every choice is made on the validation year
(2022) and fixed before the test year is opened, and no test-year result feeds back into that
version. A version is identified by its **configuration fingerprint** (`fingerprint`): a hash of
the metadata that decides what the test-year models are and how they are scored, so rebuilding
the same configuration under a new version name is still the same version.

The ledger (`LEDGER_PATH`, a CSV in the repo) has one row per read: date, version, fingerprint,
what was read, the command, whether the read was an override, and the record it comes from. Its
first three rows are the reads made before the ledger existed (v1, v2 and the Blend), taken from
their records. Every command that reads the test year calls `record_read` first, which refuses a
second read of a fingerprint already in the ledger unless the `Ledger` says `override`, and logs
the override like any other read. The notebook shows the ledger through `read_ledger`, which
reads only this file.
"""

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from exportspredictor.paths import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "docs" / "test-year-ledger.csv"
COLUMNS = ("date", "version", "fingerprint", "read", "command", "override", "record")
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


def read_ledger(path: Path = LEDGER_PATH) -> pl.DataFrame:
    """The ledger's rows, oldest first; empty (with its columns) if the file does not exist yet."""
    schema = {column: pl.String for column in COLUMNS} | {"override": pl.Boolean}
    if not Path(path).exists():
        return pl.DataFrame(schema=schema)
    return pl.read_csv(path, schema=schema, encoding="utf8")


def record_read(ledger: Ledger, metadata: Mapping, *, read: str, record: str = "") -> dict:
    """Record a read of the test year of the configuration in `metadata`, before anything reads it.

    Raises `TestYearAlreadyRead` if the ledger already holds this fingerprint and `ledger.override`
    is false. An override is recorded like any other read, with `override` true. Returns the row.
    """
    key = fingerprint(metadata)
    earlier = read_ledger(ledger.path).filter(pl.col("fingerprint") == key)
    if earlier.height and not ledger.override:
        first = earlier.row(0, named=True)
        raise TestYearAlreadyRead(
            f"The test year {metadata.get('test_year')} was already read for configuration {key} "
            f"(version {first['version']}, {first['date']}, {first['read']}; ledger {ledger.path}). "
            f"ADR 0009 allows one read per frozen version; pass {OVERRIDE_FLAG} to read it again, which is logged."
        )
    row = {
        "date": datetime.now(UTC).date().isoformat(),
        "version": str(metadata.get("version", "")),
        "fingerprint": key,
        "read": read,
        "command": ledger.command,
        "override": "true" if earlier.height else "false",
        "record": record,
    }
    path = Path(ledger.path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row
