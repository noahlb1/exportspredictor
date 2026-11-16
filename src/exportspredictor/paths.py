"""Where data and artefacts live. Both roots can be overridden by env vars.

- `EXPORTSPREDICTOR_DATA_DIR`: raw downloads and derived Parquet (default `<repo>/data`).
- `EXPORTSPREDICTOR_ARTEFACTS_DIR`: versioned artefact folders (default `<repo>/artefacts`).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("EXPORTSPREDICTOR_DATA_DIR", REPO_ROOT / "data"))


def artefacts_dir() -> Path:
    return Path(os.environ.get("EXPORTSPREDICTOR_ARTEFACTS_DIR", REPO_ROOT / "artefacts"))
