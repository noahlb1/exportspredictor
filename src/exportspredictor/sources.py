"""The version, or the download date, of every data source behind an artefact.

Sources with a published version (BACI, CEPII Gravity, the World Bank WDI's "last updated"
date) record it. Sources without one (the WTO RTA-IS export, the IMF DataMapper API, the
HMRC archive) record the date their raw file was downloaded: the file's modification date
(UTC), which the write-then-rename download leaves at the moment the download finished.
The Bank of England rate table (used only to convert HMRC's £ for the cross-check) is
copied into `currency`, so it records the copy date.

`source_versions` is stored in artefact metadata as `sources` (`pipeline.refresh` adds it
to a saved artefact without retraining).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from exportspredictor import baci, currency, gravity, hmrc, policy
from exportspredictor.paths import data_dir


def source_versions(root: Path | None = None) -> list[dict]:
    """One row per source: source, version, downloaded (ISO date or None), files (relative paths)."""
    root = Path(root or data_dir())
    raw = root / "raw"
    wdi = [raw / "wdi" / f"{indicator}.json" for indicator in gravity.WDI_INDICATORS]
    rows = [
        ("BACI", f"HS12 {baci.BACI_VERSION}", [baci.raw_dir(root) / f"BACI_HS12_{baci.BACI_VERSION}.zip"]),
        ("WTO RTA-IS", "export of all RTAs (unversioned)", [policy.rta_export_path(root)]),
        ("World Bank WDI", _wdi_version(wdi), wdi),
        (
            "IMF World Economic Outlook",
            "DataMapper API (unversioned)",
            [raw / "imf_weo" / f"{indicator}.json" for indicator in gravity.IMF_INDICATORS],
        ),
        ("CEPII Gravity", gravity.GRAVITY_VERSION, [raw / "gravity" / gravity.GRAVITY_ZIP]),
    ]
    found = [
        {
            "source": name,
            "version": version,
            "downloaded": _downloaded(files),
            "files": [f.relative_to(root).as_posix() for f in files],
        }
        for name, version, files in rows
    ]
    found.append(
        {
            "source": "Bank of England USD/GBP rate",
            "version": f"series {currency.GBP_RATE_SERIES}",
            "downloaded": currency.GBP_RATE_DOWNLOADED,
            "files": ["src/exportspredictor/currency.py"],
        }
    )
    hmrc_files = [hmrc.raw_dir(root) / hmrc.ARCHIVE_NAME, hmrc.raw_dir(root) / hmrc.CONCORDANCE_NAME]
    found.append(
        {
            "source": "HMRC Overseas Trade Statistics",
            "version": f"{hmrc.YEAR} archive {hmrc.ARCHIVE_NAME} (cross-check only)",
            "downloaded": _downloaded(hmrc_files),
            "files": [f.relative_to(root).as_posix() for f in hmrc_files],
        }
    )
    return found


def _downloaded(files: list[Path]) -> str | None:
    """The date the last of `files` was downloaded, or None if any is missing."""
    if not files or not all(f.exists() for f in files):
        return None
    latest = max(f.stat().st_mtime for f in files)
    return datetime.fromtimestamp(latest, UTC).date().isoformat()


def _wdi_version(files: list[Path]) -> str:
    """The WDI's own "last updated" date, from the API response header."""
    dates = set()
    for path in files:
        if path.exists():
            header = json.loads(path.read_text(encoding="utf-8"))[0]
            if header.get("lastupdated"):
                dates.add(header["lastupdated"])
    return f"last updated {max(dates)}" if dates else "API (last-updated date unknown)"
