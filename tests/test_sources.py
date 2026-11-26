"""The version or download date of every data source, as recorded in artefact metadata."""

import json
import os
from datetime import UTC, datetime

from exportspredictor import governance, sources
from exportspredictor.pipeline import refresh

NAMES = ["BACI", "WTO RTA-IS", "World Bank WDI", "IMF World Economic Outlook", "CEPII Gravity",
         "Bank of England USD/GBP rate", "HMRC Overseas Trade Statistics", "UNCTAD TRAINS",
         "UNSD HS correlation tables"]


def _raw_tree(root):
    """A data root holding a stand-in for every raw download, each dated 2026-11-16."""
    raw = root / "raw"
    files = {
        "baci/BACI_HS12_V202601.zip": b"",
        "gravity/Gravity_csv_V202211.zip": b"",
        "wdi/NY.GDP.MKTP.CD.json": json.dumps([{"page": 1, "lastupdated": "2026-11-11"}, []]).encode(),
        "wdi/SP.POP.TOTL.json": json.dumps([{"page": 1, "lastupdated": "2026-11-11"}, []]).encode(),
        "imf_weo/NGDPD.json": b"{}",
        "imf_weo/LP.json": b"{}",
        "wto/AllRTAs.xlsx": b"",
        "hmrc/bdsexp_23archive.zip": b"",
        "hmrc/HS2022toHS2012ConversionAndCorrelationTables.xlsx": b"",
        "trains/availability.xml": b"",
        "trains/manifest.json": json.dumps({"reporter_years": 1, "pulled": "2026-11-15"}).encode(),
        "unsd/HS2017toHS2012ConversionAndCorrelationTables.xlsx": b"",
        "unsd/HS2012toHS2007CorrelationAndConversionTables.xls": b"",
    }
    stamp = datetime(2026, 11, 16, 12, tzinfo=UTC).timestamp()
    for name, content in files.items():
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        os.utime(path, (stamp, stamp))
    return root


def test_every_source_has_a_version_or_a_download_date(tmp_path):
    found = {row["source"]: row for row in sources.source_versions(_raw_tree(tmp_path))}

    assert list(found) == NAMES
    assert found["BACI"]["version"] == "HS12 V202601"
    assert found["CEPII Gravity"]["version"] == "V202211"
    assert found["World Bank WDI"]["version"] == "last updated 2026-11-11"
    assert found["Bank of England USD/GBP rate"]["version"] == "series XUAAUSS"
    assert found["Bank of England USD/GBP rate"]["downloaded"] == "2026-11-17"
    assert found["UNCTAD TRAINS"]["downloaded"] == "2026-11-15"  # the pull date the manifest records
    for name in ("WTO RTA-IS", "IMF World Economic Outlook", "HMRC Overseas Trade Statistics", "BACI",
                 "UNSD HS correlation tables"):
        assert found[name]["downloaded"] == "2026-11-16", name


def test_a_missing_download_is_recorded_as_missing(tmp_path):
    found = {row["source"]: row for row in sources.source_versions(tmp_path)}

    assert found["WTO RTA-IS"]["downloaded"] is None
    assert found["UNCTAD TRAINS"]["downloaded"] is None
    assert found["BACI"]["version"] == "HS12 V202601"  # the version the code reads


def test_refresh_records_the_source_versions_without_retraining(fixture_artefacts, tmp_path):
    refreshed = refresh(fixture_artefacts, version="refreshed", data_root=_raw_tree(tmp_path))

    assert [row["source"] for row in refreshed.metadata["sources"]] == NAMES
    assert refreshed.fold_models is fixture_artefacts.fold_models


def test_the_model_card_lists_the_source_versions(fixture_artefacts, tmp_path):
    refreshed = refresh(fixture_artefacts, version="refreshed", data_root=_raw_tree(tmp_path))

    text = governance.render_model_card(governance.model_card(refreshed))
    assert "## Source versions" in text and "HS12 V202601" in text


def test_a_refreshed_copy_rewrites_its_model_card_under_its_own_artefact_id(fixture_artefacts_dir, tmp_path):
    import shutil

    from exportspredictor import artefacts as store
    from exportspredictor.pipeline import save_refreshed

    found = store.latest_folder(fixture_artefacts_dir)
    source = shutil.copytree(found, tmp_path / "source" / found.name)  # the session's folder stays as saved
    (source / governance.MODEL_CARD_FILE).write_text("# Model card: UK Exports Predictor, version stale\n", encoding="utf-8")

    folder = save_refreshed(source, tmp_path / "artefacts", data_root=_raw_tree(tmp_path))

    title = (folder / governance.MODEL_CARD_FILE).read_text(encoding="utf-8").splitlines()[0]
    assert title == f"# Model card: UK Exports Predictor, Artefact ID {folder.name}"
    assert folder.name != source.name
    assert store.load(folder).metadata["refreshed_from"] == source.name
