"""Shared fixtures: one fixture world and one trained artefact bundle per test session."""

import socket
import threading
import time
from contextlib import contextmanager

import polars as pl
import pytest
import uvicorn

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor import baci, policy, tariffs
from exportspredictor.fixture import build_world
from exportspredictor.pipeline import build_artefacts

FIXTURE_VERSION = "fixture-test"
TAIWAN_BACI = "S19"  # BACI's code for Taiwan ("Other Asia, nes")


@pytest.fixture(scope="session")
def fixture_world():
    return build_world()


@pytest.fixture(scope="session")
def fixture_artefacts(fixture_world):
    return build_artefacts(fixture_world, version=FIXTURE_VERSION)


@pytest.fixture(scope="session")
def fixture_artefacts_dir(fixture_artefacts, tmp_path_factory):
    """A folder holding one saved artefact version, as the API would find it."""
    root = tmp_path_factory.mktemp("artefacts")
    store.save(fixture_artefacts, root)
    return root


@contextmanager
def serve_api(root):
    """Run the API on a free local port over an artefact root; yields its base URL."""
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(root), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def live_api():
    """Start the API on a free local port over an artefact root: `with live_api(root) as url:`."""
    return serve_api


@pytest.fixture(scope="session")
def fixture_api_url(fixture_artefacts_dir):
    """The API serving the fixture artefacts for the whole session (for Streamlit page tests)."""
    with serve_api(fixture_artefacts_dir) as url:
        yield url


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def derived_data_root(tmp_path):
    """The fixture world written to disk as the BACI, gravity, policy and tariff build steps write it.

    The USA is renamed to Taiwan, which BACI keys as `S19` and every other table as `TWN`.
    """
    world = build_world()
    rename = {"USA": "TWN"}
    flows = world.flows.filter(pl.col("value_usd") > 0).with_columns(
        pl.col("exporter").replace({"USA": TAIWAN_BACI})
    )
    countries = sorted(set(flows["exporter"]) | set(flows["importer"]))
    codes = pl.Enum(countries)
    derived = tmp_path / "derived"
    baci.flows_dir(tmp_path).mkdir(parents=True)
    for year in world.flows["year"].unique():
        flows.filter(pl.col("year") == year).select(
            "year",
            pl.col("exporter").cast(codes),
            pl.col("importer").cast(codes),
            pl.col("hs6").cast(pl.Enum(sorted(world.products["hs6"]))),
            "value_usd",
        ).write_parquet(baci.flows_dir(tmp_path) / f"flows_{year}.parquet")
    baci.lookups_dir(tmp_path).mkdir(parents=True)
    pl.DataFrame(
        {
            "iso3": countries,
            "name": countries,
            "region": "x",
            "income_level": ["Lower middle income" if c == "IND" else "High income" for c in countries],
            "baci_code": list(range(len(countries))),
        }
    ).write_parquet(baci.lookups_dir(tmp_path) / "markets.parquet")
    world.products.write_parquet(baci.lookups_dir(tmp_path) / "products.parquet")
    pl.DataFrame(
        {"baci_code": list(range(len(countries))), "baci_iso3": countries,
         "iso3": [{TAIWAN_BACI: "TWN"}.get(c, c) for c in countries]}
    ).write_parquet(derived / "country_map.parquet")
    world.country_year.with_columns(pl.col("iso3").replace(rename)).write_parquet(derived / "country_year.parquet")
    world.pairs.with_columns(pl.col("iso3_o").replace(rename)).write_parquet(derived / "gravity_pairs.parquet")
    paths = policy.policy_paths(tmp_path)
    world.policy.with_columns(pl.col("exporter").replace(rename)).write_parquet(paths["agreements"])
    world.blocs.write_parquet(paths["trading_blocs"])
    tariffs.derived_dir(tmp_path).mkdir(parents=True)
    world.tariffs.schedules.write_parquet(tariffs.derived_dir(tmp_path) / "schedules.parquet")
    world.tariffs.mfn.write_parquet(tariffs.derived_dir(tmp_path) / "mfn.parquet")
    world.tariffs.preferences.with_columns(pl.col("exporter").replace(rename)).write_parquet(
        tariffs.derived_dir(tmp_path) / "preferences.parquet"
    )
    return tmp_path


@pytest.fixture(scope="session")
def v1_frozen(tmp_path_factory):
    """A synthetic frozen-v1 folder holding only the held-out Lift Exporters and their manifest entry.

    The fixture's stand-in for v1's list: FRA, and MEX, which is outside the fixture's Exporter scope.
    """
    import hashlib
    import json

    from exportspredictor import freeze, pipeline

    folder = tmp_path_factory.mktemp("v1") / freeze.FOLDER
    folder.mkdir()
    path = folder / freeze.TABLES["lift_held_out_exporters"]
    pl.DataFrame({"exporter": list(pipeline.FIXTURE_V1_LIFT_EXPORTERS)}).write_parquet(path)
    entry = {"file": path.name, "rows": len(pipeline.FIXTURE_V1_LIFT_EXPORTERS), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / freeze.MANIFEST_FILE).write_text(json.dumps({"tables": {"lift_held_out_exporters": entry}}), encoding="utf-8")
    return folder
