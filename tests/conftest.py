"""Shared fixtures: one fixture world and one trained artefact bundle per test session."""

import os
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor import baci, policy, tariffs
from exportspredictor.features import FEATURE_SET_ENV
from exportspredictor.fixture import build_world
from exportspredictor.pipeline import Parts, build_artefacts, fixture_rescore

FIXTURE_VERSION = "fixture-test"
TAIWAN_BACI = "S19"  # BACI's code for Taiwan ("Other Asia, nes")
APP_ROUTER = Path(__file__).resolve().parents[1] / "app" / "Home.py"


def pytest_configure(config):
    """Run the suite on the original feature set, whatever the shell has selected.

    A real build sets `EXPORTSPREDICTOR_FEATURE_SET` (e.g. `exporter_count`), and the fixture artefacts
    refuse a set they weren't built on. Tests of a set select it themselves (`monkeypatch.setenv`, or
    `feature_set_name=`). Cleared here, before any session fixture builds.
    """
    os.environ.pop(FEATURE_SET_ENV, None)


def run_page(page: str | None = None, *, query: dict[str, str] | None = None) -> AppTest:
    """Open one app page as a visitor does, through the router (`app/Home.py`), against the API that
    `EXPORTSPREDICTOR_API_URL` names.

    `run_page()` opens the default page (the Leaderboard); `run_page("pages/scenarios.py")` switches to another
    page, with `query` as its URL parameters. Returns the AppTest after the page's run.
    """
    app = AppTest.from_file(str(APP_ROUTER), default_timeout=30)
    if page is not None:
        app.run()  # registers the router's pages, so switch_page can find them
        app.switch_page(page)
    app.query_params.update(query or {})
    return app.run()


@pytest.fixture
def open_page(monkeypatch):
    """`run_page` against the API at a given URL: `open_page(api_url, "pages/scenarios.py")`."""

    def open_(api_url: str, page: str | None = None, *, query: dict[str, str] | None = None) -> AppTest:
        monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
        return run_page(page, query=query)

    return open_


@pytest.fixture(scope="session")
def fixture_world():
    return build_world()


@pytest.fixture(scope="session")
def fixture_artefacts(fixture_world):
    """A run with every part, and the ITC comparison added on a synthetic base-year panel (the fixture world's four
    years are too few for it), as `realisation.add_to_latest` adds it to a real version."""
    from exportspredictor import realisation

    bundle = build_artefacts(
        fixture_world, version=FIXTURE_VERSION, parts=Parts.every(), rescore=fixture_rescore(fixture_world)
    )
    return realisation.with_itc_comparison(bundle, realisation.realisation_report(realisation.fixture_panel()))


@pytest.fixture(scope="session")
def fixture_default_artefacts(fixture_world, fixture_artefacts):
    """A default run (`Parts()`): no Event Backtests, constraint comparison or gain ladder.

    It reuses the full run's search results, so it trains only the fold and test-year models.
    """
    return build_artefacts(
        fixture_world, version="fixture-default", search_results=fixture_artefacts.metadata["search"]
    )


@pytest.fixture(scope="session")
def fixture_default_artefacts_dir(fixture_default_artefacts, tmp_path_factory):
    root = tmp_path_factory.mktemp("default_artefacts")
    store.save(fixture_default_artefacts, root)
    return root


@pytest.fixture(scope="session")
def fixture_artefacts_dir(fixture_artefacts, tmp_path_factory):
    """A folder holding one saved artefact version, as the API would find it."""
    root = tmp_path_factory.mktemp("artefacts")
    store.save(fixture_artefacts, root)
    return root


@contextmanager
def serve_api(root, **options):
    """Run the API on a free local port over an artefact root (`options` go to `create_app`); yields its base URL."""
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(root, **options), host="127.0.0.1", port=port, log_level="warning"))
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


# Every amount of the Baseline and the yearly history is multiplied by this in the Blocked world (`blocked_artefacts`).
BLOCKED_WORLD_SCALE = 1000


@pytest.fixture(scope="session")
def blocked_artefacts(fixture_artefacts):
    """The fixture artefacts with every Baseline and history amount × 1,000, so some pairs clear the $1M threshold.

    The fixture's own Blocked Opportunity values are all under $1M, so it has no Blocked Opportunity. Scaled, four
    pairs clear both thresholds, two in each Market without a UK Trade Agreement: whisky and jewellery in India,
    jewellery and cars in Hong Kong. Cars in Hong Kong has no room to grow (its Baseline-year Actual Exports are above
    Export Potential with a Trade Agreement), so three are Blocked Opportunities.

    Two figures are set so every kind of row exists (ADR 0011). The tiny fold models price none of India's tariffs, so
    every Tariff part is 0; whisky in India (150%) is given a Tariff part of `BLOCKED_WHISKY_TARIFF_SHARE` of its value.
    Jewellery in Hong Kong (tariff 0%) gets Baseline-year Actual Exports halfway between its Export Potential and its
    Export Potential with a Trade Agreement, so it is "above Export Potential". Tariffs and fold models are unchanged.
    """

    def scaled(table: pl.DataFrame) -> pl.DataFrame:
        return table.with_columns(pl.col(r"^.*_usd$") * BLOCKED_WORLD_SCALE)

    baseline = scaled(fixture_artefacts.baseline)
    whisky_india = (pl.col("hs6") == "220830") & (pl.col("iso3") == "IND")
    baseline = baseline.with_columns(
        tariff_part_usd=pl.when(whisky_india)
        .then(pl.col("blocked_opportunity_value_usd") * BLOCKED_WHISKY_TARIFF_SHARE)
        .otherwise(pl.col("tariff_part_usd"))
    )
    [jewellery] = baseline.filter((pl.col("hs6") == "711319") & (pl.col("iso3") == "HKG")).iter_rows(named=True)
    above = jewellery["baseline_export_potential_usd"] + jewellery["blocked_opportunity_value_usd"] / 2
    history = scaled(fixture_artefacts.history)
    in_baseline_year = (
        (pl.col("hs6") == "711319") & (pl.col("iso3") == "HKG") & (pl.col("year") == fixture_artefacts.baseline_year)
    )
    history = history.with_columns(
        actual_exports_usd=pl.when(in_baseline_year).then(above).otherwise(pl.col("actual_exports_usd"))
    )
    return replace(fixture_artefacts, baseline=baseline, history=history)


BLOCKED_WHISKY_TARIFF_SHARE = 0.4  # whisky in India's Tariff part, as a share of its value, in the Blocked world


@pytest.fixture(scope="session")
def blocked_artefacts_dir(blocked_artefacts, tmp_path_factory):
    root = tmp_path_factory.mktemp("blocked_artefacts")
    store.save(blocked_artefacts, root)
    return root


@pytest.fixture(scope="session")
def blocked_api_url(blocked_artefacts_dir):
    """The API serving the Blocked world (`blocked_artefacts`) for the whole session."""
    with serve_api(blocked_artefacts_dir) as url:
        yield url


@pytest.fixture(scope="session")
def unscenario_artefacts_dir(fixture_artefacts, tmp_path_factory):
    """The fixture artefacts with two more Markets a Scenario does not accept, having no UK feature rows: the United
    Kingdom and Mayotte."""
    extra = pl.DataFrame(
        {"iso3": ["GBR", "MYT"], "name": ["United Kingdom", "Mayotte"], "region": ["Europe & Central Asia"] * 2}
    )
    markets = pl.concat([fixture_artefacts.markets, extra.cast(dict(fixture_artefacts.markets.schema))])
    root = tmp_path_factory.mktemp("unscenario_artefacts")
    store.save(replace(fixture_artefacts, markets=markets), root)
    return root


# The Offset world: the Blocked world with a Heading of three Products. Telephones and medicaments are
# recoded as two more Products of Heading 8703, beside cars (870323), so Heading 8703 holds three Products in every
# Market. Its twelve pairs get round Gap Period figures, so their Export Gaps are exact.
CARS, BIG_CARS, SMALL_CARS = "870323", "870324", "870322"
OFFSET_RECODED = {
    "851712": (BIG_CARS, "Motor cars, spark-ignition engine of more than 3,000 cc"),
    "300490": (SMALL_CARS, "Motor cars, spark-ignition engine of more than 1,000 cc but not more than 1,500 cc"),
}
# (iso3, hs6) -> (Export Potential, Actual Exports) in $m; the Export Gap is the difference.
OFFSET_FIGURES = {
    # Japan: cars +10, offset by 12 (the rest of the Heading nets to -12): an Offset Pair. Small cars' Export Gap is
    # zero, so they are no Opportunity, though the Heading as a whole sells above Export Potential.
    ("JPN", CARS): (30, 20), ("JPN", BIG_CARS): (10, 22), ("JPN", SMALL_CARS): (10, 10),
    # Hong Kong: cars +10, offset by exactly half (big cars -4, small cars -1): an Offset Pair, and a Blocked
    # Opportunity of the Blocked world.
    ("HKG", CARS): (30, 20), ("HKG", BIG_CARS): (10, 14), ("HKG", SMALL_CARS): (10, 11),
    # Canada: cars +10, offset by 4.99, just under half: big cars' surplus of 6 is netted against small cars' +1.01.
    ("CAN", CARS): (30, 20), ("CAN", BIG_CARS): (10, 16), ("CAN", SMALL_CARS): (10, 8.99),
    # India: every Product of the Heading under-sells.
    ("IND", CARS): (30, 20), ("IND", BIG_CARS): (10, 7), ("IND", SMALL_CARS): (10, 9),
}


def offset_recoded(table: pl.DataFrame) -> pl.DataFrame:
    """`table` (with `hs6`) in the Offset world's Product codes (`OFFSET_RECODED`)."""
    recode = {old: new for old, (new, _) in OFFSET_RECODED.items()}
    return table.with_columns(pl.col("hs6").cast(pl.String).replace(recode))


def with_gap_period_figures(bundle, figures):
    """`bundle` with these pairs' Gap Period Export Potential and Actual Exports (`{(iso3, hs6): (potential, actual)}`,
    in $m), their Export Gap the difference."""
    table = pl.DataFrame(
        [(iso3, hs6, potential * 1e6, actual * 1e6) for (iso3, hs6), (potential, actual) in figures.items()],
        schema={"iso3": pl.String, "hs6": pl.String, "potential": pl.Float64, "actual": pl.Float64},
        orient="row",
    )
    baseline = (
        bundle.baseline.join(table, on=["hs6", "iso3"], how="left")
        .with_columns(
            export_potential_usd=pl.coalesce("potential", "export_potential_usd"),
            actual_exports_usd=pl.coalesce("actual", "actual_exports_usd"),
        )
        .with_columns(export_gap_usd=pl.col("export_potential_usd") - pl.col("actual_exports_usd"))
        .select(bundle.baseline.columns)
    )
    return replace(bundle, baseline=baseline)


@pytest.fixture(scope="session")
def offset_world(blocked_artefacts):
    """The Offset world as saved before the Offset Pair tier: its Baseline lacks the tier's columns."""
    from exportspredictor import offsets
    from exportspredictor.caveats import OFFSET_PAIR

    descriptions = {new: description for new, description in OFFSET_RECODED.values()}
    bundle = blocked_artefacts
    products = offset_recoded(bundle.products).with_columns(
        description=pl.col("hs6").replace_strict(descriptions, default=pl.col("description")),
        sector=pl.col("hs6").str.slice(0, 2),
    )
    recoded_bundle = replace(
        bundle,
        baseline=offset_recoded(bundle.baseline).drop(OFFSET_PAIR.columns, strict=False),
        baseline_features=offset_recoded(bundle.baseline_features),
        history=offset_recoded(bundle.history),
        products=products,
        metadata={key: value for key, value in bundle.metadata.items() if key != offsets.METADATA_KEY},
    )
    # Cars to Hong Kong get room to grow (ADR 0011), so they are a Blocked Opportunity here: Baseline-year Actual
    # Exports of half their Baseline-year Export Potential.
    [cars] = bundle.baseline.filter((pl.col("hs6") == CARS) & (pl.col("iso3") == "HKG")).iter_rows(named=True)
    in_baseline_year = (pl.col("hs6") == CARS) & (pl.col("iso3") == "HKG") & (pl.col("year") == bundle.baseline_year)
    history = recoded_bundle.history.with_columns(
        actual_exports_usd=pl.when(in_baseline_year)
        .then(cars["baseline_export_potential_usd"] / 2)
        .otherwise(pl.col("actual_exports_usd"))
    )
    return with_gap_period_figures(replace(recoded_bundle, history=history), OFFSET_FIGURES)


@pytest.fixture(scope="session")
def offset_artefacts(offset_world):
    """The Offset world marked with the declared Caveat Tiers (`recaveat`): cars to Japan and to Hong Kong are its
    Offset Pairs."""
    from exportspredictor.pipeline import recaveat

    return recaveat(offset_world, version="fixture-offset")


@pytest.fixture(scope="session")
def offset_artefacts_dir(offset_artefacts, tmp_path_factory):
    root = tmp_path_factory.mktemp("offset_artefacts")
    store.save(offset_artefacts, root)
    return root


@pytest.fixture(scope="session")
def offset_api_url(offset_artefacts_dir):
    """The API serving the Offset world (`offset_artefacts`) for the whole session."""
    with serve_api(offset_artefacts_dir) as url:
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
def v1_frozen(fixture_world, tmp_path_factory):
    """The fixture's synthetic frozen-v1 folder (`pipeline.fixture_v1_frozen`) with its manifest.

    Its held-out Lift Exporters are FRA, and MEX, which is outside the fixture's Exporter scope.
    """
    from exportspredictor import freeze, pipeline

    frozen = pipeline.fixture_v1_frozen(fixture_world)
    return freeze.write(tmp_path_factory.mktemp("v1"), frozen.tables, frozen.manifest)
