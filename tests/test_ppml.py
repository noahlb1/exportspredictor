"""The PPML gravity benchmark of the Trade Agreement Scenario Effect, saved with the artefact."""

import numpy as np
import polars as pl
import pytest

from exportspredictor import artefacts as store
from exportspredictor import governance, ppml, prediction, scenarios


def _panel(seed=0, beta=0.4, n_exporters=6, n_importers=6, n_sectors=2, years=(2012, 2015, 2018, 2021)):
    """A small 3-way gravity panel with Poisson Flows and a Trade Agreement switching on for some pairs."""
    rng = np.random.default_rng(seed)
    rows = []
    switch = {(e, i): rng.choice([None, *years[1:]]) for e in range(n_exporters) for i in range(n_importers)}
    exp_fe = rng.normal(0, 0.5, (n_exporters, n_sectors, len(years)))
    imp_fe = rng.normal(0, 0.5, (n_importers, n_sectors, len(years)))
    pair_fe = rng.normal(0, 1.0, (n_exporters, n_importers, n_sectors))
    for e in range(n_exporters):
        for i in range(n_importers):
            if e == i:
                continue
            for s in range(n_sectors):
                for t, year in enumerate(years):
                    start = switch[(e, i)]
                    ta = int(start is not None and year >= start)
                    mu = np.exp(2 + exp_fe[e, s, t] + imp_fe[i, s, t] + pair_fe[e, i, s] + beta * ta)
                    rows.append((f"E{e}", f"E{i}", f"{s:02d}", year, float(rng.poisson(mu)), ta))
    return pl.DataFrame(
        rows,
        schema=["exporter", "importer", "sector", "year", "value_usd", "trade_agreement"],
        orient="row",
    )


def _dense_poisson(panel):
    """Reference: Newton's method on the full dummy-variable Poisson likelihood, and its pair-clustered sandwich."""
    groups = [
        panel.select(pl.concat_str("exporter", "sector", "year")).to_series(),
        panel.select(pl.concat_str("importer", "sector", "year")).to_series(),
        panel.select(pl.concat_str("exporter", "importer", "sector")).to_series(),
    ]
    dummies = []
    for g in groups:
        labels, index = np.unique(g.to_numpy(), return_inverse=True)
        dummies.append(np.eye(len(labels))[index])
    x = panel["trade_agreement"].to_numpy().astype(float)[:, None]
    design = np.hstack([x, *dummies])
    # Drop collinear dummy columns so the Hessian is invertible.
    keep = []
    for j in range(design.shape[1]):
        trial = design[:, [*keep, j]]
        if np.linalg.matrix_rank(trial) == len(keep) + 1:
            keep.append(j)
    design = design[:, keep]
    y = panel["value_usd"].to_numpy()
    coef = np.linalg.lstsq(design, np.log((y + y.mean()) / 2), rcond=None)[0]
    for _ in range(100):
        mu = np.exp(design @ coef)
        step = np.linalg.solve(design.T @ (design * mu[:, None]), design.T @ (y - mu))
        coef += step
        if np.abs(step).max() < 1e-12:
            break
    mu = np.exp(design @ coef)
    bread = np.linalg.inv(design.T @ (design * mu[:, None]))
    cluster = panel.select(pl.concat_str("exporter", "importer")).to_series().to_numpy()
    scores = design * (y - mu)[:, None]
    labels, index = np.unique(cluster, return_inverse=True)
    summed = np.zeros((len(labels), design.shape[1]))
    np.add.at(summed, index, scores)
    g = len(labels)
    meat = summed.T @ summed * g / (g - 1)
    variance = bread @ meat @ bread
    return coef[0], float(np.sqrt(variance[0, 0]))


def test_fit_matches_a_dense_dummy_variable_poisson_fit_and_its_clustered_standard_error():
    panel = _panel()
    expected_coef, expected_se = _dense_poisson(panel)

    fit = ppml.fit_ppml(panel)

    assert fit.converged
    assert fit.coefficient == pytest.approx(expected_coef, abs=1e-6)
    assert fit.std_error == pytest.approx(expected_se, rel=1e-4)


def test_sector_flows_build_a_balanced_sector_panel_of_ever_traded_cells_without_excluded_products():
    flows = pl.DataFrame(
        {
            "exporter": ["GBR", "GBR", "GBR", "GBR", "DEU", "GBR", "FRA", "GBR"],
            "importer": ["JPN", "JPN", "JPN", "CAN", "JPN", "GBR", "JPN", "CAN"],
            "hs6": ["220830", "220890", "270900", "870323", "220830", "220830", "220830", "870323"],
            "year": [2015, 2015, 2015, 2015, 2018, 2015, 2015, 2018],
            "value_usd": [10.0, 5.0, 99.0, 0.0, 7.0, 3.0, 4.0, 0.0],
        },
        schema_overrides={"year": pl.Int16},
    )
    agreements = pl.DataFrame(
        {"exporter": ["GBR"], "importer": ["JPN"], "year": [2018], "trade_agreement": [1], "customs_union": [0]},
        schema_overrides={"year": pl.Int32, "trade_agreement": pl.Int8, "customs_union": pl.Int8},
    )

    panel = ppml.sector_flows(flows, agreements, years=(2015, 2018), exporters=["GBR", "DEU"])

    assert panel.sort("exporter", "importer", "sector", "year").to_dicts() == [
        {"exporter": "DEU", "importer": "JPN", "sector": "22", "year": 2015, "value_usd": 0.0, "trade_agreement": 0},
        {"exporter": "DEU", "importer": "JPN", "sector": "22", "year": 2018, "value_usd": 7.0, "trade_agreement": 0},
        {"exporter": "GBR", "importer": "JPN", "sector": "22", "year": 2015, "value_usd": 15.0, "trade_agreement": 0},
        {"exporter": "GBR", "importer": "JPN", "sector": "22", "year": 2018, "value_usd": 0.0, "trade_agreement": 1},
    ]


def test_the_models_trade_agreement_effect_is_measured_on_pairs_without_one():
    baseline = pl.DataFrame(
        {
            "hs6": ["220830", "220830", "870323", "870323"],
            "iso3": ["JPN", "IND", "JPN", "IND"],
            "baseline_export_potential_usd": [100.0, 300.0, 50.0, 1000.0],
            "blocked_opportunity_value_usd": [0.0, 30.0, 0.0, 300.0],
        }
    )
    baseline_features = pl.DataFrame(
        {
            "hs6": ["220830", "220830", "870323", "870323"],
            "iso3": ["JPN", "IND", "JPN", "IND"],
            "trade_agreement": [1, 0, 1, 0],
        },
        schema_overrides={"trade_agreement": pl.Int8},
    )

    effect = ppml.model_agreement_effect(baseline, baseline_features)

    # IND pairs only: +$330 on a $1,300 Baseline (25.4%); per pair +10% and +30%.
    assert effect["pairs"] == 2
    assert effect["baseline_usd"] == pytest.approx(1300.0)
    assert effect["effect_usd"] == pytest.approx(330.0)
    assert effect["effect_pct"] == pytest.approx(330 / 1300 * 100)
    assert effect["mean_pair_pct"] == pytest.approx(20.0)
    assert effect["median_pair_pct"] == pytest.approx(20.0)


def test_one_table_sets_the_benchmark_against_the_model_and_the_literature():
    fit = ppml.PpmlFit(
        coefficient=0.2, std_error=0.05, rows=100, dropped_rows=0, clusters=10, iterations=5, converged=True, seconds=1.0
    )
    model = {"pairs": 2, "baseline_usd": 1300.0, "effect_usd": 330.0, "effect_pct": 25.0, "mean_pair_pct": 20.0,
             "median_pair_pct": 18.0}

    table = ppml.comparison_table(fit, model)
    rows = {row["estimate"]: row for row in table.to_dicts()}

    benchmark = rows["PPML benchmark: 3-way fixed effects, Sector level"]
    assert benchmark["effect_pct"] == pytest.approx(22.14, abs=0.01)  # exp(0.2) - 1
    assert benchmark["low_pct"] == pytest.approx(10.74, abs=0.01)  # exp(0.2 - 1.96 * 0.05) - 1
    assert benchmark["high_pct"] == pytest.approx(34.72, abs=0.01)
    assert rows["Model: Trade Agreement Scenario Effect, Baseline-weighted"]["effect_pct"] == 25.0
    assert rows["Model: Trade Agreement Scenario Effect, mean per pair"]["effect_pct"] == 20.0
    # From Yotov et al. (2016) PPML without pair effects (0.191, +21%) to Baier & Bergstrand's (2007) doubling.
    literature = rows["Literature range (RTA estimates)"]
    assert literature["low_pct"] == pytest.approx(21.0, abs=0.1)
    assert literature["high_pct"] == pytest.approx(100.0, abs=0.1)
    assert all(row["source"] for row in table.to_dicts())


def test_the_benchmark_runs_on_the_fixture_world(fixture_world):
    panel = ppml.sector_flows(fixture_world.flows, fixture_world.sources.agreements, years=(2021, 2022, 2023, 2024))

    fit = ppml.fit_ppml(panel)

    assert "27" not in panel["sector"].to_list()  # the Excluded Product (crude oil) is left out
    assert panel["trade_agreement"].n_unique() == 2
    assert fit.converged
    assert np.isfinite(fit.coefficient) and fit.std_error > 0


# --- The saved benchmark: fitted once by a package command, carried with the artefact ---


def _fixture_panel(world):
    return ppml.sector_flows(world.flows, world.sources.agreements, years=(2021, 2022, 2023, 2024))


@pytest.fixture(scope="module")
def derived(fixture_default_artefacts, fixture_world, tmp_path_factory):
    """A root holding a version without the benchmark and the version `add_benchmark` derived from it."""
    root = tmp_path_factory.mktemp("benchmark_artefacts")
    source = store.save(fixture_default_artefacts, root)
    target = ppml.add_benchmark(root, panel=_fixture_panel(fixture_world), version="fixture-default-ppml")
    return source, target


def test_the_benchmark_command_derives_a_newer_version_holding_the_benchmark(derived):
    source, target = derived

    assert target.name > source.name
    assert store.latest_folder(target.parent) == target
    bundle = store.load(target)
    assert bundle.metadata["ppml_benchmark_from"] == source.name
    benchmark = bundle.ppml_benchmark
    assert benchmark is not None
    fit = benchmark["ppml"]
    assert fit["converged"] and fit["effect_low_pct"] < fit["effect_pct"] < fit["effect_high_pct"]
    # The published range it is compared with: Yotov et al. (2016) without pair effects (+21%) to a doubling.
    assert benchmark["published_range"]["low_pct"] == pytest.approx(21.0, abs=0.1)
    assert benchmark["published_range"]["high_pct"] == pytest.approx(100.0, abs=0.1)
    assert benchmark["level"] == "Sector"


def test_the_derived_version_carries_the_models_tables_and_evaluation_over_unchanged(derived):
    source, target = derived

    for path in source.rglob("*"):
        if path.is_file() and path.name not in (store.METADATA_FILE, governance.MODEL_CARD_FILE):
            assert (target / path.relative_to(source)).read_bytes() == path.read_bytes(), path.name
    before, after = store.load(source), store.load(target)
    assert after.evaluation == before.evaluation
    assert after.metadata["rounds"] == before.metadata["rounds"]


def test_the_models_agreement_association_is_computed_on_the_served_model_form(derived, fixture_default_artefacts):
    _, target = derived
    bundle = store.load(target)
    model = bundle.ppml_benchmark["model"]
    assert model["model_form"] == fixture_default_artefacts.model_form.to_dict()

    # Independently: add a Trade Agreement to every UK pair without one and re-predict on the served form.
    rows = bundle.baseline_features.filter(pl.col("trade_agreement") == 0)
    added = scenarios.apply(rows, scenarios.Scenario(add_agreements=tuple(rows["iso3"].unique().sort())))
    with_agreement = float(prediction.export_potential(added, bundle.fold_models, form=bundle.model_form).sum())
    without = float(prediction.export_potential(rows, bundle.fold_models, form=bundle.model_form).sum())
    assert model["effect_pct"] == pytest.approx((with_agreement / without - 1) * 100, rel=1e-6)
    assert model["effect_pct"] >= 0  # the monotone constraints: an agreement never lowers Export Potential
