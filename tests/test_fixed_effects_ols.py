"""Least squares with high-dimensional fixed effects and pair-clustered errors (#173), from the PPML benchmark's code."""

import numpy as np
import polars as pl
import pytest

from exportspredictor import ppml

FIXED_EFFECTS = (("market",), ("year", "sector"))
CLUSTER = ("product", "market")


def _panel(seed=0, betas=(0.6, -0.4), n_markets=8, n_products=6, years=(2015, 2016, 2017, 2018), noise=0.5):
    """A Product × Market × year panel with known Market and year × Sector effects, errors shared within a pair."""
    rng = np.random.default_rng(seed)
    market_fe = rng.normal(0, 2.0, n_markets)
    year_sector_fe = rng.normal(0, 2.0, (len(years), 3))
    pair_error = rng.normal(0, noise, (n_products, n_markets))
    rows = []
    for p in range(n_products):
        sector = p % 3
        for m in range(n_markets):
            for t, year in enumerate(years):
                # The regressors are correlated with the fixed effects, so a fit without them would be biased.
                x1 = rng.normal() + 0.5 * market_fe[m]
                x2 = rng.normal() + 0.5 * year_sector_fe[t, sector]
                y = (
                    market_fe[m] + year_sector_fe[t, sector] + betas[0] * x1 + betas[1] * x2
                    + pair_error[p, m] + rng.normal(0, noise)
                )
                rows.append((f"{p:06d}", f"M{m}", f"{sector:02d}", year, y, x1, x2))
    return pl.DataFrame(rows, schema=["product", "market", "sector", "year", "y", "x1", "x2"], orient="row")


def _fit(panel, **kwargs):
    return ppml.fit_fixed_effects_ols(
        panel, outcome="y", regressors=["x1", "x2"], fixed_effects=FIXED_EFFECTS, cluster=CLUSTER, **kwargs
    )


def test_the_fit_recovers_known_coefficients_under_market_and_year_sector_effects():
    panel = _panel(n_markets=30, n_products=30, noise=0.1)

    fit = _fit(panel)

    assert fit.coefficients["x1"] == pytest.approx(0.6, abs=0.02)
    assert fit.coefficients["x2"] == pytest.approx(-0.4, abs=0.02)
    assert fit.rows == panel.height and fit.dropped_rows == 0


def _dense_ols(panel):
    """Reference: least squares on the full dummy-variable design, with the pair-clustered sandwich."""
    dummies = []
    for columns in FIXED_EFFECTS:
        groups = panel.select(pl.concat_str(*columns, separator="|")).to_series().to_numpy()
        labels, index = np.unique(groups, return_inverse=True)
        dummies.append(np.eye(len(labels))[index])
    design = np.hstack([panel.select("x1", "x2").to_numpy(), *dummies])
    keep = []
    for j in range(design.shape[1]):
        if np.linalg.matrix_rank(design[:, [*keep, j]]) == len(keep) + 1:
            keep.append(j)
    design = design[:, keep]
    y = panel["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    bread = np.linalg.inv(design.T @ design)
    pairs = panel.select(pl.concat_str(*CLUSTER, separator="|")).to_series().to_numpy()
    labels, index = np.unique(pairs, return_inverse=True)
    summed = np.zeros((len(labels), design.shape[1]))
    np.add.at(summed, index, design * residuals[:, None])
    g = len(labels)
    variance = bread @ (summed.T @ summed * g / (g - 1)) @ bread
    return coef[:2], np.sqrt(np.diag(variance)[:2])


def test_coefficients_and_clustered_errors_match_a_dense_dummy_variable_fit():
    panel = _panel()
    expected_coef, expected_se = _dense_ols(panel)

    fit = _fit(panel)

    assert [fit.coefficients["x1"], fit.coefficients["x2"]] == pytest.approx(expected_coef, abs=1e-8)
    assert [fit.std_errors["x1"], fit.std_errors["x2"]] == pytest.approx(expected_se, rel=1e-6)
    assert fit.clusters == 6 * 8


def test_duplicating_each_clusters_rows_does_not_shrink_the_clustered_errors():
    panel = _panel()
    doubled = pl.concat([panel, panel])

    fit, twice = _fit(panel), _fit(doubled)

    assert twice.coefficients["x1"] == pytest.approx(fit.coefficients["x1"], abs=1e-9)
    # Independent rows would shrink the errors by sqrt(2); rows repeated within a pair add no evidence.
    assert twice.std_errors["x1"] == pytest.approx(fit.std_errors["x1"], rel=1e-6)
    assert twice.std_errors["x2"] == pytest.approx(fit.std_errors["x2"], rel=1e-6)
    assert twice.rows == 2 * panel.height and twice.clusters == fit.clusters


def test_singleton_groups_are_dropped_until_none_remain_and_counted():
    panel = _panel()
    lone_market = panel.head(1).with_columns(market=pl.lit("M_lone"), y=pl.lit(100.0))
    # Two rows of a new Market, one of them alone in its year x Sector: once it goes, so does the other.
    chained = panel.head(2).with_columns(
        market=pl.lit("M_chain"), year=pl.Series([2015, 2099]), y=pl.lit(-100.0)
    )

    fit = _fit(pl.concat([panel, lone_market, chained]))

    assert (fit.rows, fit.dropped_rows) == (panel.height, 3)
    assert fit.coefficients == pytest.approx(_fit(panel).coefficients, abs=1e-9)


def test_all_zero_outcome_groups_are_kept_unlike_ppml():
    panel = _panel().with_columns(y=pl.when(pl.col("market") == "M0").then(0.0).otherwise(pl.col("y")))

    fit = _fit(panel)

    assert fit.dropped_rows == 0


def test_a_regressor_the_fixed_effects_absorb_is_reported_and_the_rest_still_fitted():
    panel = _panel().with_columns(by_market=pl.col("market").str.slice(1).cast(pl.Float64) * 0.3)

    fit = ppml.fit_fixed_effects_ols(
        panel, outcome="y", regressors=["x1", "by_market", "x2"], fixed_effects=FIXED_EFFECTS, cluster=CLUSTER
    )

    assert fit.absorbed == ("by_market",)
    assert np.isnan(fit.coefficients["by_market"]) and np.isnan(fit.std_errors["by_market"])
    expected = _fit(panel)
    assert fit.coefficients["x1"] == pytest.approx(expected.coefficients["x1"], abs=1e-9)
    assert fit.std_errors["x2"] == pytest.approx(expected.std_errors["x2"], rel=1e-9)


def test_without_fixed_effects_it_is_least_squares_with_an_intercept():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    panel = pl.DataFrame({"pair": np.arange(200) // 2, "x": x, "y": 5.0 + 2.0 * x + rng.normal(size=200)})

    fit = ppml.fit_fixed_effects_ols(panel, outcome="y", regressors=["x"], fixed_effects=(), cluster=("pair",))

    slope = np.polyfit(x, panel["y"].to_numpy(), 1)[0]
    assert fit.coefficients["x"] == pytest.approx(slope, abs=1e-9)
    assert fit.clusters == 100
