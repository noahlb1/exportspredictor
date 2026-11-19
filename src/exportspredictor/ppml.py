"""A PPML gravity benchmark of the Trade Agreement Scenario Effect (decision #33), saved with the artefact.

The model's Scenario Effects are associations learned by a Tweedie XGBoost. This module
checks their *size* against the econometric standard for Trade Agreement effects: Poisson
pseudo-maximum likelihood (PPML, Santos Silva & Tenreyro 2006) with three sets of fixed
effects (Yotov et al. 2016, the WTO/UNCTAD *Advanced Guide*):

    Flow[e, i, s, t] = exp(a[e, s, t] + b[i, s, t] + c[e, i, s] + beta * TradeAgreement[e, i, t]) * error

on Flows aggregated to Sector level (HS2 chapters), non-Excluded Products only. The Exporter-
year and Importer-year effects are taken within each Sector, and so is the pair effect, so
beta is identified only by pairs whose Trade Agreement status changes over the years, against
the same pair before and against other pairs in the same years. ``exp(beta) - 1`` is the
proportional change in a pair's Flows the regression associates with a Trade Agreement.

The pair effects are what make this a strong benchmark *and* why it can never compete on
Export Potential: a pair held out under ADR 0001 has no pair effect to predict with.

**The estimator** is hand-rolled iteratively reweighted least squares with the fixed effects
swept out by weighted alternating projections (the method of ppmlhdfe, Correia, Guimarães &
Zylkin 2020): a dense dummy-variable design would need about a million columns. Standard
errors are clustered by Exporter–Importer pair (the level the Trade Agreement varies at).
Weidner & Zylkin (2021) show these 3-way estimates are consistent but their intervals are not
exactly centred in finite samples; the interval is read as an order of magnitude.

The years are 3-year intervals (2012, 2015, 2018, 2021, 2024), as the Advanced Guide
recommends intervals rather than consecutive years so that Flows can adjust to a change.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from exportspredictor import baci, features, scope
from exportspredictor.exclusions import drop_excluded

if TYPE_CHECKING:
    from exportspredictor.artefacts import Artefacts

YEARS = (2012, 2015, 2018, 2021, 2024)
_LOG = logging.getLogger(__name__)
FIXED_EFFECTS = (("exporter", "sector", "year"), ("importer", "sector", "year"), ("exporter", "importer", "sector"))
CLUSTER = ("exporter", "importer")
OUTCOME = "value_usd"
REGRESSOR = "trade_agreement"
CI_LEVEL = 0.95
_Z = 1.959963984540054  # the standard normal 97.5% quantile, for the 95% interval


@dataclass(frozen=True)
class PpmlFit:
    """The fitted Trade Agreement coefficient (log points) with its pair-clustered standard error."""

    coefficient: float
    std_error: float
    rows: int
    dropped_rows: int
    clusters: int
    iterations: int
    converged: bool
    seconds: float

    @property
    def ci_low(self) -> float:
        return self.coefficient - _Z * self.std_error

    @property
    def ci_high(self) -> float:
        return self.coefficient + _Z * self.std_error

    @property
    def effect_pct(self) -> float:
        """The proportional change in Flows associated with a Trade Agreement, in percent."""
        return _percent_change(self.coefficient)

    def as_dict(self) -> dict:
        return {
            "coefficient": self.coefficient,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "effect_pct": self.effect_pct,
            "effect_low_pct": _percent_change(self.ci_low),
            "effect_high_pct": _percent_change(self.ci_high),
            "rows": self.rows,
            "dropped_rows": self.dropped_rows,
            "clusters": self.clusters,
            "iterations": self.iterations,
            "converged": self.converged,
            "seconds": self.seconds,
        }


def sector_flows(
    flows: pl.LazyFrame | pl.DataFrame,
    agreements: pl.DataFrame,
    *,
    years: Sequence[int] = YEARS,
    exporters: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Flows summed to Exporter × Importer × Sector × year, with the pair's Trade Agreement flag.

    Excluded Products are dropped (ADR 0005) and so is a country's trade with itself. Every
    Exporter–Importer–Sector cell with a positive Flow in any of `years` gets a row in every
    year (0 where nothing was traded); cells never traded are left out, since a pair effect
    would fit them exactly. The dense Exporter × Importer × Sector grid is never built.
    `exporters` (ISO3) limits the Exporters, as the model's training scope does (ADR 0004).
    """
    years = [int(y) for y in years]
    lf = flows.lazy().filter(pl.col("year").cast(pl.Int32).is_in(years))
    lf = drop_excluded(lf).with_columns(pl.col("exporter", "importer").cast(pl.String))
    lf = lf.filter(pl.col("exporter") != pl.col("importer"))
    if exporters is not None:
        lf = lf.filter(pl.col("exporter").is_in(list(exporters)))
    sums = (
        lf.filter(pl.col(OUTCOME) > 0)
        .group_by(
            "exporter", "importer", sector=pl.col("hs6").cast(pl.String).str.slice(0, 2), year=pl.col("year").cast(pl.Int16)
        )
        .agg(pl.col(OUTCOME).sum())
        .collect()
    )
    cells = sums.select("exporter", "importer", "sector").unique()
    grid = cells.join(pl.DataFrame({"year": years}, schema={"year": pl.Int16}), how="cross")
    flags = agreements.select(
        pl.col("exporter", "importer").cast(pl.String), pl.col("year").cast(pl.Int16), REGRESSOR
    )
    return (
        grid.join(sums, on=["exporter", "importer", "sector", "year"], how="left")
        .join(flags, on=["exporter", "importer", "year"], how="left")
        .with_columns(pl.col(OUTCOME).fill_null(0.0), pl.col(REGRESSOR).fill_null(0).cast(pl.Int8))
        .sort("exporter", "importer", "sector", "year")
    )


def real_sector_flows(root: Path | None = None, *, years: Sequence[int] = YEARS) -> pl.DataFrame:
    """The Sector panel from the real BACI Flows, for the model's Exporter scope (ADR 0004).

    Scans the yearly Parquet files lazily; only the Sector sums are collected (about 1 M
    cells, 5 M rows over the five years).
    """
    sources, to_iso3 = features.load_sources(root)
    exporters = scope.exporter_scope(sources.country_year, uk=baci.UK)
    # Only the panel's own years' files are opened: the test year (2023) is never read.
    files = [baci.flows_dir(root) / f"flows_{year}.parquet" for year in years]
    flows = pl.scan_parquet(files).with_columns(
        pl.col(side).cast(pl.String).replace_strict(to_iso3, return_dtype=pl.String) for side in ("exporter", "importer")
    )
    return sector_flows(flows, sources.agreements, years=years, exporters=exporters)


def model_agreement_effect(baseline: pl.DataFrame, baseline_features: pl.DataFrame) -> dict:
    """The model's mean Trade Agreement Scenario Effect over every UK pair without an agreement.

    Read from the saved Blocked Opportunity values (`blocked_opportunity_value_usd`, scored out-of-fold,
    ADR 0001): a Trade Agreement added, with the tariff removed as every Scenario does
    (ADR 0006), against the 2024 Baseline Export Potential. `effect_pct` weights pairs by their
    Baseline, as PPML weights pairs by their Flows; the per-pair mean and median weight each
    pair equally.
    """
    pairs = (
        baseline_features.filter(pl.col("trade_agreement") == 0)
        .select("hs6", "iso3")
        .join(baseline.select("hs6", "iso3", "baseline_export_potential_usd", "blocked_opportunity_value_usd"), on=["hs6", "iso3"])
        .filter(pl.col("baseline_export_potential_usd") > 0)
        .with_columns(pair_pct=pl.col("blocked_opportunity_value_usd") / pl.col("baseline_export_potential_usd") * 100)
    )
    baseline_usd = float(pairs["baseline_export_potential_usd"].sum())
    effect_usd = float(pairs["blocked_opportunity_value_usd"].sum())
    return {
        "pairs": pairs.height,
        "baseline_usd": baseline_usd,
        "effect_usd": effect_usd,
        "effect_pct": effect_usd / baseline_usd * 100 if baseline_usd > 0 else float("nan"),
        "mean_pair_pct": float(pairs["pair_pct"].mean()),
        "median_pair_pct": float(pairs["pair_pct"].median()),
    }


# Literature estimates of the Trade Agreement (RTA/FTA) coefficient, in log points. Sources and
# page references are in docs/research/rta-effects.md.
LITERATURE = (
    {
        "estimate": "Yotov et al. (2016): PPML, Exporter- and Importer-year effects, no pair effects",
        "coefficient": 0.191,
        "source": "Advanced Guide, ch. 1, Table 3 col. (2)",
    },
    {
        "estimate": "Head & Mayer (2014): median of 108 structural gravity estimates",
        "coefficient": 0.28,
        "source": "Handbook of International Economics vol. 4, Table 4",
    },
    {
        "estimate": "Head & Mayer (2014): mean of 108 structural gravity estimates",
        "coefficient": 0.36,
        "source": "Handbook of International Economics vol. 4, Table 4",
    },
    {
        "estimate": "Head & Mayer (2014): median of 257 estimates, all gravity",
        "coefficient": 0.47,
        "source": "Handbook of International Economics vol. 4, Table 4",
    },
    {
        "estimate": "Yotov et al. (2016): PPML with pair effects and intra-national Flows",
        "coefficient": 0.557,
        "source": "Advanced Guide, ch. 1, Table 3 col. (4)",
    },
    {
        "estimate": "Baier & Bergstrand (2007): an FTA about doubles members' trade after 10 years",
        "coefficient": float(np.log(2)),
        "source": "J. Int. Econ. 71(1), abstract",
    },
)


def comparison_table(fit: PpmlFit, model: dict, literature: Sequence[dict] = LITERATURE) -> pl.DataFrame:
    """The PPML coefficient, its interval, the model's mean effect and the literature, in one table.

    Every row is a proportional change in Flows (or Export Potential) associated with a Trade
    Agreement, in percent, with the log-point coefficient beside it.
    """
    coefficients = [row["coefficient"] for row in literature]
    rows = [
        {
            "estimate": "PPML benchmark: 3-way fixed effects, Sector level",
            "coefficient": fit.coefficient,
            "effect_pct": fit.effect_pct,
            "low_pct": _percent_change(fit.ci_low),
            "high_pct": _percent_change(fit.ci_high),
            "source": f"fitted once and saved with the artefact; {CI_LEVEL:.0%} interval clustered by Exporter–Importer pair",
        },
        {
            "estimate": "Model: Trade Agreement Scenario Effect, Baseline-weighted",
            "coefficient": float(np.log1p(model["effect_pct"] / 100)),
            "effect_pct": model["effect_pct"],
            "low_pct": None,
            "high_pct": None,
            "source": f"Blocked Opportunity values, {model['pairs']:,} UK pairs without an agreement",
        },
        {
            "estimate": "Model: Trade Agreement Scenario Effect, mean per pair",
            "coefficient": float(np.log1p(model["mean_pair_pct"] / 100)),
            "effect_pct": model["mean_pair_pct"],
            "low_pct": None,
            "high_pct": None,
            "source": f"same pairs, each weighted equally (median {model['median_pair_pct']:+.1f}%)",
        },
        *(
            {
                "estimate": row["estimate"],
                "coefficient": row["coefficient"],
                "effect_pct": _percent_change(row["coefficient"]),
                "low_pct": None,
                "high_pct": None,
                "source": row["source"],
            }
            for row in literature
        ),
        {
            "estimate": "Literature range (RTA estimates)",
            "coefficient": None,
            "effect_pct": None,
            "low_pct": _percent_change(min(coefficients)),
            "high_pct": _percent_change(max(coefficients)),
            "source": "lowest to highest of the estimates above",
        },
    ]
    return pl.DataFrame(
        rows,
        schema={
            "estimate": pl.String,
            "coefficient": pl.Float64,
            "effect_pct": pl.Float64,
            "low_pct": pl.Float64,
            "high_pct": pl.Float64,
            "source": pl.String,
        },
    )


def _percent_change(log_points: float) -> float:
    return float(np.expm1(log_points) * 100)


# --- The saved benchmark: fitted once and carried with the artefact (spec #66) -----------------

LEVEL = "Sector"  # the frozen specification: Sector-level Flows, the fit above


def published_range(literature: Sequence[dict] = LITERATURE) -> dict:
    """The lowest and highest published Trade Agreement estimates, as percent changes, with the estimates."""
    coefficients = [row["coefficient"] for row in literature]
    return {
        "low_pct": _percent_change(min(coefficients)),
        "high_pct": _percent_change(max(coefficients)),
        "estimates": [{**row, "effect_pct": _percent_change(row["coefficient"])} for row in literature],
    }


def model_agreement_association(bundle: "Artefacts") -> dict:
    """The model's Trade Agreement association for UK pairs, on the artefact's served Model Form.

    Every UK pair without a Trade Agreement is re-predicted with one added (its tariff removed,
    ADR 0006) through `prediction.export_potential` with `bundle.model_form`, each pair by the
    fold model that held it out (ADR 0001), and summarised by `model_agreement_effect`.
    """
    from exportspredictor import scenarios

    form = bundle.model_form
    values = scenarios.blocked_opportunity_values(bundle.baseline_features, bundle.baseline, bundle.fold_models, form=form)
    baseline = scenarios.with_blocked_opportunity_values(bundle.baseline, values)
    return {**model_agreement_effect(baseline, bundle.baseline_features), "model_form": form.to_dict()}


def benchmark(panel: pl.DataFrame, bundle: "Artefacts") -> dict:
    """The PPML benchmark of an artefact, JSON-ready: what `Artefacts.ppml_benchmark` holds.

    `ppml` is the fit on the Sector `panel` (`PpmlFit.as_dict`: the effect and its interval),
    `published_range` the published estimates it is compared with, and `model` the model's own
    Trade Agreement association on the served Model Form (`model_agreement_association`).
    """
    years = sorted(int(y) for y in panel["year"].unique())
    return {
        "level": LEVEL,
        "years": years,
        "ci_level": CI_LEVEL,
        "ppml": fit_ppml(panel).as_dict(),
        "published_range": published_range(),
        "model": model_agreement_association(bundle),
    }


def add_benchmark(root: Path, *, panel: pl.DataFrame | None = None, version: str | None = None) -> Path:
    """Fit the PPML benchmark and save it with a new artefact version derived from the newest one.

    The fold models, the tables and the evaluation report are copied byte for byte
    (`artefacts.copy_version`): nothing is trained or re-evaluated. `panel` defaults to the
    real Sector panel (`real_sector_flows`, which reads only `YEARS`). The new version's model
    card is rewritten, so it reads the saved benchmark. Returns the new folder.
    """
    from exportspredictor import artefacts as store
    from exportspredictor import governance

    source = store.latest_folder(root)
    bundle = store.load(source)
    started = time.perf_counter()
    panel = real_sector_flows() if panel is None else panel
    saved = {**benchmark(panel, bundle), "seconds": round(time.perf_counter() - started, 2)}
    metadata = {**bundle.metadata, "version": version or store.new_version(), "ppml_benchmark_from": bundle.version}
    target = store.copy_version(source, root, metadata=metadata, ppml_benchmark=saved)
    governance.write_model_card(store.load(target), target)
    return target


def main() -> None:
    """`python -m exportspredictor.ppml`: derive a version of the newest artefact with the PPML benchmark."""
    from exportspredictor.paths import artefacts_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(add_benchmark(artefacts_dir()), flush=True)


def fit_ppml(
    panel: pl.DataFrame,
    *,
    outcome: str = OUTCOME,
    regressor: str = REGRESSOR,
    fixed_effects: tuple[tuple[str, ...], ...] = FIXED_EFFECTS,
    cluster: tuple[str, ...] = CLUSTER,
    tol: float = 1e-8,
    max_iter: int = 200,
    demean_tol: float = 1e-9,
    max_sweeps: int = 50_000,
) -> PpmlFit:
    """PPML of `outcome` on one `regressor` with the high-dimensional `fixed_effects` absorbed.

    Rows that carry no information are dropped first, as ppmlhdfe does: rows in a fixed-effect
    group whose outcomes are all zero (their effect would be minus infinity) and singleton
    groups (fitted exactly by their own effect), repeated until none remain.
    """
    start = time.perf_counter()
    y_all = panel[outcome].to_numpy().astype(np.float64)
    x_all = panel[regressor].to_numpy().astype(np.float64)
    codes_all = [_codes(panel, columns) for columns in fixed_effects]
    cluster_all = _codes(panel, cluster)

    keep = _informative(y_all, codes_all)
    y = y_all[keep] / y_all[keep].mean()  # beta and its standard error do not depend on the scale
    x = x_all[keep]
    codes = [np.unique(c[keep], return_inverse=True)[1] for c in codes_all]
    clusters = np.unique(cluster_all[keep], return_inverse=True)[1]

    mu = (y + y.mean()) / 2
    eta = np.log(mu)
    deviance = np.inf
    z_prev = z_tilde = x_tilde = None
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        z = eta + (y - mu) / mu
        z_start = z if z_tilde is None else z_tilde + (z - z_prev)
        z_tilde = _demean(z_start, codes, mu, demean_tol, max_sweeps)
        x_tilde = _demean(x if x_tilde is None else x_tilde, codes, mu, demean_tol, max_sweeps)
        beta = float(np.sum(mu * x_tilde * z_tilde) / np.sum(mu * x_tilde * x_tilde))
        eta = z - (z_tilde - x_tilde * beta)
        mu = np.exp(eta)
        z_prev = z
        new_deviance = _deviance(y, mu)
        change = abs(new_deviance - deviance) / max(new_deviance, 0.1)
        deviance = new_deviance
        _LOG.info("iteration %d: beta %.6f, deviance change %.1e, %.0f s", iteration, beta, change, time.perf_counter() - start)
        if change < tol:
            converged = True
            break

    x_tilde = _demean(x_tilde, codes, mu, demean_tol, max_sweeps)
    bread = float(np.sum(mu * x_tilde * x_tilde))
    scores = np.bincount(clusters, weights=x_tilde * (y - mu))
    n_clusters = scores.size
    meat = float(np.sum(scores**2)) * n_clusters / (n_clusters - 1)
    return PpmlFit(
        coefficient=beta,
        std_error=float(np.sqrt(meat) / bread),
        rows=int(keep.sum()),
        dropped_rows=int((~keep).sum()),
        clusters=n_clusters,
        iterations=iteration,
        converged=converged,
        seconds=time.perf_counter() - start,
    )


def _codes(panel: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Dense integer codes 0..G-1 for the groups of `columns`."""
    keys = panel.select(pl.struct(columns).hash(seed=0)).to_series().to_numpy()
    return np.unique(keys, return_inverse=True)[1]


def _informative(y: np.ndarray, codes: list[np.ndarray]) -> np.ndarray:
    keep = np.ones(y.size, dtype=bool)
    while True:
        before = keep.sum()
        for c in codes:
            totals = np.bincount(c[keep], weights=y[keep], minlength=c.max() + 1)
            counts = np.bincount(c[keep], minlength=c.max() + 1)
            keep &= (totals[c] > 0) & (counts[c] > 1)
        if keep.sum() == before:
            return keep


def _demean(
    values: np.ndarray, codes: list[np.ndarray], weights: np.ndarray, tol: float, max_sweeps: int
) -> np.ndarray:
    """`values` with every fixed effect swept out (weighted alternating projections)."""
    v = values.copy()
    totals = [np.bincount(c, weights=weights) for c in codes]
    for _ in range(max_sweeps):
        largest = 0.0
        for c, total in zip(codes, totals, strict=True):
            means = np.bincount(c, weights=weights * v, minlength=total.size) / total
            v -= means[c]
            largest = max(largest, float(np.abs(means).max()))
        if largest < tol:
            break
    return v


def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
    ratio = np.where(y > 0, y / mu, 1.0)
    return float(2 * np.sum(np.where(y > 0, y * np.log(ratio), 0.0) - (y - mu)))


if __name__ == "__main__":
    main()
