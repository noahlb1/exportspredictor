"""Lift and Event Backtests: Opportunities and Scenario Effects checked against what actually happened.

**Lift** asks whether large Export Gaps point at Flows that later grow. A model is trained on
the early years (2012–2017) with 20% of Exporters held out entirely, so their Flows are as
unseen as the UK's are to its fold models (ADR 0001). Their Export Gaps in the last training
year (2017) are ranked, and Lift is how many times more often the top tenth of pairs by
Export Gap **grew materially** by the outcome year (2023) than pairs picked at random:

    Lift = P(grew | top decile by Export Gap) / P(grew | any pair)

A pair grows materially when its Flow rises by at least half and by at least $100,000, so a
pair that appears from nothing counts once it reaches $100,000. Sampled zero cells count by
their weight. The 95% confidence interval comes from a bootstrap that resamples whole
Exporters, because one Exporter's pairs rise and fall together. Large gaps tend to belong to
large pairs, which grow by $100,000 more easily, so a **size-matched** Lift compares the top
decile with pairs of similar Export Potential instead of with all pairs. Lift is a tendency
over many pairs, not a promise about any one Opportunity.

**Event Backtests** check Scenario Effects against a real Trade Agreement. The primary one
is the EU–Japan Economic Partnership Agreement (in force 1 February 2019): a model trained
on 2012–2018 only, a period before the agreement, predicts the post-period Flows with the
agreement on and off. The model's Scenario Effect is compared with an observed
difference-in-differences: the change in EU exports to Japan from the pre-period
(2016–2018) to each post year, relative to the change in EU exports to similar Markets
with no Trade Agreement with the EU (the control group). Japan's exports to the EU are
checked the same way. COVID hit 2020, and differently by Market, so the 2020 and 2021
comparisons are noisier. The UK–Australia and UK–New Zealand agreements (May 2023) are a
secondary check with one post year, 2024, scored by the test-year fold models of
`evaluation` (trained on 2012–2022), each UK pair by the model that held it out.

No Event Backtest model reads a year from its event onwards: the training data are cut to
the event's training years before they reach XGBoost, and `train_backtest_model` refuses
any later year.

`uv run python -m exportspredictor.backtests` adds both results to the newest artefact's
evaluation report as a new version. Trained models are cached under the data root.
"""

import hashlib
import math
from collections.abc import Iterable

import numpy as np
import polars as pl

HELD_OUT_SHARE = 0.2
TOP_DECILE = 1
DECILES = 10
MIN_GROWTH_SHARE = 0.5
MIN_GROWTH_USD = 100_000.0
CI_LEVEL = 0.95
RESAMPLES = 1000


# --- Lift ------------------------------------------------------------------------------


def held_out_exporters(exporters: Iterable[str], *, share: float = HELD_OUT_SHARE, seed: int = 0) -> list[str]:
    """A deterministic `share` of the Exporters (at least one), chosen by a hash of each code."""
    codes = sorted(set(exporters))
    count = max(1, round(share * len(codes)))

    def order(code: str) -> bytes:
        return hashlib.sha256(f"{seed}|{code}".encode()).digest()

    return sorted(sorted(codes, key=order)[:count])


def grows_materially(before: str, after: str, *, min_share: float = MIN_GROWTH_SHARE, min_usd: float = MIN_GROWTH_USD) -> pl.Expr:
    """A Flow grew materially: up by at least `min_share` of its earlier value and at least `min_usd`."""
    rise = pl.col(after) - pl.col(before)
    return (rise >= min_usd) & (rise >= min_share * pl.col(before)) & (rise > 0)


def _with_decile(frame: pl.DataFrame, by: str, name: str) -> pl.DataFrame:
    """Add the weighted decile of `by` (1 = largest), each row placed at the midpoint of its weight."""
    weight = pl.col("weight")
    cum = weight.cum_sum() - weight / 2
    decile = ((cum / weight.sum() * DECILES).floor().cast(pl.Int32) + 1).clip(1, DECILES)
    return frame.sort(by, descending=True, maintain_order=True).with_columns(decile.alias(name))


def lift(pairs: pl.DataFrame, *, resamples: int = RESAMPLES, seed: int = 0, level: float = CI_LEVEL) -> dict:
    """Lift of the top decile by Export Gap, with a bootstrap interval over Exporters.

    `pairs` has exporter, export_gap_usd, export_potential_usd, grew (bool) and weight.
    Also returns the growth rate in every Export Gap decile and the size-matched Lift
    (against pairs in the same Export Potential decile).
    """
    frame = _with_decile(pairs, "export_gap_usd", "gap_decile")
    frame = _with_decile(frame, "export_potential_usd", "size_decile")
    frame = frame.with_columns(
        _w=pl.col("weight").cast(pl.Float64),
        _g=pl.col("weight").cast(pl.Float64) * pl.col("grew").cast(pl.Float64),
        _top=pl.col("gap_decile") == TOP_DECILE,
    )
    exporters = sorted(frame["exporter"].cast(pl.String).unique().to_list())
    index = {code: i for i, code in enumerate(exporters)}
    e = frame["exporter"].cast(pl.String).replace_strict(index, return_dtype=pl.Int64).to_numpy()
    size = frame["size_decile"].to_numpy() - 1
    w, g, top = frame["_w"].to_numpy(), frame["_g"].to_numpy(), frame["_top"].to_numpy()

    # Per-Exporter sums, by size decile: all pairs and top-decile pairs.
    shape = (len(exporters), DECILES)
    all_w, all_g, top_w, top_g = (np.zeros(shape) for _ in range(4))
    np.add.at(all_w, (e, size), w)
    np.add.at(all_g, (e, size), g)
    np.add.at(top_w, (e, size), w * top)
    np.add.at(top_g, (e, size), g * top)

    def statistics(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plain and size-matched Lift for each row of Exporter resampling counts."""
        aw, ag, tw, tg = (counts @ m for m in (all_w, all_g, top_w, top_g))
        with np.errstate(divide="ignore", invalid="ignore"):
            top_rate = tg.sum(1) / tw.sum(1)
            plain = top_rate / (ag.sum(1) / aw.sum(1))
            expected = np.nansum(tw / tw.sum(1, keepdims=True) * (ag / aw), axis=1)
            matched = top_rate / expected
        return plain, matched

    plain, matched = statistics(np.ones((1, len(exporters))))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(exporters), [1 / len(exporters)] * len(exporters), size=resamples)
    boot_plain, boot_matched = statistics(counts)

    rates = (
        frame.group_by("gap_decile")
        .agg(pairs=pl.col("_w").sum(), rate=pl.col("_g").sum() / pl.col("_w").sum())
        .sort("gap_decile")
    )
    total_w, total_g = float(w.sum()), float(g.sum())
    top_rate = float(top_g.sum() / top_w.sum()) if top_w.sum() else None
    return {
        "pairs": total_w,
        "sampled_rows": frame.height,
        "exporters": len(exporters),
        "top_rate": top_rate,
        "base_rate": total_g / total_w if total_w else None,
        "lift": _finite(plain[0]),
        "ci": _interval(boot_plain, level),
        "ci_level": level,
        "bootstrap_unit": "Exporter",
        "resamples": resamples,
        "size_matched": {
            "expected_rate": _finite(top_rate / matched[0]) if top_rate is not None and matched[0] else None,
            "lift": _finite(matched[0]),
            "ci": _interval(boot_matched, level),
        },
        "by_decile": [
            {"decile": row["gap_decile"], "rate": row["rate"], "pairs": row["pairs"]} for row in rates.iter_rows(named=True)
        ],
    }


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _interval(values: np.ndarray, level: float) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [None, None]
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(finite, [tail, 100 - tail])
    return [float(low), float(high)]


# --- Event Backtest: observed and predicted effects --------------------------------------


def control_markets(
    rows: pl.DataFrame,
    *,
    exporters: Iterable[str],
    treated: Iterable[str],
    pre_years: Iterable[int],
    high_income: Iterable[str] | None,
    n: int,
) -> list[str]:
    """The control group: similar Markets with no Trade Agreement with the treated Exporters.

    `rows` hold the treated Exporters' Flows (exporter, importer, year, value_usd,
    trade_agreement) over every year of the window. A control Market is not treated, not
    one of the Exporters, has no Trade Agreement with any of them in any window year, is
    high income (if `high_income` is given), and is among the `n` largest by the
    pre-period imports from the Exporters.
    """
    exporters, treated = [str(c) for c in exporters], [str(c) for c in treated]
    frame = rows.with_columns(pl.col("exporter", "importer").cast(pl.String)).filter(
        pl.col("exporter").is_in(exporters) & ~pl.col("importer").is_in([*treated, *exporters])
    )
    if high_income is not None:
        frame = frame.filter(pl.col("importer").is_in(sorted(high_income)))
    ranked = (
        frame.group_by("importer")
        .agg(
            agreement=pl.col("trade_agreement").max(),
            pre_imports=pl.col("value_usd").filter(pl.col("year").is_in(list(pre_years))).sum(),
        )
        .filter((pl.col("agreement") == 0) & (pl.col("pre_imports") > 0))
        .sort("pre_imports", "importer", descending=[True, False])
    )
    return ranked["importer"].head(n).to_list()


def compare(
    flows: pl.DataFrame,
    predicted: pl.DataFrame,
    *,
    exporters: Iterable[str],
    treated: Iterable[str],
    controls: Iterable[str],
    pre_years: Iterable[int],
    post_years: Iterable[int],
    resamples: int = RESAMPLES,
    seed: int = 0,
    level: float = CI_LEVEL,
) -> dict:
    """The model's Scenario Effect against the observed difference-in-differences, per post year.

    `flows` hold Flows (exporter, importer, hs6, year, value_usd). Observed effect in a post
    year = (treated total / pre-period mean) / (control total / pre-period mean) - 1, where
    treated is the Exporters' Flows to the treated Markets and control their Flows to the
    control Markets. `predicted` holds the treated rows of the post years with the
    agreement on (`on_usd`) and off (`off_usd`) and a `weight`; the predicted effect is
    sum(on) / sum(off) - 1. Intervals come from a bootstrap over Products. `pooled` uses
    the mean of the post years.
    """
    pre, post = sorted(pre_years), sorted(post_years)
    treated, controls = [str(c) for c in treated], [str(c) for c in controls]
    active = flows.with_columns(pl.col("exporter", "importer", "hs6").cast(pl.String)).filter(
        pl.col("exporter").is_in([str(c) for c in exporters])
        & pl.col("year").is_in([*pre, *post])
        & (pl.col("value_usd") > 0)
    )
    group = pl.when(pl.col("importer").is_in(treated)).then(pl.lit("treated")).when(
        pl.col("importer").is_in(controls)
    ).then(pl.lit("control"))
    sums = (
        active.with_columns(group=group)
        .filter(pl.col("group").is_not_null())
        .group_by("hs6", "year", "group")
        .agg(value=pl.col("value_usd").cast(pl.Float64).sum())
    )
    model = predicted.with_columns(pl.col("hs6").cast(pl.String)).group_by("hs6", "year").agg(
        on=(pl.col("on_usd") * pl.col("weight")).sum(), off=(pl.col("off_usd") * pl.col("weight")).sum()
    )
    products = sorted(set(sums["hs6"].to_list()) | set(model["hs6"].to_list()))
    p_index = {p: i for i, p in enumerate(products)}
    years = [*pre, *post]
    y_index = {y: i for i, y in enumerate(years)}

    def matrix(frame: pl.DataFrame, column: str) -> np.ndarray:
        out = np.zeros((len(products), len(years)))
        for hs6, year, value in frame.select("hs6", "year", column).iter_rows():
            if year in y_index:
                out[p_index[hs6], y_index[year]] += value
        return out

    t = matrix(sums.filter(pl.col("group") == "treated"), "value")
    c = matrix(sums.filter(pl.col("group") == "control"), "value")
    on, off = matrix(model, "on"), matrix(model, "off")
    n_pre = len(pre)

    def effects(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Observed and predicted effects (resamples × [post years..., pooled])."""
        tt, cc, oo, ff = (counts @ m for m in (t, c, on, off))
        t_post = np.column_stack([tt[:, n_pre:], tt[:, n_pre:].mean(1)])
        c_post = np.column_stack([cc[:, n_pre:], cc[:, n_pre:].mean(1)])
        on_post = np.column_stack([oo[:, n_pre:], oo[:, n_pre:].sum(1)])
        off_post = np.column_stack([ff[:, n_pre:], ff[:, n_pre:].sum(1)])
        with np.errstate(divide="ignore", invalid="ignore"):
            observed = (t_post / tt[:, :n_pre].mean(1, keepdims=True)) / (c_post / cc[:, :n_pre].mean(1, keepdims=True)) - 1
            model_effect = on_post / off_post - 1
        return observed, model_effect

    observed, model_effect = effects(np.ones((1, len(products))))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(products), [1 / len(products)] * len(products), size=resamples)
    boot_observed, boot_model = effects(counts)

    def entry(j: int) -> dict:
        return {
            "observed": _finite(observed[0, j]),
            "observed_ci": _interval(boot_observed[:, j], level),
            "predicted": _finite(model_effect[0, j]),
            "predicted_ci": _interval(boot_model[:, j], level),
        }

    return {
        "by_year": [{"year": year, **entry(j)} for j, year in enumerate(post)],
        "pooled": entry(len(post)),
        "products": len(products),
        "treated_pre_usd": float(t[:, :n_pre].sum(0).mean()),
        "control_pre_usd": float(c[:, :n_pre].sum(0).mean()),
        "ci_level": level,
        "bootstrap_unit": "Product",
        "resamples": resamples,
    }
