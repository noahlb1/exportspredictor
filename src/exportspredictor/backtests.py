"""Lift and Event Backtests: Opportunities and Scenario Effects checked against what actually happened.

**Lift** asks whether large Export Gaps point at Flows that later grow. A model is trained on
the early years (2012–2017) without v1's held-out Exporters (its frozen list, restricted to
v2's Exporter scope), so their Flows are as unseen as the UK's are to its fold models (ADR
0001), and v1 and v2 are scored on the same countries. Their pairs are ranked by Export Gap
over a Gap Period (2015–2017: the mean of the yearly gaps, as the Baseline's Export Gap is
the mean over 2022–2024), and Lift is how many times more often the top tenth of pairs by
Export Gap **grew materially** by the outcome year (2023) than pairs picked at random:

    Lift = P(grew | top decile by Export Gap) / P(grew | any pair)

A pair grows materially when its outcome-year Flow is at least half and at least $100,000
above its Gap Period mean Actual Exports, so a pair that appears from nothing counts once it
reaches $100,000. Sampled zero cells count by their weight. The 95% confidence interval comes from a bootstrap that resamples whole
Exporters, because one Exporter's pairs rise and fall together. Large gaps tend to belong to
large pairs, which grow by $100,000 more easily, so a **size-matched** Lift compares the top
decile with pairs of similar Export Potential instead of with all pairs. Lift is a tendency
over many pairs, not a promise about any one Opportunity.

**Event Backtests** check Scenario Effects against a real Trade Agreement. The primary one
is the EU–Japan Economic Partnership Agreement (in force 1 February 2019): a model trained
on 2012–2018 only, a period before the agreement, predicts the post-period Flows with the
agreement on and off: on sets the Trade Agreement flag and zeroes the Applied Tariff,
off clears the flag and restores the Importer's MFN rate (ADR 0006). The model's
Scenario Effect is compared with an observed difference-in-differences: the change in EU
exports to Japan from the pre-period (2016–2018) to each post year, relative to the change in EU exports to similar Markets
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
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor import ledger
from exportspredictor.features import FEATURES, TARIFF_FEATURES, FeatureTransform
from exportspredictor.prediction import (
    MODEL_ALONE,
    NAIVE_COLUMN,
    ModelForm,
    blend,
    export_potential,
)
from exportspredictor.scenarios import AGREEMENT_TARIFF
from exportspredictor.training import (
    Chunk,
    model_params,
    preparer,
    read_rows,
    scan_chunks,
    training_matrix,
)
from exportspredictor.wording import percent_text, years_text

if TYPE_CHECKING:
    from exportspredictor.evaluation import TestModels

MODEL_FILE = "backtest_model.json"
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
        "lift": finite(plain[0]),
        "ci": interval(boot_plain, level),
        "ci_level": level,
        "bootstrap_unit": "Exporter",
        "resamples": resamples,
        "size_matched": {
            "expected_rate": finite(top_rate / matched[0]) if top_rate is not None and matched[0] else None,
            "lift": finite(matched[0]),
            "ci": interval(boot_matched, level),
        },
        "by_decile": [
            {"decile": row["gap_decile"], "rate": row["rate"], "pairs": row["pairs"]} for row in rates.iter_rows(named=True)
        ],
    }


def finite(value) -> float | None:
    """`value` as a plain float for a saved report, or None where it is missing, NaN or infinite."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def interval(values: np.ndarray, level: float) -> list[float | None]:
    """The central `level` percentile interval of the finite `values` (bootstrap resamples); [None, None] if none."""
    kept = values[np.isfinite(values)]
    if not kept.size:
        return [None, None]
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(kept, [tail, 100 - tail])
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
    exclude: Iterable[str] = (),
) -> list[str]:
    """The control group: similar Markets with no Trade Agreement with the treated Exporters.

    `rows` hold the treated Exporters' Flows (exporter, importer, year, value_usd,
    trade_agreement) over every year of the window. A control Market is not treated, not
    one of the Exporters, has no Trade Agreement with any of them in any window year, is
    high income (if `high_income` is given), and is among the `n` largest by the
    pre-period imports from the Exporters. Markets in `exclude` are never controls.
    """
    exporters, treated = [str(c) for c in exporters], [str(c) for c in treated]
    frame = rows.with_columns(pl.col("exporter", "importer").cast(pl.String)).filter(
        pl.col("exporter").is_in(exporters) & ~pl.col("importer").is_in([*treated, *exporters, *exclude])
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
            "observed": finite(observed[0, j]),
            "observed_ci": interval(boot_observed[:, j], level),
            "predicted": finite(model_effect[0, j]),
            "predicted_ci": interval(boot_model[:, j], level),
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


# --- Backtest models ------------------------------------------------------------------------


class LeakageError(AssertionError):
    """A backtest model would see, or has seen, data from its outcome or event years."""


@dataclass(frozen=True)
class BacktestModel:
    """One model trained on the early years only, optionally without some Exporters."""

    name: str
    training_years: list[int]
    held_out_exporters: list[str]
    model: xgb.Booster
    transform: FeatureTransform

    def predict(self, rows: pl.DataFrame) -> np.ndarray:
        """Export Potential (USD) for untransformed feature rows."""
        if not rows.height:
            return np.zeros(0)
        return export_potential(self.transform.apply(rows), self.model).to_numpy()

    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_model(folder / "model.ubj")
        info = {
            "name": self.name,
            "training_years": self.training_years,
            "held_out_exporters": self.held_out_exporters,
            "transform": self.transform.to_dict(),
        }
        # Written last: its presence marks the folder as complete.
        (folder / MODEL_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> "BacktestModel":
        folder = Path(folder)
        info = json.loads((folder / MODEL_FILE).read_text(encoding="utf-8"))
        model = xgb.Booster()
        model.load_model(folder / "model.ubj")
        return cls(
            info["name"], info["training_years"], info["held_out_exporters"], model,
            FeatureTransform.from_dict(info["transform"]),
        )


@dataclass(frozen=True)
class FoldPredictor:
    """Fold models (`evaluation.TestModels`) scoring UK rows out-of-fold and other rows by their mean."""

    test_models: "TestModels"
    uk: str

    @property
    def training_years(self) -> list[int]:
        return list(self.test_models.training_years)

    def predict(self, rows: pl.DataFrame) -> np.ndarray:
        from exportspredictor.evaluation import score_rows

        indexed = rows.with_row_index("_row")
        scored = score_rows(indexed, self.test_models.models, self.test_models.transform, self.uk).sort("_row")
        return scored["export_potential_usd"].to_numpy()


def train_backtest_model(
    training: Mapping[int, Chunk],
    *,
    name: str,
    training_years: Sequence[int],
    before: int,
    uk: str,
    params: dict,
    rounds: int,
    device: str = "cuda",
    non_uk_rate: float = 1.0,
    seed: int = 0,
    held_out: Sequence[str] = (),
) -> BacktestModel:
    """Train one model on `training_years` only, without the `held_out` Exporters' Flows.

    Raises LeakageError if any training year is `before` or later, so a model never sees
    its event or outcome years. Only the training years' chunks are ever read. Hyperparameters
    and the non-UK sample are those of the Baseline models.
    """
    years = sorted(training_years)
    if not years or max(years) >= before:
        raise LeakageError(f"Backtest model {name!r} would train on {years}, not only on years before {before}")
    chunks = [training[y] for y in years]
    held = sorted(held_out)
    kept = ~pl.col("exporter").cast(pl.String).is_in(held)

    def keep(table: pl.DataFrame) -> pl.DataFrame:
        return table.filter(kept)

    transform = FeatureTransform.fit(scan_chunks(chunks).filter(kept), years)
    prepare = preparer(uk=uk, transform=transform, non_uk_rate=non_uk_rate, seed=seed, keep=keep)
    matrix = training_matrix(chunks, prepare, device=device)
    model = xgb.train(model_params(params, device=device, seed=seed), matrix, num_boost_round=rounds)
    del matrix
    model.set_param({"device": "cpu"})
    return BacktestModel(name, years, held, model, transform)


# --- Lift on held-out Exporters ----------------------------------------------------------


@dataclass(frozen=True)
class LiftSettings:
    """When the Lift model trains, which years form its Gap Period, and when growth is measured.

    The Lift model may not train on a year after the Gap Period, nor on the outcome year:
    the Gap Period must end before the outcome year and every training year must fall
    on or before the Gap Period's last year (else LeakageError).
    """

    training_years: tuple[int, ...]
    gap_years: tuple[int, ...]
    outcome_year: int
    min_growth_share: float = MIN_GROWTH_SHARE
    min_growth_usd: float = MIN_GROWTH_USD

    def __post_init__(self) -> None:
        if not self.gap_years or not self.training_years:
            raise ValueError("Lift needs training years and a Gap Period")
        if max(self.gap_years) >= self.outcome_year:
            raise LeakageError(f"The Gap Period {list(self.gap_years)} must end before {self.outcome_year}")
        if max(self.training_years) > max(self.gap_years):
            raise LeakageError(
                f"The Lift model would train on {list(self.training_years)}, after the Gap Period {list(self.gap_years)}"
            )

    @property
    def before(self) -> int:
        """The first year the Lift model may not train on."""
        return max(self.gap_years) + 1


def lift_exporters(v1_held_out: Iterable[str], *, scope: Iterable[str], uk: str) -> list[str]:
    """The Lift model's held-out Exporters: v1's frozen list, restricted to v2's Exporter scope.

    Both models are then scored on the same Exporters. Raises ValueError if the UK is in the
    list (it can never be held out) or if no listed Exporter is in scope.
    """
    listed = {str(code) for code in v1_held_out}
    if uk in listed:
        raise ValueError(f"{uk} cannot be a held-out Exporter")
    held = sorted(listed & {str(code) for code in scope})
    if not held:
        raise ValueError("None of v1's held-out Exporters is in v2's Exporter scope")
    return held


_PAIR = ["exporter", "importer", "hs6"]


def gap_period_pairs(yearly: pl.DataFrame, later: pl.DataFrame, settings: LiftSettings) -> pl.DataFrame:
    """Each pair's Gap Period Export Gap and whether it grew materially by the outcome year.

    `yearly` holds training rows of the Gap Period years (exporter, importer, hs6, year,
    value_usd, weight, export_potential_usd); `later` holds the outcome year's positive Flows
    (exporter, importer, hs6, value_usd). The pairs, and their weights, are the rows of the
    Gap Period's last year: positive Flows and sampled Zero Cells, so weighted counts stay
    unbiased. A pair's mean Actual Exports is over every Gap Period year, a missing row
    counting as 0 (every positive Flow is a row). Its Export Potential is the mean over the
    years it has a row (Zero Cells are sampled each year apart), so its Export Gap is the mean
    of its yearly gaps whenever it has a row in every year. It grew materially when its
    outcome-year Flow rose from the mean Actual Exports by at least `min_growth_share` and
    `min_growth_usd`. The same rule scores v1's frozen Lift rows.
    """
    years = list(settings.gap_years)
    rows = yearly.with_columns(pl.col(_PAIR).cast(pl.String)).filter(pl.col("year").is_in(years))
    anchor = rows.filter(pl.col("year") == max(years)).select(*_PAIR, "weight")
    means = rows.group_by(_PAIR).agg(
        actual_exports_usd=pl.col("value_usd").cast(pl.Float64).sum() / len(years),
        export_potential_usd=pl.col("export_potential_usd").cast(pl.Float64).mean(),
    )
    outcome = later.select(pl.col(_PAIR).cast(pl.String), later_usd=pl.col("value_usd").cast(pl.Float64))
    return (
        anchor.join(means, on=_PAIR, how="left")
        .join(outcome, on=_PAIR, how="left")
        .with_columns(pl.col("later_usd").fill_null(0.0))
        .with_columns(
            export_gap_usd=pl.col("export_potential_usd") - pl.col("actual_exports_usd"),
            grew=grows_materially(
                "actual_exports_usd", "later_usd", min_share=settings.min_growth_share, min_usd=settings.min_growth_usd
            ),
        )
    )


def lift_rows(
    training: Mapping[int, Chunk],
    settings: LiftSettings,
    model: BacktestModel,
    *,
    form: ModelForm = MODEL_ALONE,
    year_flows: Callable[[int], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """The held-out Exporters' rows of every Gap Period year, with the Lift model's Export Potential.

    In a blend (`form`), each row's Export Potential blends the Lift model with the row's naive
    baseline (`evaluation.naive_potential`, kept as `naive_usd`), computed from every Exporter's
    Flows of the row's own year, `year_flows(year)`: the Lift model's own form of the served model.
    A form with Market Reach then scales each held-out Exporter's rows by its own Market Reach, learned
    from its own rows of the year before (`market_reach.exporter_reach`, at the form's beta), as the
    served form scales UK pairs by the UK's; `training` must hold that year (ValueError otherwise).
    """
    from exportspredictor import market_reach
    from exportspredictor.evaluation import naive_potential

    _check_lift_model(model, settings)
    if form.naive_weight and year_flows is None:
        raise ValueError("A blend's Lift rows need each Gap Period year's Flows for the naive baseline")
    reach = form.market_reach
    if reach is not None:
        missing = [year - reach.lag_years for year in settings.gap_years if year - reach.lag_years not in training]
        if missing:
            raise ValueError(f"Lift with Market Reach needs the training rows of {missing}, the years before the Gap Period's")
    is_held = pl.col("exporter").cast(pl.String).is_in(model.held_out_exporters)
    frames = []
    for year in settings.gap_years:
        rows = read_rows(training[year], is_held)
        potential = model.predict(rows)
        kept = [pl.col(_PAIR).cast(pl.String), pl.col("year").cast(pl.Int16), "value_usd", "weight"]
        if form.naive_weight:
            rows = naive_potential(rows, year_flows(year))
            potential = blend(potential, rows[NAIVE_COLUMN].to_numpy(), form.naive_weight)
            kept.append(NAIVE_COLUMN)
        if reach is not None:
            source = read_rows(training[year - reach.lag_years], is_held).select(
                "exporter", "hs6", "importer", "value_usd", market_reach.DEMAND, "weight",
                year=pl.lit(year - reach.lag_years),
            )
            potential = market_reach.exporter_reach(potential, rows, source, scored_year=year, beta=reach.beta)
        frames.append(
            rows.select(kept).with_columns(export_potential_usd=pl.Series(potential, dtype=pl.Float64))
        )
    return pl.concat(frames, how="vertical_relaxed")


def _check_lift_model(model: BacktestModel, settings: LiftSettings) -> None:
    if max(model.training_years) > max(settings.gap_years):
        raise LeakageError(f"The Lift model trained on {model.training_years}, after the Gap Period")
    if not model.held_out_exporters:
        raise ValueError("The Lift model holds out no Exporters")


def lift_backtest(
    training: Mapping[int, Chunk],
    settings: LiftSettings,
    model: BacktestModel,
    *,
    resamples: int = RESAMPLES,
    seed: int = 0,
    form: ModelForm = MODEL_ALONE,
    year_flows: Callable[[int], pl.DataFrame] | None = None,
) -> dict:
    """Raw and size-matched Lift of the held-out Exporters' top-decile Gap Period Export Gaps, by the outcome year.

    The Export Potential is in the model form `form` (`lift_rows`); the result records it (with Market Reach,
    its settings: each held-out Exporter's tables and constants are fitted per year, `market_reach.exporter_reach`).
    """
    yearly = lift_rows(training, settings, model, form=form, year_flows=year_flows)
    is_held = pl.col("exporter").cast(pl.String).is_in(model.held_out_exporters)
    later = read_rows(training[settings.outcome_year], is_held & (pl.col("value_usd") > 0)).select(*_PAIR, "value_usd")
    result = lift(gap_period_pairs(yearly, later, settings), resamples=resamples, seed=seed)
    result.update(
        held_out_exporters=model.held_out_exporters,
        held_out_from="v1",
        training_years=model.training_years,
        gap_years=list(settings.gap_years),
        outcome_year=settings.outcome_year,
        min_growth_share=settings.min_growth_share,
        min_growth_usd=settings.min_growth_usd,
        model_form=_lift_form(form),
    )
    result["text"] = lift_text(result)
    return result


def _lift_form(form: ModelForm) -> dict:
    """The Model Form a Lift result records: with Market Reach, its settings, not one fitted table and constant."""
    from exportspredictor import market_reach

    plain = form.without_market_reach().to_dict()
    return plain if form.market_reach is None else {**plain, "market_reach": market_reach.settings(form.market_reach.beta)}


def lift_text(result: dict) -> str:
    """The plain-English Lift result, with what the size-matched comparison adds, worded as a tendency."""
    lift_value, (low, high) = result["lift"], result["ci"]
    if lift_value is None:
        return "Lift could not be measured: no held-out pair grew materially."
    interval = "" if low is None else f" (95% confidence interval {low:.1f} to {high:.1f})"
    growth = f"rose by at least {result['min_growth_share']:.0%} and ${result['min_growth_usd']:,.0f}"
    matched, (m_low, m_high) = result["size_matched"]["lift"], result["size_matched"]["ci"]
    if matched is None:
        size = ""
    else:
        if m_low is not None and m_high < 1:
            verdict = "less often than pairs of similar Export Potential"
            meaning = "Much of the Lift comes from size: large gaps sit on large pairs, which grow by $100,000 more easily."
        elif m_low is not None and m_low > 1:
            verdict = "more often than pairs of similar Export Potential"
            meaning = "So the Export Gap points at growth beyond what size alone explains."
        else:
            verdict = "about as often as pairs of similar Export Potential"
            meaning = "So the Export Gap adds little beyond size."
        m_interval = "" if m_low is None else f", 95% interval {m_low:.2f} to {m_high:.2f}"
        size = f" They grew {verdict} (ratio {matched:.2f}{m_interval}). {meaning}"
    return (
        f"For countries the model never trained on, the tenth of Product–Market pairs with the largest Export Gaps "
        f"averaged over {years_text(result['gap_years'])} {growth} from their {years_text(result['gap_years'])} mean by "
        f"{result['outcome_year']} {lift_value:.1f} times as often as pairs "
        f"picked at random{interval}: {result['top_rate']:.0%} of them against {result['base_rate']:.0%}.{size} "
        f"This is a tendency across many pairs, not a promise that any one Opportunity will grow."
    )


# --- Event Backtests ----------------------------------------------------------------------------

(TARIFF,) = TARIFF_FEATURES


@dataclass(frozen=True)
class Direction:
    label: str
    exporters: tuple[str, ...]
    importers: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    name: str
    title: str
    in_force: str
    event_year: int  # first year the Trade Agreement is in force in the policy features
    training_years: tuple[int, ...]
    pre_years: tuple[int, ...]
    post_years: tuple[int, ...]
    directions: tuple[Direction, ...]
    caveat: str
    n_controls: int = 10
    excluded_controls: tuple[str, ...] = ()  # Markets that may not serve as controls, e.g. under sanctions


def event_backtest(
    training: Mapping[int, Chunk],
    event: Event,
    model,
    *,
    high_income: Iterable[str] | None,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> dict:
    """Predicted against observed effect of one real Trade Agreement, for each direction.

    `model` has `training_years` and `predict(rows)` (a BacktestModel or a FoldPredictor);
    it must have trained only on years before the event. `training` must hold the event's
    pre and post years.
    """
    if max(model.training_years) >= event.event_year:
        raise LeakageError(
            f"The {event.name} backtest model trained on {model.training_years}, not only before {event.event_year}"
        )
    window = [*event.pre_years, *event.post_years]
    exporters = sorted({c for d in event.directions for c in d.exporters})
    rows = pl.concat(
        [read_rows(training[y], pl.col("exporter").cast(pl.String).is_in(exporters)) for y in window],
        how="vertical_relaxed",
    ).with_columns(pl.col("exporter", "importer").cast(pl.String))
    directions = []
    for d in event.directions:
        own = rows.filter(pl.col("exporter").is_in(list(d.exporters)))
        treated = own.filter(pl.col("importer").is_in(list(d.importers)))
        _check_event(treated, event)
        controls = control_markets(
            own, exporters=d.exporters, treated=d.importers, pre_years=event.pre_years,
            high_income=high_income, n=event.n_controls, exclude=event.excluded_controls,
        )
        if not controls:
            raise ValueError(f"No control Markets for {d.label}")
        post = treated.filter(pl.col("year").is_in(list(event.post_years)))
        on_rows, off_rows = agreement_states(post)
        on, off = model.predict(on_rows), model.predict(off_rows)
        predicted = post.select("hs6", "year", "weight").with_columns(
            on_usd=pl.Series(on, dtype=pl.Float64), off_usd=pl.Series(off, dtype=pl.Float64)
        )
        result = compare(
            own.select("exporter", "importer", "hs6", "year", "value_usd"),
            predicted,
            exporters=d.exporters,
            treated=d.importers,
            controls=controls,
            pre_years=event.pre_years,
            post_years=event.post_years,
            resamples=resamples,
            seed=seed,
        )
        directions.append(
            {"label": d.label, "exporters": list(d.exporters), "importers": list(d.importers), "controls": controls, **result}
        )
    report = {
        "name": event.name,
        "title": event.title,
        "in_force": event.in_force,
        "event_year": event.event_year,
        "training_years": list(model.training_years),
        "pre_years": list(event.pre_years),
        "post_years": list(event.post_years),
        "control_rule": _control_rule(event, high_income is not None),
        "caveat": event.caveat,
        "states": AGREEMENT_STATES,
        "directions": directions,
    }
    report["text"] = event_text(report)
    return report


AGREEMENT_STATES = (
    "With the agreement on, the Trade Agreement flag is set and a known Applied Tariff is 0 (an unknown one stays unknown); with it off, the flag is "
    "cleared and the Importer's MFN tariff is restored (the pair's own tariff where no MFN rate is known). These are "
    "the states a Trade Agreement Scenario compares (ADR 0006)."
)


def agreement_states(rows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The rows with the Trade Agreement on (a known Applied Tariff 0) and off (the Importer's MFN rate restored).

    Only policy columns change: `trade_agreement` and `log_applied_tariff`, as a Trade
    Agreement Scenario changes them (ADR 0006). `mfn_tariff` is in percent; where it is
    missing the off state keeps the row's own tariff, and an unknown tariff stays unknown in the on state.
    """
    flag = rows.schema["trade_agreement"]
    tariff = pl.col(TARIFF)
    dtype = rows.schema[TARIFF]
    mfn = (pl.col("mfn_tariff").cast(pl.Float64) / 100).log1p().cast(dtype)
    on = rows.with_columns(trade_agreement=pl.lit(1, flag), **{TARIFF: AGREEMENT_TARIFF.cast(dtype)})
    off = rows.with_columns(trade_agreement=pl.lit(0, flag), **{TARIFF: mfn.fill_null(tariff)})
    return on, off


def _check_event(treated: pl.DataFrame, event: Event) -> None:
    """The treated pairs have no Trade Agreement before the event and one from it on."""
    flags = treated.group_by(after=pl.col("year") >= event.event_year).agg(
        lo=pl.col("trade_agreement").min(), hi=pl.col("trade_agreement").max()
    )
    for after, lo, hi in flags.iter_rows():
        if (after and lo != 1) or (not after and hi != 0):
            raise ValueError(f"The policy data do not show {event.title} switching on in {event.event_year}")


def _control_rule(event: Event, high_income: bool) -> str:
    income = "high-income " if high_income else ""
    return (
        f"The {event.n_controls} largest {income}Markets by imports from the same Exporters in "
        f"{years_text(event.pre_years)} that had no Trade Agreement with any of them in any year from "
        f"{event.pre_years[0]} to {event.post_years[-1]}"
        + (f", leaving out {', '.join(event.excluded_controls)}." if event.excluded_controls else ".")
    )


def event_text(report: dict) -> str:
    """The plain-English Event Backtest result: predicted against observed, and whether they agree."""
    parts = []
    for d in report["directions"]:
        pooled = d["pooled"]
        low, high = pooled["observed_ci"]
        predicted = pooled["predicted"]
        interval = "" if low is None else f" (95% interval {percent_text(low)} to {percent_text(high)})"
        if low is None or predicted is None:
            verdict = ""
        elif predicted > high:
            verdict = " The model's effect is larger than what was observed, outside the interval."
        elif predicted < low:
            verdict = " The model's effect is smaller than what was observed, outside the interval."
        else:
            verdict = " The model's effect is within the observed interval."
        parts.append(
            f"For {d['label']} in {years_text(report['post_years'])}, the model associated the agreement with "
            f"{percent_text(predicted)} in Export Potential; the observed difference-in-differences against "
            f"{len(d['controls'])} similar Markets without an agreement was {percent_text(pooled['observed'])}{interval}.{verdict}"
        )
    return (
        f"{report['title']} (in force {report['in_force']}), with a model trained only on "
        f"{years_text(report['training_years'])}: " + " ".join(parts)
    )


# --- Running every backtest ---------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """The backtests to run: Lift, the primary Event Backtest and an optional secondary one.

    The secondary event is scored by the test-year fold models (`evaluation.TestModels`).
    """

    lift: LiftSettings
    event: Event
    secondary: Event | None = None
    high_income_controls: bool = True
    resamples: int = RESAMPLES


def run(
    training: Mapping[int, Chunk],
    plan: Plan,
    *,
    uk: str,
    params: dict,
    rounds: int,
    device: str = "cuda",
    non_uk_rate: float = 1.0,
    seed: int = 0,
    high_income: Iterable[str] | None = None,
    test_models=None,
    cache: Path | None = None,
    lift_held_out: Sequence[str],
    form: ModelForm = MODEL_ALONE,
    year_flows: Callable[[int], pl.DataFrame] | None = None,
) -> dict:
    """Train (or read from `cache`) the backtest models; return the report's `lift` and `event_backtest` keys.

    Lift is measured in the served model form `form` (a blend takes each Gap Period year's
    naive baseline from `year_flows(year)`); the Event Backtests measure the model alone. When `form`
    has Market Reach, the same Lift model is also scored without it, under `lift_without_market_reach`:
    the kept change's effect on Lift (and the Lift the pre-registered v1 comparison reads).

    `training` maps each year to its feature table (training rows) and must cover every
    year the plan reads. `lift_held_out` are the Exporters the Lift model never sees:
    v1's frozen list within v2's scope (`lift_exporters`). Models are cached under `cache`
    by their settings.
    """
    started = time.perf_counter()
    held = sorted(lift_held_out)
    if not held or uk in held:
        raise ValueError(f"The Lift model needs held-out Exporters other than {uk}, got {held}")
    spec = {"params": params, "rounds": rounds, "non_uk_rate": non_uk_rate, "seed": seed, "features": FEATURES}

    def model(name: str, years: Sequence[int], before: int, held_out: Sequence[str] = ()) -> BacktestModel:
        folder = None
        if cache is not None:
            key_spec = {**spec, "name": name, "training_years": sorted(years), "held_out": sorted(held_out)}
            key = hashlib.sha256(json.dumps(key_spec, sort_keys=True).encode()).hexdigest()[:12]
            folder = Path(cache) / f"{name}_{key}"
            if (folder / MODEL_FILE).exists():
                return BacktestModel.load(folder)
        trained = train_backtest_model(
            training, name=name, training_years=years, before=before, uk=uk, params=params, rounds=rounds,
            device=device, non_uk_rate=non_uk_rate, seed=seed, held_out=held_out,
        )
        if folder is not None:
            trained.save(folder)
        return trained

    lift_model = model("lift", plan.lift.training_years, plan.lift.before, held)
    lift_result = lift_backtest(
        training, plan.lift, lift_model, resamples=plan.resamples, seed=seed, form=form, year_flows=year_flows
    )
    without = {}
    if form.market_reach is not None:
        without["lift_without_market_reach"] = lift_backtest(
            training, plan.lift, lift_model, resamples=plan.resamples, seed=seed, form=form.without_market_reach(),
            year_flows=year_flows,
        )
    income = sorted(high_income) if plan.high_income_controls and high_income is not None else None
    event_model = model(plan.event.name, plan.event.training_years, plan.event.event_year)
    events = [
        {
            "role": "primary",
            **event_backtest(training, plan.event, event_model, high_income=income, resamples=plan.resamples, seed=seed),
        }
    ]
    if plan.secondary is not None:
        if test_models is None:
            raise ValueError("The secondary Event Backtest needs the test-year fold models")
        predictor = FoldPredictor(test_models, uk)
        secondary = event_backtest(training, plan.secondary, predictor, high_income=income, resamples=plan.resamples, seed=seed)
        events.append({"role": "secondary", **secondary})
    return {
        "lift": lift_result,
        **without,
        "event_backtest": {
            "events": events,
            "text": events[0]["text"],
            "caveat": events[0]["caveat"],
            "seconds": round(time.perf_counter() - started, 2),
        },
    }


# --- The real plan ----------------------------------------------------------------------------

# The EU's 27 members after the UK left (all members throughout 2019–2021).
EU27 = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GRC", "HRV", "HUN",
    "IRL", "ITA", "LTU", "LUX", "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
)

EU_JAPAN = Event(
    name="eu_japan",
    title="The EU–Japan Economic Partnership Agreement",
    in_force="1 February 2019",
    event_year=2019,
    training_years=tuple(range(2012, 2019)),
    pre_years=(2016, 2017, 2018),
    post_years=(2019, 2020, 2021),
    directions=(
        Direction("EU exports to Japan", EU27, ("JPN",)),
        Direction("Japan's exports to the EU", ("JPN",), EU27),
    ),
    caveat=(
        "COVID-19 disrupted trade in 2020 and 2021, unevenly across Markets, so the control Markets may not show "
        "what EU–Japan trade would otherwise have done; 2019 is the cleanest year. The comparison assumes treated "
        "and control trade would otherwise have moved in parallel. Values are nominal USD, so exchange-rate swings "
        "move them, and Japan also joined CPTPP in December 2018."
    ),
)

UK_AUSTRALIA_NZ = Event(
    name="uk_australia_nz",
    title="The UK–Australia and UK–New Zealand Free Trade Agreements",
    in_force="31 May 2023",
    event_year=2023,
    training_years=tuple(range(2012, 2023)),  # the test-year fold models of `evaluation`
    pre_years=(2021, 2022),
    post_years=(2024,),
    directions=(Direction("UK exports to Australia and New Zealand", ("GBR",), ("AUS", "NZL")),),
    # Sanctions from 2022 cut UK exports to Russia and Belarus, which would flatter the comparison.
    excluded_controls=("BLR", "RUS"),
    caveat=(
        "A secondary check with one post year: 2023 is skipped because the agreements were in force for only "
        "seven months of it, and 2024 is provisional. The pre-period starts in 2021, after the UK left the EU's "
        "single market."
    ),
)

REAL_PLAN = Plan(
    lift=LiftSettings(training_years=tuple(range(2012, 2018)), gap_years=(2015, 2016, 2017), outcome_year=2023),
    event=EU_JAPAN,
    secondary=UK_AUSTRALIA_NZ,
)


# --- Adding the backtests to a saved artefact ---------------------------------------------------


def cache_dir(root: Path | None = None) -> Path:
    """Where real backtest models are cached under the data root: apart per feature set (`features.feature_set`), as
    the models' cache keys hold their settings but not the files they were trained on."""
    from exportspredictor.features import feature_set, features_dir

    name = feature_set()
    return features_dir(root).parent / ("backtest_models" + (f"_{name}" if name else ""))


def add_to_latest(
    root: Path | None = None,
    artefacts_root: Path | None = None,
    *,
    plan: Plan = REAL_PLAN,
    device: str = "cuda",
    version: str | None = None,
    v1_frozen: Path | None = None,
    test_year_ledger: ledger.Ledger | None,
) -> Path:
    """Run the backtests for the newest evaluated artefact and save it again, with them, as a new version.

    Lift's outcome year is the test year, so the read is recorded in `test_year_ledger` first and
    refused if the newest version's configuration was read before (ADR 0009); `None` only on synthetic data.

    The Lift model holds out v1's frozen held-out Exporters within the artefact's scope, read
    from `v1_frozen` (default: the real v1 artefact's frozen tables under `artefacts_root`).

    The backtest models use the artefact's tuned parameters, round count, non-UK sample and
    seed, and are read from the cache when already trained; the secondary Event Backtest uses
    the cached test-year models of `evaluation`. Everything else is copied file for file.
    """
    from exportspredictor import artefacts as store
    from exportspredictor import baci, evaluation, freeze
    from exportspredictor.features import features_dir, load_sources, load_year_flows
    from exportspredictor.paths import artefacts_dir, data_dir

    root = Path(root or data_dir())
    artefacts_root = Path(artefacts_root or artefacts_dir())
    source = store.latest_folder(artefacts_root)
    bundle = store.load(source)
    if bundle.evaluation is None:
        raise ValueError(f"Artefact {bundle.version} has no evaluation; run `python -m exportspredictor.evaluation` first")
    metadata = bundle.metadata
    new = {**metadata, "version": version or store.new_version(), "copied_from": bundle.version}
    if test_year_ledger is not None:
        ledger.record_read(test_year_ledger, new, read="Lift and Event Backtests")
    folder = features_dir(root)
    years = json.loads((folder / "features.json").read_text(encoding="utf-8"))["years"]
    result = run(
        {y: folder / f"training_{y}.parquet" for y in years},
        plan,
        uk=baci.UK,
        params=metadata["params"],
        rounds=metadata["rounds"],
        device=device,
        non_uk_rate=metadata["non_uk_sampling_rate"],
        seed=metadata["model_params"].get("seed", 0),
        high_income=load_sources(root)[0].high_income,
        test_models=evaluation.real_test_models(bundle, root, device=device) if plan.secondary else None,
        cache=cache_dir(root),
        lift_held_out=lift_exporters(
            freeze.lift_held_out_exporters(v1_frozen or freeze.v1_folder(artefacts_root)), scope=metadata["exporter_scope"], uk=baci.UK
        ),
        form=bundle.model_form,  # each held-out Exporter's own Market Reach; also scored without it
        year_flows=lambda year: load_year_flows(year, load_sources(root)[1], root),
    )
    report = {**bundle.evaluation, **result}
    folder = store.copy_version(source, artefacts_root, metadata=new, evaluation=report)
    if test_year_ledger is not None:  # the figures are saved: the read is completed (ADR 0009)
        ledger.complete_read(test_year_ledger, ledger.fingerprint(new))
    return folder


if __name__ == "__main__":
    import sys

    command = ledger.Ledger.for_command("python -m exportspredictor.backtests", sys.argv[1:])
    print(add_to_latest(test_year_ledger=command), flush=True)
