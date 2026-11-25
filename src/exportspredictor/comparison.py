"""The pre-registered v1/v2 comparison: fixed before 2023 is read for it.

v1's predictions come only from its frozen tables (`freeze`), checked against their manifest.
v1, v2 and the naive baseline are scored on exactly the same **common rows**:

- **UK**: every UK Product–Market pair of the test year (2023) outside the Excluded Products,
  each scored out-of-fold by both models (ADR 0001).
- **All Exporters**: v1's frozen test-year keys (every positive Flow plus v1's own seeded,
  weighted sample of Zero Cells, with v1's weights), restricted to v2's Exporter scope and to
  non-Excluded Products (ADR 0004, ADR 0005).

**The primary metric** is the UK out-of-fold within-Product Spearman rank correlation, with a
seeded paired bootstrap over Products of v2 − v1. **Secondary metrics** are the Tweedie
deviance and RMSE-log on the all-Exporters rows (paired bootstrap over Products), the
size-matched Lift (both models on the same held-out Exporters and Gap Period, with the same
seeded resamples of Exporters; raw Lift is shown but does not count) and the Event Backtest
distance: how far each model's Scenario Effect lies from the observed difference-in-differences,
per direction, in the event's first year (later years are shown with their caveat but do
not count). Landing inside the observed interval is noted as a bonus.

**The rule.** "v2 beats v1" only if all three hold:

1. the paired-bootstrap interval of the primary metric lies above 0;
2. v2's margin over the naive baseline on the primary metric is larger than v1's;
3. no counted secondary metric gets worse beyond its own interval.

Otherwise the result is mixed and is reported metric by metric, in associational language.
v1 is also measured on all Products (Excluded Products included), so the write-up can show
what the exclusions alone change. The block is stored in the evaluation results as
`v1_comparison`; the API does not serve it.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from exportspredictor import backtests, experiments, freeze
from exportspredictor.evaluation import (
    naive_potential,
    tweedie_unit_deviance,
)
from exportspredictor.evaluation import relabelled as relabelled_report
from exportspredictor.exclusions import drop_excluded, is_excluded
from exportspredictor.features import (
    FEATURES,
    KEYS,
    MARKET_TREND_FEATURES,
    RELATEDNESS_FEATURES,
    TARIFF_FEATURES,
    Sources,
    build_features,
)
from exportspredictor.selection_measure import within_product_correlations
from exportspredictor.wording import number_text

SEED = 0
RESAMPLES = 1000
CI_LEVEL = 0.95
ROW_SETS = ("uk", "all_exporters")
PREDICTORS = {"v1": "v1_usd", "v2": "v2_usd", "naive": "naive_usd"}
PRIMARY = {
    "metric": "spearman_within_product",
    "scope": "uk_out_of_fold",
    "label": "UK out-of-fold within-Product rank correlation",
}
CONDITIONS = {
    "primary_interval_above_zero": "the paired-bootstrap interval of the primary metric lies above 0",
    "larger_margin_over_naive": "{name} beats the Naive Baseline by more than v1 did",
    "no_secondary_worse": "no secondary metric gets worse beyond its own interval",
}
# What the words call the new version. The comparison was written when v2 was new, and its keys
# ("v2", "v2_beats_v1") keep that name; the words name the version that was actually compared.
NEW_VERSION = "v2"
BLEND_VERSION = "the Blend"
# Where the served form has Market Reach (ADR 0010): the comparison still scores the Blend without it, as pre-registered.
WITHOUT_MARKET_REACH_NOTE = (
    "This comparison scores the Blend without Market Reach, the form it was pre-registered on; the served form "
    "scales the Blend by each Market's Market Reach. "
)
_PAIR = ["exporter", "importer", "hs6"]

# --- The gain ladder (reporting only; it decides nothing) -------------------------------------------
# v1's model inputs: v2's features without the ones v2 added (ADR 0006).
V1_FEATURES = [f for f in FEATURES if f not in {*RELATEDNESS_FEATURES, *MARKET_TREND_FEATURES, *TARIFF_FEATURES}]
V1_ROUNDS = 1491  # v1's round count (artefact 20261117T022550Z)


@dataclass(frozen=True)
class LadderPlan:
    """v1's configuration, which rungs A and B train with on v2's rows (scope, sampling, exclusions)."""

    params: dict
    rounds: int


REAL_LADDER = LadderPlan(params=experiments.V1_CONFIG, rounds=V1_ROUNDS)
# The fixture's v1 stand-in: small trees, few rounds.
FIXTURE_LADDER = LadderPlan(params={"max_depth": 2, "eta": 0.3}, rounds=20)
# The cumulative rungs, in order: v1 (frozen), the two extra test-year models, v2 itself.
RUNGS = {
    "v1": "v1 as frozen",
    "A": "v1's features and configuration, with v2's scope, sampling and exclusions",
    "B": "A plus v2's features",
    "C": "v2, tuned",
}
STEPS = {
    "A": "v2's Exporter scope, sampling rates and Excluded Products",
    "B": "v2's new features",
    "C": "v2's tuned configuration",
}
LADDER_NOTE = (
    "The ladder is for reporting only and decides nothing. Its steps are cumulative, so the order of the steps "
    "affects how the change is split between them; another order could split it differently."
)


@dataclass(frozen=True)
class Frozen:
    """v1's frozen predictions: the manifest and one table per `freeze.TABLES` entry."""

    manifest: dict
    tables: Mapping[str, pl.DataFrame]

    @property
    def lift_held_out_exporters(self) -> list[str]:
        return sorted(self.tables["lift_held_out_exporters"]["exporter"].cast(pl.String).to_list())


@dataclass(frozen=True)
class Inputs:
    """What the evaluation needs to run the comparison: the frozen v1 and v2's feature rows on the common keys."""

    frozen: Frozen
    rows: Mapping[str, pl.DataFrame]  # v2's feature rows per common row set (`feature_rows`)
    flows: pl.DataFrame  # every Exporter's Flows of the test year and the Lift outcome year
    ladder: "LadderPlan | None" = None  # v1's configuration for the gain ladder's rungs A and B; None: no ladder


def load_frozen(folder: Path) -> Frozen:
    """The frozen v1 tables in `folder`, refused (ValueError) unless every file matches its manifest checksum."""
    folder = Path(folder)
    manifest = freeze.verify(folder)
    missing = set(freeze.TABLES) - set(manifest["tables"])
    if missing:
        raise ValueError(f"The frozen v1 manifest lists no {sorted(missing)}")
    return Frozen(
        manifest=json.loads(json.dumps(manifest)),
        tables={name: pl.read_parquet(folder / manifest["tables"][name]["file"]) for name in freeze.TABLES},
    )


# --- The common rows ------------------------------------------------------------------------------


def common_keys(frozen: Frozen, *, scope: Sequence[str], uk: str) -> dict[str, pl.DataFrame]:
    """Both common row sets, keyed like feature rows, with v1's Export Potential as `v1_usd`.

    Columns: exporter, importer, hs6, year, value_usd, weight, v1_usd.
    """
    grid = frozen.tables["uk_grid"]
    uk_rows = grid.select(
        exporter=pl.lit(uk),
        importer=pl.col("iso3").cast(pl.String),
        hs6=pl.col("hs6").cast(pl.String),
        year=pl.col("year").cast(pl.Int16),
        value_usd=pl.col("actual_exports_usd").cast(pl.Float64),
        weight=pl.lit(1.0),
        v1_usd=pl.col("export_potential_usd").cast(pl.Float64),
    )
    test_rows = frozen.tables["test_rows"].select(
        pl.col(_PAIR).cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("value_usd", "weight").cast(pl.Float64),
        v1_usd=pl.col("export_potential_usd").cast(pl.Float64),
    )
    in_scope = pl.col("exporter").is_in([str(code) for code in scope])
    return {
        "uk": drop_excluded(uk_rows).sort(_PAIR),
        "all_exporters": drop_excluded(test_rows.filter(in_scope)).sort(_PAIR),
    }


def feature_rows(
    frozen: Frozen, *, scope: Sequence[str], uk: str, flows: pl.DataFrame, sources: Sources
) -> dict[str, pl.DataFrame]:
    """v2's feature rows for both common row sets, for v2's models to score.

    `flows` holds every Flow the feature build needs (the test year and the years Market
    import growth reads), as `features.build_features` takes them.
    """
    out = {}
    for name, keys in common_keys(frozen, scope=scope, uk=uk).items():
        rows = keys.select(
            *(pl.col(c).cast(flows.schema[c]) for c in KEYS), pl.col("value_usd").cast(flows.schema["value_usd"]), "weight"
        )
        out[name] = build_features(rows, flows, sources)
    return out


def scored_rows(
    frozen: Frozen, *, v2: Mapping[str, pl.DataFrame], flows: pl.DataFrame, scope: Sequence[str], uk: str
) -> dict[str, pl.DataFrame]:
    """The common rows with `v1_usd`, `v2_usd` and `naive_usd` on every row.

    `v2` maps each row set to v2's scores (exporter, importer, hs6, export_potential_usd);
    they must cover exactly the common keys, once each, or ValueError. `flows` holds the
    test year's Flows of every Exporter, from which the naive baseline is computed.
    """
    test_year = frozen.manifest["test_year"]
    year_flows = flows.filter(pl.col("year") == test_year) if "year" in flows.columns else flows
    year_flows = year_flows.filter(pl.col("value_usd") > 0).select(*_PAIR, "value_usd")
    out = {}
    for name, keys in common_keys(frozen, scope=scope, uk=uk).items():
        scores = v2[name].select(pl.col(_PAIR).cast(pl.String), v2_usd=pl.col("export_potential_usd").cast(pl.Float64))
        if scores.height != keys.height or scores.select(_PAIR).is_duplicated().any():
            raise ValueError(f"v2's {name} scores must cover the common rows exactly once each")
        joined = keys.join(scores, on=_PAIR, how="left")
        if joined["v2_usd"].null_count():
            raise ValueError(f"v2's {name} scores do not cover the common rows")
        out[name] = naive_potential(joined, year_flows)
    return out


# --- Metrics and the paired bootstrap ---------------------------------------------------------------


def _by_product(frame: pl.DataFrame, predictors: Sequence[str]) -> pl.DataFrame:
    """Per-Product sums for every metric and predictor: weight, deviance, squared log errors, rank correlation."""
    positive = pl.col("value_usd") > 0
    weight = pl.col("weight").cast(pl.Float64)
    sums = frame.group_by("hs6").agg(
        _rows=pl.len(),
        _w=weight.sum(),
        _n=positive.sum().cast(pl.Float64),
        **{f"{p}_dev": (tweedie_unit_deviance(p) * weight).sum() for p in predictors},
        **{
            f"{p}_sq": (pl.col(p).cast(pl.Float64).log1p() - pl.col("value_usd").cast(pl.Float64).log1p())
            .filter(positive)
            .pow(2)
            .sum()
            for p in predictors
        },
    )
    # Pre-registered on the measure of its day (average ranks for ties), so it keeps the legacy measure: a
    # continuity table of the 2023 reads already made, not a choice.
    for p in predictors:
        rho = within_product_correlations(frame, p, tie_proof=False).select("hs6", pl.col("rho").alias(f"{p}_rho"))
        sums = sums.join(rho, on="hs6", how="left")
    return sums.sort("hs6")


def _metrics(products: pl.DataFrame, predictor: str, rows: int) -> dict:
    n = products["_n"].sum()
    rho = products[f"{predictor}_rho"].drop_nulls()
    return {
        "rows": rows,
        "products": products.height,
        "ranked_products": rho.len(),
        "tweedie_deviance": backtests.finite(products[f"{predictor}_dev"].sum() / products["_w"].sum()),
        "rmse_log": backtests.finite(np.sqrt(products[f"{predictor}_sq"].sum() / n)) if n else None,
        "spearman_within_product": backtests.finite(rho.mean()) if rho.len() else None,
    }


def _resamples(n: int, *, resamples: int, seed: int) -> np.ndarray:
    """Seeded bootstrap counts over `n` units (resamples × n)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, [1 / n] * n, size=resamples).astype(float)


def _primary(rows: pl.DataFrame, *, resamples: int, seed: int, level: float) -> dict:
    """Within-Product rank correlation of v1, v2 and naive on the Products both models rank, and the paired bootstrap of v2 − v1."""
    products = _by_product(rows, list(PREDICTORS.values())).drop_nulls(["v1_usd_rho", "v2_usd_rho"])
    if not products.height:
        raise ValueError("No Product of the UK common rows is ranked by both models")
    v1, v2 = products["v1_usd_rho"].to_numpy(), products["v2_usd_rho"].to_numpy()
    naive = products["naive_usd_rho"].drop_nulls()
    counts = _resamples(products.height, resamples=resamples, seed=seed)
    boot = counts @ (v2 - v1) / products.height
    return {
        **PRIMARY,
        "products": products.height,
        "v1": float(v1.mean()),
        "v2": float(v2.mean()),
        "naive": backtests.finite(naive.mean()) if naive.len() else None,
        "difference": float((v2 - v1).mean()),
        "ci": backtests.interval(boot, level),
        "bootstrap_unit": "Product",
    }


def _accuracy_secondaries(rows: pl.DataFrame, *, resamples: int, seed: int, level: float) -> list[dict]:
    """Tweedie deviance and RMSE-log on the all-Exporters rows: v1, v2, and the paired bootstrap of v2 − v1."""
    products = _by_product(rows, ["v1_usd", "v2_usd"])
    counts = _resamples(products.height, resamples=resamples, seed=seed)

    def column(name: str) -> np.ndarray:
        return products[name].to_numpy().astype(float)

    w, n = column("_w"), column("_n")
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance = {m: (column(f"{m}_usd_dev"), w) for m in ("v1", "v2")}
        rmse = {m: (column(f"{m}_usd_sq"), n) for m in ("v1", "v2")}
        out = []
        for metric, label, parts, root in (
            ("tweedie_deviance", "Tweedie deviance, all Exporters", deviance, False),
            ("rmse_log", "RMSE of log(1 + value), all Exporters' non-zero Flows", rmse, True),
        ):

            def value(model: str, c: np.ndarray, parts=parts, root=root) -> np.ndarray:
                top, bottom = parts[model]
                ratio = (c @ top) / (c @ bottom)
                return np.sqrt(ratio) if root else ratio

            ones = np.ones((1, products.height))
            v1, v2 = value("v1", ones)[0], value("v2", ones)[0]
            ci = backtests.interval(value("v2", counts) - value("v1", counts), level)
            out.append(
                {
                    "metric": metric,
                    "label": label,
                    "scope": "all_exporters",
                    "better": "lower",
                    "v1": backtests.finite(v1),
                    "v2": backtests.finite(v2),
                    "difference": backtests.finite(v2 - v1),
                    "ci": ci,
                    "worse": ci[0] is not None and ci[0] > 0,
                    "counts": True,
                    "rule": "worse when the paired-bootstrap interval of v2 − v1 lies above 0",
                }
            )
    return out


def _lift_secondaries(
    frozen: Frozen,
    *,
    lift_v2: dict,
    flows: pl.DataFrame,
    settings: backtests.LiftSettings,
    scope: Sequence[str],
    uk: str,
    seed: int,
) -> tuple[list[dict], dict]:
    """Size-matched (counted) and raw Lift of v1 on its frozen rows against v2's, on the same held-out Exporters."""
    held = backtests.lift_exporters(frozen.lift_held_out_exporters, scope=scope, uk=uk)
    if held != sorted(lift_v2["held_out_exporters"]):
        raise ValueError(f"v2's Lift held out {lift_v2['held_out_exporters']}, not v1's Exporters in scope {held}")
    is_held = pl.col("exporter").cast(pl.String).is_in(held)
    rows = drop_excluded(frozen.tables["lift"].filter(is_held))
    later = flows.filter(
        (pl.col("year") == settings.outcome_year) & is_held & (pl.col("value_usd") > 0)
    ).select(*_PAIR, "value_usd")
    v1 = backtests.lift(backtests.gap_period_pairs(rows, later, settings), resamples=lift_v2["resamples"], seed=seed)
    out = []
    for metric, label, one, two, counts in (
        ("size_matched_lift", "Size-matched Lift", v1["size_matched"], lift_v2["size_matched"], True),
        ("lift", "Raw Lift (shown, not counted)", v1, lift_v2, False),
    ):
        low = one["ci"][0]
        measured = one["lift"] is not None and two["lift"] is not None
        out.append(
            {
                "metric": metric,
                "label": label,
                "scope": "lift",
                "better": "higher",
                "v1": one["lift"],
                "v2": two["lift"],
                "difference": two["lift"] - one["lift"] if measured else None,
                "v1_ci": one["ci"],
                "v2_ci": two["ci"],
                "worse": counts and measured and low is not None and two["lift"] < low,
                "counts": counts,
                "rule": "worse when v2's Lift lies below v1's interval (same Exporters, same resamples)",
            }
        )
    return out, {"held_out_exporters": held, "v1_pairs": v1["pairs"], "resamples": lift_v2["resamples"]}


def _event_secondaries(frozen: Frozen, *, event_v2: dict) -> list[dict]:
    """Each direction's distance from the observed difference-in-differences, per post year; the event year counts."""
    if frozen.manifest.get("event") != event_v2["name"]:
        raise ValueError(f"The frozen Event Backtest is {frozen.manifest.get('event')}, not {event_v2['name']}")
    table = drop_excluded(frozen.tables["event_backtest"])
    weight = pl.col("weight").cast(pl.Float64)
    v1_effects = {
        (row["direction"], row["year"]): row["effect"]
        for row in table.group_by("direction", "year")
        .agg(
            effect=(pl.col("export_potential_on_usd") * weight).sum() / (pl.col("export_potential_off_usd") * weight).sum()
            - 1
        )
        .iter_rows(named=True)
    }
    out = []
    for direction in event_v2["directions"]:
        for row in direction["by_year"]:
            key = (direction["label"], row["year"])
            if key not in v1_effects:
                raise ValueError(f"The frozen Event Backtest has no rows for {key}")
            observed, (low, high) = row["observed"], row["observed_ci"]
            v1, v2 = backtests.finite(v1_effects[key]), row["predicted"]
            counts = row["year"] == event_v2["event_year"]
            measured = None not in (observed, v1, v2)
            d1 = abs(v1 - observed) if measured else None
            d2 = abs(v2 - observed) if measured else None
            tolerance = (high - low) / 2 if low is not None else 0.0
            out.append(
                {
                    "metric": "event_distance",
                    "label": f"Event Backtest distance, {direction['label']}, {row['year']}",
                    "scope": event_v2["name"],
                    "direction": direction["label"],
                    "year": row["year"],
                    "better": "lower",
                    "observed": observed,
                    "observed_ci": [low, high],
                    "v1_predicted": v1,
                    "v2_predicted": v2,
                    "v1": d1,
                    "v2": d2,
                    "difference": d2 - d1 if measured else None,
                    "v1_inside": measured and low is not None and low <= v1 <= high,
                    "v2_inside": measured and low is not None and low <= v2 <= high,
                    "worse": counts and measured and d2 - d1 > tolerance,
                    "counts": counts,
                    "rule": "worse when v2's distance exceeds v1's by more than half the observed interval",
                    **({} if counts else {"caveat": event_v2["caveat"]}),
                }
            )
    return out


def _all_products(frozen: Frozen, scored: Mapping[str, pl.DataFrame], *, scope: Sequence[str], uk: str) -> dict:
    """v1 on the common rows against v1 on every Product it scored (Excluded Products included)."""
    keys = common_keys(frozen, scope=scope, uk=uk)
    grid = frozen.tables["uk_grid"]
    everything = {
        "uk": grid.select(
            hs6=pl.col("hs6").cast(pl.String),
            value_usd=pl.col("actual_exports_usd").cast(pl.Float64),
            weight=pl.lit(1.0),
            v1_usd=pl.col("export_potential_usd").cast(pl.Float64),
        ),
        "all_exporters": frozen.tables["test_rows"]
        .filter(pl.col("exporter").cast(pl.String).is_in([str(c) for c in scope]))
        .select(
            pl.col("hs6").cast(pl.String),
            pl.col("value_usd", "weight").cast(pl.Float64),
            v1_usd=pl.col("export_potential_usd").cast(pl.Float64),
        ),
    }
    out = {}
    for name in ROW_SETS:
        common = _metrics(_by_product(scored[name], ["v1_usd"]), "v1_usd", keys[name].height)
        full = _metrics(_by_product(everything[name], ["v1_usd"]), "v1_usd", everything[name].height)
        out[name] = {
            "common_rows": common,
            "all_products": full,
            "excluded_rows": everything[name].select(is_excluded().sum()).item(),
        }
    return out


def _ladder(
    uk: pl.DataFrame, rungs: Mapping[str, pl.DataFrame], *, resamples: int, seed: int, level: float
) -> dict:
    """The primary metric for each rung of the gain ladder, and each step's change with a paired-bootstrap interval.

    `uk` is the scored UK common rows (`scored_rows`); `rungs` maps "A" and "B" to their scores
    (exporter, importer, hs6, export_potential_usd) on exactly those keys. The Products are the
    primary metric's (ranked by v1 and v2), so v1's and C's values are the primary metric's.
    """
    columns = {"v1": "v1_usd", "A": "A_usd", "B": "B_usd", "C": "v2_usd"}
    rows = uk
    for name in ("A", "B"):
        scores = rungs[name].select(pl.col(_PAIR).cast(pl.String), pl.col("export_potential_usd").alias(columns[name]))
        if scores.height != uk.height or scores.select(_PAIR).is_duplicated().any():
            raise ValueError(f"Rung {name}'s scores must cover the UK common rows exactly once each")
        rows = rows.join(scores, on=_PAIR, how="left")
        if rows[columns[name]].null_count():
            raise ValueError(f"Rung {name}'s scores do not cover the UK common rows")
    products = _by_product(rows, list(columns.values())).drop_nulls(["v1_usd_rho", "v2_usd_rho"])
    rho = {name: products[f"{column}_rho"].to_numpy().astype(float) for name, column in columns.items()}
    counts = _resamples(products.height, resamples=resamples, seed=seed)
    names = list(RUNGS)
    steps = []
    for before, after in pairwise(names):
        change = rho[after] - rho[before]
        known = np.isfinite(change)
        boot = (counts[:, known] @ change[known]) / counts[:, known].sum(axis=1)
        steps.append(
            {
                "from": before,
                "to": after,
                "label": STEPS[after],
                "change": backtests.finite(np.nanmean(rho[after]) - np.nanmean(rho[before])),
                "ci": backtests.interval(boot, level),
            }
        )
    return {
        "metric": PRIMARY["label"],
        "products": products.height,
        "rungs": [{"name": name, "label": RUNGS[name], "value": backtests.finite(np.nanmean(rho[name]))} for name in names],
        "steps": steps,
        "note": LADDER_NOTE,
    }


# --- The verdict -----------------------------------------------------------------------------------


def verdict(primary: dict, secondary: Sequence[dict], *, name: str = NEW_VERSION) -> dict:
    """Whether the new version ("v2" in the keys) beats v1 under the pre-registered rule, and the result in
    words, metric by metric, calling the new version `name`.

    `primary` has v1, v2, naive and the bootstrap `ci` of v2 − v1; each `secondary` entry
    has `label`, v1, v2, `worse` and `counts`.
    """
    naive = primary["naive"]
    margins = {
        model: None if naive is None or primary[model] is None else primary[model] - naive for model in ("v1", "v2")
    }
    low = primary["ci"][0]
    conditions = {
        "primary_interval_above_zero": low is not None and low > 0,
        "larger_margin_over_naive": None not in margins.values() and margins["v2"] > margins["v1"],
        "no_secondary_worse": not any(entry["worse"] for entry in secondary if entry["counts"]),
    }
    beats = all(conditions.values())
    lines = [_primary_text(primary, margins, name)]
    lines += [_secondary_text(entry, name) for entry in secondary]
    if beats:
        head = f"{_capital(name)} beats v1 under the pre-registered rule: all three conditions hold."
    else:
        failed = "; ".join(CONDITIONS[key].format(name=name) for key, held in conditions.items() if not held)
        head = f"The result is mixed, so it is reported metric by metric. Not met: {failed}."
    return {
        "v2_beats_v1": beats,
        "conditions": conditions,
        "margins_over_naive": margins,
        "text": " ".join([head, *lines]),
    }


def _number(value: float | None, signed: bool = False) -> str:
    return number_text(value, "+.3f" if signed else ".3f")


def _capital(name: str) -> str:
    """A name opening a sentence: "the Blend" becomes "The Blend"; a version tag such as "v2" stays as it is."""
    return "The" + name[3:] if name.startswith("the ") else name


def _primary_text(primary: dict, margins: dict, name: str) -> str:
    low, high = primary["ci"]
    interval = "" if low is None else f", 95% interval {_number(low, True)} to {_number(high, True)}"
    difference = primary["difference"]
    agreement = "closer" if difference >= 0 else "less close"
    return (
        f"On the {primary['label']}, {name}'s Export Potential ranks Markets in {agreement} agreement with Actual "
        f"Exports than v1's ({_number(primary['v2'])} against {_number(primary['v1'])}, a difference of "
        f"{_number(difference, True)}{interval}); against the Naive Baseline ({_number(primary['naive'])}) the "
        f"margins are {_number(margins['v2'], True)} for {name} and {_number(margins['v1'], True)} for v1."
    )


def _secondary_text(entry: dict, name: str) -> str:
    if entry["v1"] is None or entry["v2"] is None:
        return f"{entry['label']}: could not be measured for both models."
    judged = (
        "worse beyond its interval" if entry["worse"] else ("not counted" if not entry["counts"] else "not worse beyond its interval")
    )
    bonus = f" {_capital(name)}'s Scenario Effect lies inside the observed interval." if entry.get("v2_inside") else ""
    return f"{entry['label']}: v1 {_number(entry['v1'])}, {name} {_number(entry['v2'])} ({judged}).{bonus}"


def _not_run_text(not_run: Sequence[str]) -> str:
    if not not_run:
        return ""
    return (
        f" The {' and '.join(not_run)} secondaries were not run for this version, so the third condition "
        "is judged on the other secondaries only."
    )


def version_name(report: dict) -> str:
    """What the comparison's words call the version a report evaluates: the Blend when it serves one, else v2."""
    return BLEND_VERSION if (report.get("model_form") or {}).get("naive_weight") else NEW_VERSION


def saved_verdict_text(block: dict, *, name: str) -> str:
    """A saved `v1_comparison` block's verdict, worded again from its saved figures, the new version called `name`;
    a block that scored the Blend without the served form's Market Reach says so first (`WITHOUT_MARKET_REACH_NOTE`)."""
    judged = verdict(block["primary"], block["secondary"], name=name)
    note = WITHOUT_MARKET_REACH_NOTE if block.get("scored_without_market_reach") else ""
    return note + judged["text"] + _not_run_text(block.get("secondaries_not_run") or [])


def relabelled(report: dict) -> dict:
    """A saved evaluation report in today's words: `evaluation.relabelled`, and its v1 comparison's verdict
    worded for the version it evaluated (`version_name`). Nothing is recomputed; the saved report is not changed."""
    out = relabelled_report(report)
    block = out.get("v1_comparison")
    if block:
        text = saved_verdict_text(block, name=version_name(report))
        out = {**out, "v1_comparison": {**block, "verdict": {**block["verdict"], "text": text}}}
    return out


# --- The comparison --------------------------------------------------------------------------------


def compare(
    frozen: Frozen,
    *,
    v2: Mapping[str, pl.DataFrame],
    flows: pl.DataFrame,
    lift_v2: dict | None,
    event_v2: dict | None,
    lift_settings: backtests.LiftSettings,
    scope: Sequence[str],
    uk: str,
    test_year: int,
    seed: int = SEED,
    resamples: int = RESAMPLES,
    level: float = CI_LEVEL,
    ladder: Mapping[str, pl.DataFrame] | None = None,
    ladder_plan: LadderPlan | None = None,
    name: str = NEW_VERSION,
) -> dict:
    """The `v1_comparison` block of v2's evaluation results; its words call the new version `name`.

    `v2` maps each common row set to v2's scores on exactly its keys (`scored_rows`).
    `flows` holds the Flows (exporter, importer, hs6, year, value_usd) of the test year and of
    the Lift outcome year, of every Exporter. `lift_v2` is v2's Lift result
    (`backtests.lift_backtest`) and `event_v2` its primary Event Backtest
    (`backtests.event_backtest`). The bootstraps are seeded by `seed`. `ladder` maps the gain
    ladder's rungs "A" and "B" to their scores on the UK common rows (`ladder_plan` is the v1
    configuration they were trained with, recorded); the ladder is reported, never judged.
    When the Event Backtests were not run (`lift_v2` and `event_v2` None), their secondaries are
    left out and listed under `secondaries_not_run`, and the verdict says so.
    """
    if frozen.manifest.get("test_year") != test_year:
        raise ValueError(f"The frozen v1 tables are for {frozen.manifest.get('test_year')}, not the test year {test_year}")
    scored = scored_rows(frozen, v2=v2, flows=flows, scope=scope, uk=uk)
    primary = _primary(scored["uk"], resamples=resamples, seed=seed, level=level)
    lifts, lift_info, not_run = [], None, []
    if lift_v2 is None:
        not_run.append("Lift")
    else:
        lifts, lift_info = _lift_secondaries(
            frozen, lift_v2=lift_v2, flows=flows, settings=lift_settings, scope=scope, uk=uk, seed=seed
        )
    if event_v2 is None:
        not_run.append("Event Backtest")
    secondary = [
        *_accuracy_secondaries(scored["all_exporters"], resamples=resamples, seed=seed, level=level),
        *lifts,
        *([] if event_v2 is None else _event_secondaries(frozen, event_v2=event_v2)),
    ]
    judged = verdict(primary, secondary, name=name)
    judged["text"] += _not_run_text(not_run)
    block = {
        "test_year": test_year,
        "frozen": {key: frozen.manifest.get(key) for key in ("tag", "commit", "source_models", "frozen_at")},
        "rows": {name: rows.height for name, rows in scored.items()},
        "seed": seed,
        "resamples": resamples,
        "ci_level": level,
        "primary": primary,
        "secondary": secondary,
        "lift": lift_info,
        "v1_all_products": _all_products(frozen, scored, scope=scope, uk=uk),
        "verdict": judged,
    }
    if not_run:
        block["secondaries_not_run"] = not_run
    if ladder is not None:
        block["ladder"] = _ladder(scored["uk"], ladder, resamples=resamples, seed=seed, level=level)
        if ladder_plan is not None:
            block["ladder"]["v1_configuration"] = {
                "params": ladder_plan.params, "rounds": ladder_plan.rounds, "features": {"A": V1_FEATURES, "B": FEATURES}
            }
    return block
