"""Accuracy of Export Potential on the test year, the Baseline's stability, and the known limitations.

The time split is train ≤ 2021, validate 2022, test 2023 (Baseline year 2024). Tuning
reads only the training and validation years (`pipeline`). The **test-year models** here
are five fold models refitted on every year before the test year with the tuned
hyperparameters, so the test year is never seen by tuning or training.

Each metric is computed for the model and for a **naive baseline**, the Exporter's share
of the Product's world exports times the Importer's imports of that Product (both with
the row's own Flow left out, as the model's supply and demand features are):

- **Tweedie deviance** (variance power 1.5, weighted, so sampled zero cells count by
  their weight): the training loss, on every row. Lower is better.
- **RMSE of log(1 + v)** on non-zero Flows: the typical error in orders of magnitude.
  Lower is better.
- **Spearman rank correlation within each Product**, averaged over Products: how well
  the ranking of Flows within a Product matches reality. Higher is better. This is how
  Advisors use the model (Markets ranked for one Product), so it is the headline. It is
  the Selection Measure (`selection_measure`): ties in a prediction are broken by a stable
  hash, so tied predictions cannot raise it.

Two scopes are evaluated:

- `all_exporters`: every test-year training row (every non-zero Flow plus the sampled
  zero cells). UK rows are scored out-of-fold; other rows by the mean of the fold models.
- `uk_out_of_fold`: the UK's full Product × Market grid of the test year, each pair scored
  by the fold model that held it out (ADR 0001).

Each is broken down by Sector (HS2 chapter) and by the Importer's region, weakest first.
Excluded Products are never measured (ADR 0005), and each scope is also split by
`tariff_imputed` into rows with an observed Applied Tariff and rows carrying 2021 forward
(ADR 0006). TRAINS ends in 2021, so on the 2023 test year every row is imputed.

The **stability** check compares the Baseline's rankings with the same models' rankings of
the test year's UK grid: how much the rankings move from one year of data to the next.

The result is a JSON-ready dict stored in the artefact as `evaluation.json`; Lift and the
Event Backtest (#21) add their own keys to it. `uv run python -m exportspredictor.evaluation`
adds an evaluation to the newest real artefact without retraining its Baseline models.
"""

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import xgboost as xgb

from exportspredictor import artefacts as store
from exportspredictor import baci, ledger
from exportspredictor.conformal import COVERAGE_TOLERANCE, coverage_text
from exportspredictor.currency import format_usd
from exportspredictor.exclusions import drop_excluded
from exportspredictor.features import (
    FEATURES,
    FeatureTransform,
    features_dir,
    load_sources,
    load_year_flows,
)
from exportspredictor.folds import N_FOLDS, score_out_of_fold, train_fold_models
from exportspredictor.gap_period import baseline_year_scores
from exportspredictor.gravity import GDP_GAPS
from exportspredictor.paths import artefacts_dir, data_dir
from exportspredictor.prediction import (
    MIN_PREDICTION_USD,
    MODEL_ALONE,
    ModelForm,
    export_potential,
)
from exportspredictor.selection_measure import (
    MIN_RANKED_ROWS,
    selection_measure,
    with_tie_key,
    within_product_correlations,
)
from exportspredictor.tariffs import LAST_TRAINS_YEAR
from exportspredictor.training import Chunk, scan_chunks
from exportspredictor.wording import number_text, rate_text

TWEEDIE_POWER = 1.5
# A row of every Exporter is one Product, Exporter and Importer: the tie key hashes all three.
_ROW_MARKET = ("exporter", "importer")
STABILITY_TOP_N = 1000
TEST_MODELS_FILE = "test_models.json"
METRICS = ["tweedie_deviance", "rmse_log", "spearman_within_product"]
METRIC_INFO = {
    "tweedie_deviance": {"label": "Tweedie deviance", "better": "lower"},
    "rmse_log": {"label": "RMSE of log(1 + value), non-zero Flows", "better": "lower"},
    "spearman_within_product": {"label": "Spearman rank correlation within each Product", "better": "higher"},
}
HEADLINE_METRIC = "spearman_within_product"
HEADLINE_SCOPE = "uk_out_of_fold"
# A headline win smaller than this is reported as "only slightly better".
NARROW_MARGIN = 0.05
_PREDICTORS = {"model": "export_potential_usd", "naive": "naive_usd"}
# The two sides of the test-metric split by `tariff_imputed` (ADR 0006), in display order.
TARIFF_GROUPS = ("observed", "imputed")
# Evaluation results kept in the artefact (and the model card) but never served by the API:
# the pre-registered v1/v2 comparison (`comparison`, spec #39).
NOT_SERVED = frozenset({"v1_comparison"})

NOT_RUN = "not run for this version"


def served_report(report: dict) -> dict:
    """An evaluation report as the API serves it: without the parts in `NOT_SERVED`, with today's names
    (`relabelled`) and with the Potential Range worded now (`conformal.coverage_text`), not when saved, so
    older versions state the share of Uncertain Opportunities the same way. The saved report is not changed."""
    served = {key: value for key, value in relabelled(report).items() if key not in NOT_SERVED}
    if served.get("potential_range"):
        served["potential_range"] = {**served["potential_range"], "text": coverage_text(served["potential_range"])}
    return served

# The opt-in parts of a pipeline run (`pipeline.Parts`, spec #65 story 23): title, the report
# block that holds its result, and the switch that runs it. A part whose block is absent was skipped.
OPTIONAL_PARTS = (
    ("Event Backtests", "event_backtest", "--event-backtests",
     ("Lift and the Event Backtests test Export Gaps against later growth and Scenario Effects against a real Trade "
     "Agreement.")),
    ("Monotone-constraint comparison", "monotone_constraints", "--constraint-comparison",
     "It compares the monotone-constrained model with an unconstrained one."),
    ("Gain ladder", "ladder", "--gain-ladder",
     "It splits the change in the v1 comparison's primary metric from v1 to v2 into steps."),
    ("PPML benchmark", "ppml_benchmark", "--ppml-benchmark",
     ("It compares the model's Trade Agreement association with a 3-way fixed-effects PPML gravity model and with "
     "published estimates.")),
)


def parts_not_run(report: dict, ppml_benchmark: dict | None) -> list[dict]:
    """The opt-in parts an artefact version does not hold, each with its "not run for this version" text.

    `report` is its evaluation report and `ppml_benchmark` its saved benchmark
    (`Artefacts.ppml_benchmark`), which lives beside the report, not in it. The gain ladder
    lives inside the v1 comparison; a version with every part lists none.
    """
    blocks = {**report, **(report.get("v1_comparison") or {})}
    if ppml_benchmark is not None:
        blocks["ppml_benchmark"] = ppml_benchmark
    return [
        {"title": title, "text": f"{title}: {NOT_RUN}. {why} It runs when the pipeline is asked for it "
         f"(`python -m exportspredictor.pipeline {switch}`)."}
        for title, block, switch, why in OPTIONAL_PARTS
        if block not in blocks
    ]

LIMITATIONS = [
    {
        "title": "Goods only",
        "text": "The model covers goods trade only. Services, about half of UK exports, are not in the data.",
    },
    {
        "title": "Tariffs carried forward from 2021",
        "text": "Applied Tariffs come from UNCTAD TRAINS, which ends in 2021. 2022–2024, the test year and the "
        "whole Gap Period included, reuse each pair's 2021 rate, or 0 where a Trade Agreement entered force after "
        "2021 (which overstates liberalisation while its tariff cuts phase in). Tariff changes since 2021 are "
        "invisible, and every test-year row is scored on a carried-forward tariff.",
    },
    {
        "title": "Excluded Products",
        "text": "Crude and refined oil, gas, coal and electricity, bullion and coins, precious-metal scrap, antiques "
        "and radioactive elements are Excluded Products: their trade follows pipelines, bullion markets or licences "
        "the model cannot see. They are never scored or ranked, and every metric here leaves them out.",
    },
    {
        "title": "Trained on the top 100 Exporters",
        "text": "The model trains on Flows from the top 100 economies by mean GDP over 2012–2023, the UK always "
        "included, into every Market; other Exporters' rows are sampled to fit the GPU. Smaller Exporters still "
        "count in every total the features use, but the model never learns from their own Flows.",
    },
    {
        "title": "Values are nominal US dollars",
        "text": "Every value is in nominal US dollars, as BACI reports it, with no adjustment for inflation or "
        "exchange rates. A change between years mixes real trade growth with price and currency movements.",
    },
    {
        "title": "What the Potential Range assumes",
        "text": "The Potential Range is calibrated on other Exporters' 2024 Flows, never the UK's, so it assumes UK "
        "pairs vary around their Export Potential as comparable pairs of other top-100 Exporters do "
        "(exchangeability). That is assumed, not guaranteed: the coverage check above shows how well it held on "
        "held-out rows. It describes how much comparable pairs vary, not a confidence interval for Export "
        "Potential, and Scenario Effects get no range.",
    },
    {
        "title": "Associations, not causes",
        "text": "Export Potential and Scenario Effects are associations the model has learned from global trade. "
        "\"The model associates a Trade Agreement with +X%\" does not mean the agreement causes +X%, and an "
        "Opportunity is a tendency, not a forecast.",
    },
    {
        "title": "Provisional 2024 data",
        "text": "The Baseline year, 2024, is provisional in BACI: some countries had not reported when it was "
        "built, and it will be revised in the next release. 2024 has about 5% fewer Flows than 2023.",
    },
    {
        "title": "The 2012 product edition",
        "text": "Products use HS 2012 codes (BACI's HS12 revision, chosen for its longer history). A few HS6 codes "
        "Advisors know from current tariff schedules (HS 2017 or 2022) differ or are split.",
    },
    {
        "title": "Reporting quality differs between countries",
        "text": "BACI reconciles what exporters and importers declare, and countries report with different care "
        "and delay. A missing Flow is treated as zero, so the model may be weaker for low-income and poorly "
        "reporting Markets; the region breakdown shows where.",
    },
    {
        "title": "Sanctions and other non-tariff restrictions are invisible",
        "text": "Sanctions, embargoes, quotas, licensing and product standards are not model inputs. A Market under "
        "sanctions can still show a large Export Gap: Russia ranks high for UK whisky because the model sees "
        "its fundamentals, not the sanctions in force since 2022. Check such Markets before acting on them.",
    },
    {
        "title": "No GDP for some small Markets",
        "text": "GDP comes from the World Bank, the IMF and CEPII, and for small territories they miss from UN, "
        "Statistics Netherlands and CIA World Factbook figures. No source gives one for "
        + ", ".join(sorted(GDP_GAPS.values()))
        + ", so the model reads their size from their trade alone, and their Export Potential is less reliable.",
    },
]


def known_limitations(artefacts: store.Artefacts) -> list[dict]:
    """Every known limitation of an artefact version that can be read from it: its results, its
    Potential Range bands and the fixed `LIMITATIONS`. The API serves these; the model card adds
    the gravity benchmark, which no artefact holds.
    """
    report = artefacts.evaluation or {}
    return [
        *result_limitations(artefacts.metadata, report),
        *range_limitations(artefacts.potential_range_bands),
        *size_coverage_limitations(report.get("potential_range")),
        *LIMITATIONS,
    ]


def result_limitations(metadata: dict, report: dict) -> list[dict]:
    """The limitations an artefact's own results show: its headline, Lift, Event Backtest and sampling."""
    limits = []
    headline = report.get("headline")
    if headline and headline.get("beats_naive") is False:
        limits.append({"title": "Does not beat a naive ranking for UK pairs",
                       "text": f"Within a Product, the model ranks UK Markets with a rank correlation of "
                       f"{number_text(headline['model'], '.2f')}, against {number_text(headline['naive'], '.2f')} for the UK's market share "
                       f"times each Market's imports: on the headline measure it does not beat that naive rule. Its "
                       f"value is in the size of Export Potential (a much lower deviance), not in the order of Markets."})
    elif headline:
        blend = bool((metadata.get("model_form") or {}).get("naive_weight"))
        trade_off = (
            "The Blend was chosen on this ranking, not on the size of Export Potential: read it as where to look "
            "first, and treat dollar values with more caution than rankings."
            if blend else "Its value is mostly in the size of Export Potential, not its order."
        )
        limits.append({"title": "Only slightly better than a naive ranking for UK pairs",
                       "text": f"Within a Product, the model ranks UK Markets with a rank correlation of "
                       f"{number_text(headline['model'], '.2f')}, against {number_text(headline['naive'], '.2f')} for the UK's market share times "
                       f"each Market's imports. {trade_off}"})
    lift = report.get("lift")
    if lift:
        limits.append({"title": "Large Export Gaps mostly mark large pairs",
                       "text": f"Lift is {number_text(lift['lift'], '.1f')}, but against pairs of similar Export Potential it is "
                       f"{number_text(lift['size_matched']['lift'], '.2f')}: an Opportunity is worth a look, not more likely to grow "
                       f"than its peers of the same size."})
    events = report.get("event_backtest", {}).get("events", [])
    if events:
        pooled = events[0]["directions"][0]["pooled"]
        limits.append({"title": "Scenario Effects are larger than observed changes",
                       "text": f"For {events[0]['title'][0].lower() + events[0]['title'][1:]}, the model associated the agreement with "
                       f"{number_text(pooled['predicted'], '+.0%')}, against an observed {number_text(pooled['observed'], '+.0%')}. Read Scenario "
                       f"Effects as an upper-end association."})
    rate = metadata.get("non_uk_sampling_rate")
    if rate is not None:
        limits.append({"title": "Trained on a sample of other Exporters' Flows",
                       "text": f"The models train on every UK row but a reweighted {rate_text(rate)} sample of other "
                       f"Exporters' rows (and {rate_text(metadata.get('zero_sampling_rate'))} of their Zero Cells), to fit "
                       f"the GPU's memory. Unbiased, but noisier than training on all rows."})
    return limits


def range_limitations(bands: pl.DataFrame) -> list[dict]:
    """For a version whose offsets are dollars per band, the top band gives every large pair one width.

    Versions since the log-scale amendment of ADR 0007 store log offsets (`low_offset_log`), whose
    range scales with Export Potential, and have no such limitation.
    """
    if bands.is_empty() or "low_offset_usd" not in bands.columns:
        return []
    top = bands.sort("min_export_potential_usd").row(-1, named=True)
    width = top["high_offset_usd"] - top["low_offset_usd"]
    return [{
        "title": "The Potential Range is a fixed width in dollars",
        "text": f"Its offsets are fixed amounts of dollars within each band of Export Potential (ADR 0007), so every "
        f"pair in the top band, from {format_usd(top['min_export_potential_usd'])} of Export Potential upwards, gets "
        f"the same range, {format_usd(width)} wide, whatever its size. For a large pair that range is a sliver, and "
        f"it is an Uncertain Opportunity only when its Export Gap is under the band's lower offset, "
        f"{format_usd(-top['low_offset_usd'])}: for a large pair that means \"a small gap for its size\", not that "
        f"the model is unsure of it. A relative (log-scale) range is a candidate for a later version.",
    }]


def size_coverage_limitations(potential_range: dict | None) -> list[dict]:
    """A size of Export Potential whose held-out Potential Range coverage (`conformal.coverage_report`'s
    `by_size`) is more than `COVERAGE_TOLERANCE` from the target: the range is too narrow or too wide
    for pairs of that size. Groups with no held-out rows, and versions without the breakdown, show none.
    """
    if not potential_range or not potential_range.get("by_size"):
        return []
    target = potential_range.get("target", 0.8)
    missed = [row for row in potential_range["by_size"]
              if row["coverage"] is not None and round(abs(row["coverage"] - target), 9) > COVERAGE_TOLERANCE]
    if not missed:
        return []

    def listed(rows: list[dict]) -> str:
        return ", ".join(f"{row['size']} ({row['coverage']:.0%} inside it)" for row in rows)

    narrow = [row for row in missed if row["coverage"] < target]
    wide = [row for row in missed if row["coverage"] > target]
    parts = [
        *([f"too narrow, so it understates their variation, for pairs with Export Potential {listed(narrow)}"]
          if narrow else []),
        *([f"too wide, so it overstates their variation, for pairs with Export Potential {listed(wide)}"]
          if wide else []),
    ]
    return [{
        "title": "The Potential Range misses its target for some sizes of pair",
        "text": f"On a held-out half of the calibration rows, the range is {'; and '.join(parts)}. The target is "
        f"{target:.0%} of Actual Exports inside the range, give or take {COVERAGE_TOLERANCE * 100:.0f} points.",
    }]


# --- Metrics ---------------------------------------------------------------------------


def tweedie_unit_deviance(prediction: str, *, power: float = TWEEDIE_POWER) -> pl.Expr:
    """Each row's Tweedie unit deviance of `prediction` against `value_usd` (1 < power < 2)."""
    y = pl.col("value_usd").cast(pl.Float64)
    mu = pl.col(prediction).cast(pl.Float64).clip(lower_bound=MIN_PREDICTION_USD)
    return 2 * (
        y.pow(2 - power) / ((1 - power) * (2 - power))
        - y * mu.pow(1 - power) / (1 - power)
        + mu.pow(2 - power) / (2 - power)
    )


def tweedie_deviance(prediction: str, *, power: float = TWEEDIE_POWER) -> pl.Expr:
    """Weighted mean Tweedie unit deviance of `prediction` against `value_usd` (1 < power < 2)."""
    weight = pl.col("weight").cast(pl.Float64)
    return (tweedie_unit_deviance(prediction, power=power) * weight).sum() / weight.sum()


def _rmse_log(prediction: str) -> pl.Expr:
    error = pl.col(prediction).cast(pl.Float64).log1p() - pl.col("value_usd").cast(pl.Float64).log1p()
    return error.filter(pl.col("value_usd") > 0).pow(2).mean().sqrt()


def naive_from_totals(supply_usd: pl.Expr, demand_usd: pl.Expr, world_usd: pl.Expr) -> pl.Expr:
    """The Naive Baseline's one formula: supply × demand / world exports, 0 where the world exports nothing.

    Each total leaves the row's own Flow out: the Exporter's exports of the Product, the
    Importer's imports of it and the Product's world exports. A negative total (float
    error) counts as 0.
    """
    return (
        pl.when(world_usd > 0)
        .then(supply_usd.clip(lower_bound=0) * demand_usd.clip(lower_bound=0) / world_usd)
        .otherwise(0.0)
    )


def naive_potential(rows: pl.DataFrame, flows: pl.DataFrame) -> pl.DataFrame:
    """Add `naive_usd`: the Exporter's share of the Product's world exports times the Importer's imports of it.

    `flows` holds every non-zero Flow of the same year (exporter, importer, hs6, value_usd).
    Each row's own Flow is left out of all three totals, as the model's features leave it out.
    """
    value = pl.col("value_usd")
    active = flows.filter(value > 0).select(
        pl.col("exporter", "importer", "hs6").cast(pl.String), value.cast(pl.Float64)
    )
    supply = active.group_by("exporter", "hs6").agg(_supply_usd=value.sum())
    demand = active.group_by("importer", "hs6").agg(_demand_usd=value.sum())
    world = active.group_by("hs6").agg(_world_usd=value.sum())
    keys = rows.select(
        _e=pl.col("exporter").cast(pl.String),
        _i=pl.col("importer").cast(pl.String),
        _p=pl.col("hs6").cast(pl.String),
        _own_usd=pl.col("value_usd").cast(pl.Float64),
    )
    joined = (
        keys.join(supply.rename({"exporter": "_e", "hs6": "_p"}), on=["_e", "_p"], how="left", maintain_order="left")
        .join(demand.rename({"importer": "_i", "hs6": "_p"}), on=["_i", "_p"], how="left", maintain_order="left")
        .join(world.rename({"hs6": "_p"}), on="_p", how="left", maintain_order="left")
        .fill_null(0.0)
    )
    own = pl.col("_own_usd")
    naive = joined.select(
        naive_usd=naive_from_totals(pl.col("_supply_usd") - own, pl.col("_demand_usd") - own, pl.col("_world_usd") - own)
    )
    return rows.with_columns(naive["naive_usd"])


def accuracy(scored: pl.DataFrame, markets: pl.DataFrame) -> dict:
    """Every metric for the model and the naive baseline: overall, by Sector, by region and by income level.

    `scored` has exporter, importer, hs6, value_usd, weight, `export_potential_usd` and
    `naive_usd`; `markets` maps `iso3` to `region` and, optionally, World Bank
    `income_level` (the bias review's stand-in for reporting quality; `by_income` is
    added only when it is there). Breakdowns are sorted weakest first (lowest
    within-Product rank correlation of the model).

    Excluded Products are never measured (ADR 0005). When `scored` carries
    `tariff_imputed`, `by_tariff` splits the rows into observed tariffs and imputed ones
    (carried forward from 2021, ADR 0006): always both, in `TARIFF_GROUPS` order, an empty
    side with zero rows.
    """
    scored = drop_excluded(scored)
    has_income = "income_level" in markets.columns
    regions = markets.select(
        *(["income_level"] if has_income else []), _i=pl.col("iso3").cast(pl.String), region=pl.col("region")
    )
    frame = (
        scored.with_columns(
            hs6=pl.col("hs6").cast(pl.String),
            _i=pl.col("importer").cast(pl.String),
        )
        .with_columns(sector=pl.col("hs6").str.slice(0, 2))
        .join(regions, on="_i", how="left")
        .with_columns(pl.col("region", *(["income_level"] if has_income else [])).fill_null("Unclassified"))
    )
    overall = _metrics(frame.with_columns(_all=pl.lit("all")), "_all").drop("group").row(0, named=True)
    groups = {"by_sector": "sector", "by_region": "region", **({"by_income": "income_level"} if has_income else {})}
    return {
        "overall": _clean(overall),
        **{
            name: [_clean(row) for row in _weakest_first(_metrics(frame, column)).iter_rows(named=True)]
            for name, column in groups.items()
        },
        **({"by_tariff": _tariff_split(frame)} if "tariff_imputed" in frame.columns else {}),
    }


def _tariff_split(frame: pl.DataFrame) -> list[dict]:
    labelled = frame.with_columns(
        _tariff=pl.when(pl.col("tariff_imputed")).then(pl.lit("imputed")).otherwise(pl.lit("observed"))
    )
    found = {row["group"]: row for row in _metrics(labelled, "_tariff").iter_rows(named=True)}
    return [_clean(found.get(group) or _empty_group(group)) for group in TARIFF_GROUPS]


def _empty_group(group: str) -> dict:
    counts = {"rows": 0, "non_zero_flows": 0, "products": 0, "ranked_products": 0, "actual_usd": 0.0}
    return {"group": group, **counts, **{f"{name}_{metric}": None for name in _PREDICTORS for metric in METRICS}}


def tariff_split_text(accuracy_by_scope: Mapping[str, dict], test_year: int) -> str:
    """How the UK pairs of the test year split between observed and imputed Applied Tariffs, in plain words."""
    split = {row["group"]: row for row in accuracy_by_scope[HEADLINE_SCOPE].get("by_tariff", [])}
    if not split:
        return "The test rows carry no tariff mark, so they are not split by imputed and observed tariffs."
    observed, imputed = split["observed"]["rows"], split["imputed"]["rows"]
    if observed == 0:
        return (
            f"Every {test_year} UK pair uses an Applied Tariff carried forward from {LAST_TRAINS_YEAR}, because tariff "
            f"data end in {LAST_TRAINS_YEAR}, so no test row has an observed tariff to compare with: all the accuracy "
            f"above is measured on carried-forward tariffs."
        )
    return (
        f"Of the {observed + imputed:,} UK pairs in {test_year}, {imputed:,} use an Applied Tariff carried forward "
        f"from {LAST_TRAINS_YEAR} and {observed:,} an observed one. The table compares the model's accuracy on each."
    )


def _metrics(frame: pl.DataFrame, group: str) -> pl.DataFrame:
    sums = frame.group_by(group).agg(
        rows=pl.len(),
        non_zero_flows=(pl.col("value_usd") > 0).sum(),
        products=pl.col("hs6").n_unique(),
        actual_usd=pl.col("value_usd").cast(pl.Float64).dot(pl.col("weight")),
        **{f"{name}_tweedie_deviance": tweedie_deviance(col) for name, col in _PREDICTORS.items()},
        **{f"{name}_rmse_log": _rmse_log(col) for name, col in _PREDICTORS.items()},
    )
    within = (
        frame.group_by(group, "hs6").len().filter(pl.col("len") >= MIN_RANKED_ROWS).group_by(group).agg(
            ranked_products=pl.len()
        )
    )
    keyed = with_tie_key(frame, market=_ROW_MARKET)  # hashed once for both predictors
    for name, col in _PREDICTORS.items():
        rho = within_product_correlations(keyed, col, by=[group]).group_by(group).agg(
            pl.col("rho").mean().alias(f"{name}_spearman_within_product")
        )
        within = within.join(rho, on=group, how="left")
    return (
        sums.join(within, on=group, how="left")
        .with_columns(pl.col("ranked_products").fill_null(0))
        .rename({group: "group"})
        .with_columns(pl.col("group").cast(pl.String))
    )


def _weakest_first(table: pl.DataFrame) -> pl.DataFrame:
    return table.sort("model_spearman_within_product", "group", nulls_last=True)


def _clean(row: dict) -> dict:
    """JSON-safe: NaN becomes None, numpy scalars become Python numbers."""
    out = {}
    for key, value in row.items():
        if isinstance(value, float) and math.isnan(value):
            value = None
        out[key] = value
    return out


def headline(overall: dict, test_year: int) -> dict:
    """The plain-English accuracy headline: the model's within-Product ranking against the naive baseline."""
    model = overall[f"model_{HEADLINE_METRIC}"]
    naive = overall[f"naive_{HEADLINE_METRIC}"]
    beats = model is not None and naive is not None and model > naive
    scale = "(1 would be a perfect ranking, 0 no better than chance)"
    if beats:
        degree = "only slightly better" if model - naive < NARROW_MARGIN else "better"
        verdict = (
            f"That is {degree} than the Naive Baseline (the UK's world market share times each Market's imports), "
            f"which scores {naive:.2f}."
        )
    else:
        verdict = (
            f"The model does not beat the Naive Baseline on this measure: the UK's world market share times each "
            f"Market's imports scores {naive:.2f}. Treat its rankings with extra caution."
        )
    text = (
        f"For a typical Product, the model's ranking of Markets for UK exports in {test_year}, a year it never "
        f"saw, has a rank correlation of {model:.2f} with what the UK actually exported {scale}. {verdict}"
    )
    return {
        "scope": HEADLINE_SCOPE,
        "metric": HEADLINE_METRIC,
        "model": model,
        "naive": naive,
        "beats_naive": beats,
        "text": text,
    }


# v2's 2023 figures, from its evaluation report (artefact 20261118T100524Z, model alone). Its naive
# baseline was computed from the test year's training rows (top-100 Exporters' Flows and sampled
# Zero Cells); since spec #65 the naive baseline comes from every Exporter's Flows, so this run's
# naive figure is reported beside it, not in place of it.
V2_REPORTED = {
    "version": "20261118T100524Z",
    "uk_ranking": 0.5114955233364546,
    "naive": 0.5233246980004151,
    "lift": 5.226937575036582,
    "size_matched_lift": 0.7876591646085601,
    "tweedie_deviance_all_exporters": 355.5288087632071,
    "tweedie_deviance_uk": 568.5883343927953,
}
NAIVE_THIS_RUN = "Naive Baseline, every Exporter's Flows (this run)"
NAIVE_AS_REPORTED_FOR_V2 = "Naive Baseline as reported for v2 (training rows' Flows)"
# The names reports saved before review2 used, and today's (GLOSSARY.md), for `relabelled`.
_SAVED_NAMES = {
    "Naive baseline, every Exporter's Flows (this run)": NAIVE_THIS_RUN,
    "Naive baseline as reported for v2 (training rows' Flows)": NAIVE_AS_REPORTED_FOR_V2,
}
_SAVED_THIS_VERSION = "This version ("


def version_name(form: str) -> str:
    """This version's row in `earlier_versions`: the Blend (GLOSSARY.md) when it serves one, else the model alone."""
    return "Blend" if form == "blend" else f"{_SAVED_THIS_VERSION}{form})"


def earlier_versions(report: dict, *, v2: dict | None = V2_REPORTED) -> dict:
    """The test year's UK ranking, size-matched Lift and Tweedie deviance beside v2's and v1's.

    `report` is this version's evaluation report; v1's UK ranking is the v1 comparison's (the same
    test-year rows). `v2` holds v2's reported figures (`V2_REPORTED`), or None where they do not
    apply (the fixture). Both naive figures are listed, each named by the Flows it was computed from.
    Whether the model beats the naive baseline is judged against this run's naive baseline.
    """
    head, year = report["headline"], report["test_year"]
    overall = {scope: report["accuracy"][scope]["overall"] for scope in ("all_exporters", "uk_out_of_fold")}
    lift = report.get("lift") or {}
    form = (report.get("model_form") or {}).get("form", "model alone")
    rows = [{
        "name": version_name(form),
        "uk_ranking": head["model"],
        "lift": lift.get("lift"),
        "size_matched_lift": (lift.get("size_matched") or {}).get("lift"),
        "tweedie_deviance_all_exporters": overall["all_exporters"]["model_tweedie_deviance"],
        "tweedie_deviance_uk": overall["uk_out_of_fold"]["model_tweedie_deviance"],
    }]
    naive = [{"name": NAIVE_THIS_RUN, "uk_ranking": head["naive"]}]
    if v2 is not None:
        rows.append({"name": "v2", **{k: v for k, v in v2.items() if k != "naive"}})
        naive.append({"name": NAIVE_AS_REPORTED_FOR_V2, "uk_ranking": v2["naive"]})
    v1 = (report.get("v1_comparison") or {}).get("primary", {}).get("v1")
    if v1 is not None:
        rows.append({"name": "v1", "uk_ranking": v1})
    beats = head["model"] is not None and head["naive"] is not None and head["model"] > head["naive"]
    block = {"test_year": year, "rows": rows, "naive": naive, "beats_naive": beats}
    return {**block, "text": _earlier_versions_text(block)}


def _earlier_versions_text(block: dict) -> str:
    """The `earlier_versions` sentence, worded from the block's rows."""
    this, *others = block["rows"]
    naive = {row["name"]: row["uk_ranking"] for row in block["naive"]}
    verdict = "beats the Naive Baseline" if block["beats_naive"] else "does not beat the Naive Baseline"
    reported = naive.get(NAIVE_AS_REPORTED_FOR_V2)
    earlier = ", ".join(f"{row['name']} {row['uk_ranking']:.3f}" for row in others)
    return (
        f"On {block['test_year']}, read once, the UK ranking of this version is {this['uk_ranking']:.3f}: it {verdict} "
        f"({naive[NAIVE_THIS_RUN]:.3f} from every Exporter's Flows"
        + (f"; {reported:.3f} as reported for v2" if reported is not None else "")
        + ")."
        + (f" Earlier versions: {earlier}." if earlier else "")
        + " These are associations on one test year, not a forecast."
    )


def relabelled(report: dict) -> dict:
    """A saved evaluation report with today's names (the Blend, the Naive Baseline; GLOSSARY.md) in its
    headline and `earlier_versions` block, worded again from their saved figures. Nothing is recomputed, keys keep their
    names and the saved report is not changed. `comparison.relabelled` also rewords the v1 comparison."""
    saved = report.get("headline")
    if saved and saved.get("model") is not None and report.get("test_year") is not None:
        worded = headline({f"model_{HEADLINE_METRIC}": saved["model"], f"naive_{HEADLINE_METRIC}": saved["naive"]},
                          report["test_year"])
        report = {**report, "headline": {**saved, "text": worded["text"]}}
    block = report.get("earlier_versions")
    if not block:
        return report
    form = (report.get("model_form") or {}).get("form", "model alone")
    this, *others = block["rows"]
    if this["name"].startswith(_SAVED_THIS_VERSION):
        this = {**this, "name": version_name(form)}
    naive = [{**row, "name": _SAVED_NAMES.get(row["name"], row["name"])} for row in block["naive"]]
    block = {**block, "rows": [this, *others], "naive": naive}
    return {**report, "earlier_versions": {**block, "text": _earlier_versions_text(block)}}


def stability(current: pl.DataFrame, previous: pl.DataFrame, *, top_n: int = STABILITY_TOP_N) -> dict:
    """How much UK rankings move between two years scored by the same models.

    Both tables hold hs6, iso3, export_potential_usd and export_gap_usd. Reports the mean
    within-Product rank correlation of Export Potential and of Export Gap between the years
    (the Selection Measure's, ties in this year's figures broken by the tie key), and the share of the top `top_n` Opportunities (by Export Gap) shared by both.
    """
    columns = ["hs6", "iso3", "export_potential_usd", "export_gap_usd"]
    joined = current.select(columns).join(previous.select(columns), on=["hs6", "iso3"], suffix="_prev")
    n = max(1, min(top_n, joined.height))
    top_now = set(joined.top_k(n, by="export_gap_usd").select("hs6", "iso3").iter_rows())
    top_before = set(joined.top_k(n, by="export_gap_usd_prev").select("hs6", "iso3").iter_rows())
    return {
        "pairs": joined.height,
        "spearman_potential": selection_measure(
            joined, "export_potential_usd", target="export_potential_usd_prev", market="iso3"
        ),
        "spearman_gap": selection_measure(joined, "export_gap_usd", target="export_gap_usd_prev", market="iso3"),
        "top_n": n,
        "top_overlap": len(top_now & top_before) / n,
    }


# --- Test-year models and scoring -------------------------------------------------------


@dataclass(frozen=True)
class TestModels:
    """Fold models refitted on every year before the test year, their feature transform and inputs."""

    test_year: int
    training_years: list[int]
    models: list[xgb.Booster]
    transform: FeatureTransform
    features: list[str] = field(default_factory=lambda: list(FEATURES))  # the models' inputs, in order

    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for k, model in enumerate(self.models):
            model.save_model(folder / f"fold_{k}.ubj")
        info = {
            "test_year": self.test_year,
            "training_years": self.training_years,
            "n_folds": len(self.models),
            "transform": self.transform.to_dict(),
            "features": list(self.features),
        }
        # Written last: its presence marks the folder as complete.
        (folder / TEST_MODELS_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> "TestModels":
        folder = Path(folder)
        info = json.loads((folder / TEST_MODELS_FILE).read_text(encoding="utf-8"))
        models = []
        for k in range(info["n_folds"]):
            model = xgb.Booster()
            model.load_model(folder / f"fold_{k}.ubj")
            models.append(model)
        return cls(
            info["test_year"],
            info["training_years"],
            models,
            FeatureTransform.from_dict(info["transform"]),
            info.get("features", list(FEATURES)),  # folders saved before the ladder hold FEATURES
        )


def train_test_models(
    training: Mapping[int, Chunk],
    *,
    test_year: int,
    uk: str,
    params: dict,
    rounds: int,
    n_folds: int = N_FOLDS,
    device: str = "cuda",
    non_uk_rate: float = 1.0,
    seed: int = 0,
    features: Sequence[str] = FEATURES,
) -> TestModels:
    """Fit the fold models on every year before `test_year`; the test year itself is never read.

    `features` are the models' inputs (default: v2's; a gain-ladder rung uses v1's).
    """
    years = [y for y in sorted(training) if y < test_year]
    if not years:
        raise ValueError(f"No training years before the test year {test_year}")
    chunks = [training[y] for y in years]
    transform = FeatureTransform.fit(scan_chunks(chunks), years)
    models = train_fold_models(
        chunks, uk=uk, params=params, rounds=rounds, n_folds=n_folds, device=device, seed=seed,
        transform=transform, non_uk_rate=non_uk_rate, features=features,
    )
    return TestModels(
        test_year=test_year, training_years=years, models=models, transform=transform, features=list(features)
    )


def score_rows(
    table: pl.DataFrame,
    models: Sequence[xgb.Booster],
    transform: FeatureTransform,
    uk: str,
    *,
    features: Sequence[str] = FEATURES,
    form: ModelForm = MODEL_ALONE,
) -> pl.DataFrame:
    """Add `export_potential_usd`: UK rows out-of-fold, other rows by the mean of the fold models.

    `features` are the models' inputs, in order; `form` is the model form (a blend needs `naive_usd`).
    """
    transformed = transform.apply(table)
    is_uk = pl.col("exporter").cast(pl.String) == uk
    uk_rows = score_out_of_fold(transformed, list(models), uk=uk, inputs=features, form=form).drop("fold")
    others = transformed.filter(~is_uk)
    if others.height:
        others = others.with_columns(export_potential(others, models, features=features, by="mean", form=form))
    else:
        others = others.with_columns(export_potential_usd=pl.lit(None, pl.Float64))
    return pl.concat([uk_rows.select(others.columns), others])


def _uk_table(scored_grid: pl.DataFrame) -> pl.DataFrame:
    return scored_grid.select(
        pl.col("hs6").cast(pl.String),
        iso3=pl.col("importer").cast(pl.String),
        export_potential_usd="export_potential_usd",
        export_gap_usd=pl.col("export_potential_usd") - pl.col("value_usd"),
    )


def evaluate(
    test_rows: pl.DataFrame,
    test_grid: pl.DataFrame,
    *,
    test_models: TestModels,
    baseline: pl.DataFrame,
    baseline_models: Sequence[xgb.Booster],
    baseline_transform: FeatureTransform,
    markets: pl.DataFrame,
    uk: str,
    settings: dict,
    flows: pl.DataFrame | None = None,
    form: ModelForm = MODEL_ALONE,
) -> dict:
    """The evaluation report for the test year.

    `test_rows` are the test year's training rows (every non-zero Flow plus sampled zeros),
    `test_grid` its full UK grid, both with features untransformed. `baseline` holds the
    Baseline year's UK scores (`gap_period.baseline_year_scores`) and `baseline_models` /
    `baseline_transform` the models that scored them, used for the stability check. `settings` (validation year, params, sampling rates, ...)
    is recorded in the report. `flows` are the test year's Flows the naive baseline is computed
    from (default: the non-zero Flows among `test_rows`); `form` is the model form the test-year
    models are scored in (a blend takes that naive baseline).
    """
    year = test_models.test_year
    if set(test_rows["year"].unique().to_list()) != {year} or set(test_grid["year"].unique().to_list()) != {year}:
        raise ValueError(f"Test rows and grid must hold the test year {year} only")
    if year in test_models.training_years:
        raise ValueError("The test-year models were trained on the test year")

    if flows is None:
        flows = test_rows.filter(pl.col("value_usd") > 0)
    flows = flows.select("exporter", "importer", "hs6", "value_usd")
    test_rows, test_grid = naive_potential(test_rows, flows), naive_potential(test_grid, flows)
    scored_rows = score_rows(test_rows, test_models.models, test_models.transform, uk, form=form)
    scored_grid = score_rows(test_grid, test_models.models, test_models.transform, uk, form=form)
    accuracy_by_scope = {
        "all_exporters": accuracy(scored_rows, markets),
        "uk_out_of_fold": accuracy(scored_grid, markets),
    }

    previous = _uk_table(
        score_out_of_fold(baseline_transform.apply(test_grid), list(baseline_models), uk=uk, form=form)
    )
    baseline_year = year + 1
    stable = stability(baseline, previous)
    stable["years"] = [year, baseline_year]
    stable["text"] = _stability_text(stable, year, baseline_year)

    return {
        "test_year": year,
        "test_training_years": test_models.training_years,
        **settings,
        "model_form": form.to_dict(),
        "tweedie_power": TWEEDIE_POWER,
        "min_prediction_usd": MIN_PREDICTION_USD,
        "metrics": {name: METRIC_INFO[name] for name in METRICS},
        "naive_baseline": "The Exporter's share of the Product's world exports times the Importer's imports "
        "of that Product, each leaving the Flow itself out.",
        "accuracy": accuracy_by_scope,
        "headline": headline(accuracy_by_scope[HEADLINE_SCOPE]["overall"], year),
        "tariff_split_text": tariff_split_text(accuracy_by_scope, year),
        "stability": stable,
    }


def _stability_text(result: dict, year: int, baseline_year: int) -> str:
    potential = result["spearman_potential"]
    agreement = "n/a" if potential is None else f"{potential:.2f}"
    return (
        f"Scored by the same models, each Product's ranking of Markets by Export Potential in {baseline_year} "
        f"has a rank correlation of {agreement} with its {year} ranking, and {result['top_overlap']:.0%} of the "
        f"top {result['top_n']:,} Opportunities by Export Gap are the same in both years."
    )


# --- Real data ----------------------------------------------------------------------------

# Metadata keys of the trained artefacts that the evaluation report records.
SETTINGS_KEYS = (
    "validation_year",
    "tuning_years",
    "params",
    "rounds",
    "n_folds",
    "non_uk_sampling_rate",
    "zero_sampling_rate",
)


def report_settings(metadata: dict) -> dict:
    """The training settings an evaluation report records, from the artefact metadata."""
    return {key: metadata[key] for key in SETTINGS_KEYS}


def test_models_dir(root: Path, metadata: dict) -> Path:
    """Where real test-year models are cached under the data root, keyed by their training settings."""
    from exportspredictor.features import features_dir

    settings = {**report_settings(metadata), "model_params": metadata["model_params"], "test_year": metadata["baseline_year"] - 1}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return features_dir(root).parent / "test_models" / f"{settings['test_year']}_{key}"


def real_test_models(artefacts, root: Path | None = None, *, device: str = "cuda") -> TestModels:
    """The real test-year fold models for an artefact: from the cache, or trained (about 20 min on GPU).

    They use the artefact's tuned parameters, round count, sampling rate and seed, so they
    differ from its Baseline models only in leaving out the test year.
    """
    root = Path(root or data_dir())
    metadata = artefacts.metadata
    test_year = metadata["baseline_year"] - 1
    cache = test_models_dir(root, metadata)
    if (cache / TEST_MODELS_FILE).exists():
        return TestModels.load(cache)
    folder = features_dir(root)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    training = {y: folder / f"training_{y}.parquet" for y in settings["years"] if y < test_year}
    models = train_test_models(
        training,
        test_year=test_year,
        uk=baci.UK,
        params=metadata["params"],
        rounds=metadata["rounds"],
        n_folds=metadata["n_folds"],
        device=device,
        non_uk_rate=metadata["non_uk_sampling_rate"],
        seed=metadata["model_params"].get("seed", 0),
    )
    models.save(cache)
    return models


def evaluate_real(artefacts, test_models: TestModels, root: Path | None = None) -> dict:
    """The evaluation report of a real artefact, from the per-year feature files, in its model form."""
    folder = features_dir(root)
    year = test_models.test_year
    return evaluate(
        pl.read_parquet(folder / f"training_{year}.parquet"),
        pl.read_parquet(folder / f"uk_grid_{year}.parquet"),
        test_models=test_models,
        baseline=baseline_year_scores(artefacts.history, artefacts.baseline_year),
        baseline_models=artefacts.fold_models,
        baseline_transform=FeatureTransform.from_dict(artefacts.metadata["feature_transform"]),
        markets=markets_with_income(artefacts.markets, root),
        uk=baci.UK,
        settings=report_settings(artefacts.metadata),
        flows=load_year_flows(year, load_sources(root)[1], root),
        form=artefacts.model_form,
    )


def markets_with_income(markets: pl.DataFrame, root: Path | None = None) -> pl.DataFrame:
    """The artefact's Market lookup plus World Bank `income_level` from the BACI lookups (keyed by standard ISO3)."""
    if "income_level" in markets.columns:
        return markets
    country_map = pl.read_parquet((root or data_dir()) / "derived" / "country_map.parquet")
    income = (
        baci.markets(root)
        .select(baci_iso3="iso3", income_level="income_level")
        .join(country_map.select("baci_iso3", "iso3").unique("baci_iso3"), on="baci_iso3", how="left")
        .select(iso3=pl.col("iso3").fill_null(pl.col("baci_iso3")), income_level="income_level")
        .unique("iso3")
    )
    return markets.join(income, on="iso3", how="left")


def add_to_latest(
    root: Path | None = None,
    artefacts_root: Path | None = None,
    *,
    device: str = "cuda",
    test_year_ledger: ledger.Ledger | None,
) -> Path:
    """Evaluate the newest artefact and save it again, with the report, as a new version.

    The Baseline models and tables are copied unchanged; only the test-year models are
    trained (or read from the cache). Returns the new folder. The read is recorded in
    `test_year_ledger` first and refused if the newest version's configuration was read before
    (ADR 0009); `None` only on synthetic data.
    """
    artefacts_root = Path(artefacts_root or artefacts_dir())
    source = store.latest_folder(artefacts_root)
    bundle = store.load(source)
    metadata = {**bundle.metadata, "version": store.new_version(), "copied_from": bundle.version}
    if test_year_ledger is not None:
        ledger.record_read(test_year_ledger, metadata, read="test-year evaluation")
    started = time.perf_counter()
    test_models = real_test_models(bundle, root, device=device)
    report = evaluate_real(bundle, test_models, root)
    report["evaluation_seconds"] = round(time.perf_counter() - started, 2)
    return store.copy_version(source, artefacts_root, metadata=metadata, evaluation=report)


if __name__ == "__main__":
    import sys

    command = ledger.Ledger.for_command("python -m exportspredictor.evaluation", sys.argv[1:])
    print(add_to_latest(test_year_ledger=command), flush=True)
