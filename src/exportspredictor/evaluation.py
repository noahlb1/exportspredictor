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

Beside the headline, `large_markets` gives the same measure on the UK's 30 and 60 largest
importing Markets (`large_markets`); nothing is chosen on it.

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
from exportspredictor import (
    baci,
    ledger,
    market_reach,
    offsets,
    realisation,
    realisation_wording,
)
from exportspredictor.conformal import COVERAGE, COVERAGE_TOLERANCE, coverage_text
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
from exportspredictor.large_markets import (
    LARGE_MARKET_COUNTS,
    large_markets_report,
    restricted_measures,
)
from exportspredictor.paths import artefacts_dir, data_dir
from exportspredictor.prediction import (
    MIN_PREDICTION_USD,
    MODEL_ALONE,
    ModelForm,
    export_potential,
)
from exportspredictor.sectors import sector_of
from exportspredictor.selection_measure import (
    MIN_RANKED_ROWS,
    legacy_selection_measure,
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
    if served.get(realisation.REPORT_KEY):
        served[realisation.REPORT_KEY] = realisation_wording.served(served[realisation.REPORT_KEY])
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
# Parts added to a saved version by their own command, not by a pipeline switch: title, report block, command, why.
ADDED_PARTS = (
    ("ITC comparison", realisation.REPORT_KEY, "python -m exportspredictor.realisation",
     "It runs ITC's realisation test, the one behind its Export Potential Map, on the model's Export Potential."),
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
    ] + [
        {"title": title, "text": f"{title}: {NOT_RUN}. {why} It is added to a saved version with `{command}`."}
        for title, block, command, why in ADDED_PARTS
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
    {
        # Review flag H1 (#161).
        "title": "A Tariff part is not a tariff elasticity",
        "text": "The model's response to an Applied Tariff is more a step than a slope: it separates duty-free pairs "
        "from dutiable ones, which differ in more than their tariff. So removing a tariff of 1% or less can be "
        "associated with a rise of tens of percent in Export Potential (on the 2024 Baseline, turbo-jets to China "
        "at 1%: +37%), far more per tariff point than trade elasticities suggest. A Tariff part is Illustrative: the "
        "part of the agreement association the model ties to the tariff, not the priced effect of a tariff cut.",
    },
    {
        # Review flag H3 (#161).
        "title": "Some Trade Agreements carry most-favoured-nation tariffs",
        "text": "TRAINS records no preferential rates in some Markets' schedules where a Trade Agreement is in force, "
        "so every Exporter with an agreement there carries the Market's MFN rate. Among UK Markets these are Canada, "
        "Israel, Serbia, Vietnam, Ghana, Côte d'Ivoire, Cameroon and Kenya. The UK left the EU customs union, and "
        "with it its customs union with Türkiye, at the end of 2020. TRAINS holds no rates under the UK–Türkiye "
        "agreement that followed, so in the data Türkiye charges UK goods its MFN rates from then on (the tariff data "
        f"end in {LAST_TRAINS_YEAR}, and later years carry those rates forward). These pairs have a Trade Agreement, so they have no Blocked Opportunity "
        "value or Tariff part, but their Export Potential reads them as dutiable.",
    },
]


def known_limitations(artefacts: store.Artefacts) -> list[dict]:
    """Every known limitation of an artefact version that can be read from it: its results, its
    Potential Range bands and the fixed `LIMITATIONS`. The API serves these, and the model card states them all
    (`governance.model_card`, the one source) before its own extra, the gravity benchmark. They read only figures that
    `relabelled` leaves as saved, so the card's relabelled report would give the same list.
    """
    report = artefacts.evaluation or {}
    return [
        *result_limitations(artefacts.metadata, report),
        *range_limitations(artefacts.potential_range_bands),
        *range_below_potential_limitations(artefacts.potential_range_bands),
        *size_coverage_limitations(report.get("potential_range")),
        *uk_pair_coverage_limitations(artefacts.metadata, report.get("potential_range")),
        *offset_pair_limitations(artefacts.metadata),
        *read_configuration_limitations(artefacts.metadata, report),
        *LIMITATIONS,
    ]


def range_below_potential_limitations(bands: pl.DataFrame) -> list[dict]:
    """When a log-scale band's upper offset is negative, its pairs' whole Potential Range lies below their Export
    Potential (review flag M4, #161). Stated up to the lower edge of the band above the last such band; none for
    dollar-offset versions or when every upper offset is positive."""
    if bands.is_empty() or "high_offset_log" not in bands.columns:
        return []
    ordered = bands.sort("min_export_potential_usd")
    negative = [i for i, offset in enumerate(ordered["high_offset_log"]) if offset < 0]
    if not negative:
        return []
    above = negative[-1] + 1
    size = (
        f"under {format_usd(ordered['min_export_potential_usd'][above])}" if above < ordered.height else "of every size"
    )
    return [{
        "title": "The Potential Range lies below Export Potential for small pairs",
        "text": f"For many pairs with Export Potential {size}, the whole Potential Range lies below Export Potential. "
        "Most comparable pairs of that size trade nothing, and Export Potential is an average that the few which do "
        "trade pull up, so at least 90% of comparable pairs trade less than it. A range such as $0 to $200 beside an "
        "Export Potential of $1.3k is that pattern, not an error.",
    }]


# A Model Form's fitted values, which a rebuild of the same configuration recomputes (ADR 0010): no change of form.
_FITTED_FORM_KEYS = ("reference_year", "constant")


def _form_settings(form: dict | None) -> dict:
    form = dict(form or {})
    reach = {k: v for k, v in (form.pop("market_reach", None) or {}).items() if k not in _FITTED_FORM_KEYS}
    return {**form, **{f"Market Reach {k}": v for k, v in reach.items()}}


def read_configuration_limitations(metadata: dict, report: dict) -> list[dict]:
    """When the served Model Form differs from the one the report's test-year figures were read on, as after `rereach`
    corrected Market Reach's form without a new read (ADR 0009, ADR 0010 Amendment). None for a version without a
    report, or one whose report records no Model Form."""
    served, read_on = _form_settings(metadata.get("model_form")), _form_settings(report.get("model_form"))
    if not report.get("model_form") or served == read_on:
        return []
    changes = "; ".join(
        f"{key} {read_on.get(key)!s} then, {served.get(key)!s} now"
        for key in sorted(set(served) | set(read_on)) if served.get(key) != read_on.get(key)
    )
    year = report.get("test_year", "the test year")
    return [{
        "title": "The test-year figures were read on an earlier configuration",
        "text": f"Every {year} figure shown (the headline, Lift and the comparison with earlier versions) was read on "
        f"the Model Form served before a correction chosen on the validation year ({changes}). This "
        f"version has not read {year} itself: its own figures come from the next frozen version's one read through "
        "the Test-year Ledger (ADR 0009).",
    }]


def offset_pair_limitations(metadata: dict) -> list[dict]:
    """When a version's Baseline holds Offset Pairs (`offsets.METADATA_KEY`, spec #129): the model can be right about a
    Heading and wrong about the split between its Products. None for a version that records no Offset Pair."""
    saved = metadata.get(offsets.METADATA_KEY)
    if not saved:
        return []
    reach = offsets.OffsetPairReach.from_dict(saved)
    if not reach.offset_pair_count:
        return []
    return [{
        "title": "Right about a Heading, wrong about the split between its Products",
        "text": "The model can be right about a Heading and wrong about the split between its Products, so an Export "
        "Gap can sit beside a surplus one Product away in the same Market. No feature can see the split without "
        "reading the UK's own trade with the pair's Market. Opportunities whose Export Gap the rest of their Heading "
        f"offsets by at least {reach.threshold:.0%} are marked as Offset Pairs: {reach.offset_pair_count:,} of "
        f"{reach.opportunities:,} Opportunities ({reach.share_of_opportunities:.0%}), {reach.in_top_text()}. Their "
        "Export Gap may be trade recorded under a neighbouring Product.",
    }]


# UK-pair coverage further than this below the Potential Range's target is a known limitation (spec #113 story 13).
UK_PAIR_COVERAGE_TOLERANCE = 0.05


def uk_pair_coverage_limitations(metadata: dict, potential_range: dict | None) -> list[dict]:
    """When fewer UK pairs than the target, less `UK_PAIR_COVERAGE_TOLERANCE`, have their Gap Period Actual Exports inside
    their Potential Range. Read from the report's `uk_pair_coverage`, or, for a version upgraded by `rereach`, its
    metadata's `market_reach_pair_coverage`; none for a version that records neither."""
    found = (potential_range or {}).get("uk_pair_coverage")
    saved = metadata.get("market_reach_pair_coverage")
    if found:
        coverage, without = found["with_market_reach"], found["without_market_reach"]
    elif saved:
        coverage, without = saved["after"], saved["before"]
    else:
        return []
    target = (potential_range or {}).get("target", COVERAGE)
    if coverage is None or round(target - coverage, 9) <= UK_PAIR_COVERAGE_TOLERANCE:
        return []
    return [{
        "title": "The Potential Range holds fewer UK pairs than its target",
        "text": f"Only {coverage:.1%} of UK pairs have their Gap Period Actual Exports inside their Potential Range"
        + ("" if without is None else f" ({without:.1%} without Market Reach)")
        + f", against a target of {target:.0%}. The range is "
        "calibrated on other Exporters' pairs (ADR 0007), and Market Reach moves UK Export Potential by a Market-wide "
        "factor those pairs do not take (ADR 0010), so for UK pairs the range understates how far Actual Exports "
        "vary around Export Potential.",
    }]


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
        narrow = headline["model"] - headline["naive"] < NARROW_MARGIN  # as the headline words the margin
        limits.append({"title": "Only slightly better than a naive ranking for UK pairs" if narrow
                       else "Ranks Markets better than it sizes them",
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
        .with_columns(sector=sector_of("hs6"))
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
# The Blend's 2023 figures, from its evaluation report (artefact 20261118T185910Z, the ledger's third read), on the
# old measure: the version served before spec #113. Its naive baseline came from every Exporter's Flows, as this
# run's does, so it is not listed apart.
BLEND_REPORTED = {
    "version": "20261118T185910Z",
    "uk_ranking": 0.555192379963398,
    "lift": 4.993143365742925,
    "size_matched_lift": 0.760532455406604,
    "tweedie_deviance_all_exporters": 433.76459865636,
    "tweedie_deviance_uk": 566.3460040913492,
}
# The configuration fingerprints of the earlier versions the ledger was seeded with (`docs/test-year-ledger.csv`):
# re-scoring their saved scores (`rescore_saved`) is recorded against them (`ledger.record_rescore`).
EARLIER_FINGERPRINTS = {"Blend": "f4d0be3058fd", "v2": "f7cdd149f731", "v1": "a77726cf17b0"}
TIE_PROOF_MEASURE = "tie-proof"  # the measure earlier versions' saved scores are re-scored under
NAIVE_THIS_RUN = "Naive Baseline, every Exporter's Flows (this run)"
NAIVE_AS_REPORTED_FOR_V2 = "Naive Baseline as reported for v2 (training rows' Flows)"
# The names reports saved before review2 used, and today's (GLOSSARY.md), for `relabelled`.
_SAVED_NAMES = {
    "Naive baseline, every Exporter's Flows (this run)": NAIVE_THIS_RUN,
    "Naive baseline as reported for v2 (training rows' Flows)": NAIVE_AS_REPORTED_FOR_V2,
}
_SAVED_THIS_VERSION = "This version ("


def model_form_label(model_form: Mapping | None) -> str:
    """A saved Model Form (`ModelForm.to_dict`) in a few words: "Blend", "model alone", each "with Market Reach"
    where it has one."""
    form = (model_form or {}).get("form", "model alone")
    name = "Blend" if form == "blend" else form
    return f"{name} with Market Reach" if (model_form or {}).get("market_reach") else name


def version_name(model_form: Mapping | None) -> str:
    """This version's row in `earlier_versions`: the Blend (GLOSSARY.md) when it serves one without Market Reach,
    else this version named by its Model Form, so a new version is never confused with the earlier Blend."""
    label = model_form_label(model_form)
    return label if label == "Blend" else f"{_SAVED_THIS_VERSION}{label})"


def old_measure(scored_grid: pl.DataFrame) -> dict:
    """The headline on the old measure (`legacy_selection_measure`, average ranks for ties) for the model and the
    Naive Baseline: what v1, v2 and the Blend reported on the test year, so this version stands beside them.
    `scored_grid` is the UK's scored grid of the test year; Excluded Products are left out. Nothing is chosen on it."""
    rows = drop_excluded(scored_grid)
    return {name: legacy_selection_measure(rows, column) for name, column in _PREDICTORS.items()}


def rescore_saved(
    scores: Mapping[str, pl.DataFrame], *, flows: pl.DataFrame | None = None, counts: Sequence[int] = LARGE_MARKET_COUNTS
) -> dict[str, dict]:
    """Earlier versions' saved test-year UK scores on both measures: name -> `uk_ranking` (the old measure, which
    should match what the version reported) and `uk_ranking_tie_proof` (the Selection Measure). With `flows` (every
    Exporter's Flows of the test year), also `large_markets`: the Selection Measure on the `counts` largest
    importing Markets, as this version's `large_markets` block measures the headline.

    Each table holds one row per UK Product and Market (`hs6`, `importer`, `value_usd`, `export_potential_usd`).
    Nothing is trained, predicted or chosen. The caller records the re-score in the test-year ledger first
    (`ledger.record_rescore` with `EARLIER_FINGERPRINTS` and `TIE_PROOF_MEASURE`, ADR 0009) and puts the result in
    the report with `with_rescored_earlier_versions`.
    """
    out = {}
    for name, table in scores.items():
        # Ties are broken by the same key as this version's headline (`_ROW_MARKET`: the UK as Exporter).
        rows = drop_excluded(table.with_columns(
            pl.col("hs6").cast(pl.String), pl.col("importer").cast(pl.String), exporter=pl.lit(baci.UK)
        ))
        out[name] = {
            "uk_ranking": legacy_selection_measure(rows, "export_potential_usd"),
            "uk_ranking_tie_proof": selection_measure(rows, "export_potential_usd", market=_ROW_MARKET),
        }
        if flows is not None:
            large = restricted_measures(
                rows, flows, "export_potential_usd", counts=counts,
                measure=lambda part, column: selection_measure(part, column, market=_ROW_MARKET),
            )
            out[name]["large_markets"] = [{"markets": n, "model": large[n]} for n in counts]
    return out


def with_rescored_earlier_versions(
    report: dict, rescored: Mapping[str, Mapping], *, v2: dict | None = V2_REPORTED, blend: dict | None = BLEND_REPORTED
) -> dict:
    """`report` with earlier versions' re-scored figures (`rescore_saved`, keyed by row name: "Blend", "v2", "v1")
    and its `earlier_versions` block laid out again with them. The given report is not changed."""
    out = {**report, "rescored_earlier_versions": {name: dict(figures) for name, figures in rescored.items()}}
    return {**out, "earlier_versions": earlier_versions(out, v2=v2, blend=blend)}


def earlier_versions(report: dict, *, v2: dict | None = V2_REPORTED, blend: dict | None = BLEND_REPORTED) -> dict:
    """The test year's UK ranking, size-matched Lift and Tweedie deviance beside the Blend's, v2's and v1's.

    Every row's `uk_ranking` is the old measure (average ranks for ties), the one every earlier version
    reported, and `uk_ranking_tie_proof` the Selection Measure, so the two are never mixed. This version
    has both (`report["old_measure"]` and its headline). An earlier version's tie-proof figure comes from
    re-scoring its saved scores (`report["rescored_earlier_versions"]`, see `with_rescored_earlier_versions`)
    and is None until then. `blend` and `v2` hold their reported figures (`BLEND_REPORTED`, `V2_REPORTED`),
    or None where they do not apply (the fixture); v1's comes from the v1 comparison (the same test-year
    rows, old measure). Both naive figures are listed, each named by the Flows it was computed from. Whether
    the model beats the naive baseline is judged on the headline (tie-proof) against this run's.

    A report without `old_measure` was saved before the two measures were told apart: its headline was the
    old measure, and its rows hold `uk_ranking` only.
    """
    head, year = report["headline"], report["test_year"]
    overall = {scope: report["accuracy"][scope]["overall"] for scope in ("all_exporters", "uk_out_of_fold")}
    lift = report.get("lift") or {}
    old = report.get("old_measure")
    rescored = report.get("rescored_earlier_versions") or {}

    def measures(name: str, reported: float | None, tie_proof: float | None = None) -> dict:
        if old is None:
            return {"uk_ranking": reported}
        return {"uk_ranking": reported,
                "uk_ranking_tie_proof": (rescored.get(name) or {}).get("uk_ranking_tie_proof", tie_proof)}

    rows = [{
        "name": version_name(report.get("model_form")),
        **measures("this version", head["model"] if old is None else old["model"], head["model"]),
        "lift": lift.get("lift"),
        "size_matched_lift": (lift.get("size_matched") or {}).get("lift"),
        "tweedie_deviance_all_exporters": overall["all_exporters"]["model_tweedie_deviance"],
        "tweedie_deviance_uk": overall["uk_out_of_fold"]["model_tweedie_deviance"],
    }]
    naive = [{"name": NAIVE_THIS_RUN,
              **measures("this run's naive", head["naive"] if old is None else old["naive"], head["naive"])}]
    for name, reported in (("Blend", blend), ("v2", v2)):
        if reported is not None:
            figures = {k: v for k, v in reported.items() if k != "naive"}
            rows.append({"name": name, **figures, **measures(name, reported["uk_ranking"])})
    if v2 is not None:
        naive.append({"name": NAIVE_AS_REPORTED_FOR_V2, **measures("v2's naive", v2["naive"])})
    v1 = (report.get("v1_comparison") or {}).get("primary", {}).get("v1")
    if v1 is not None:
        rows.append({"name": "v1", **measures("v1", v1)})
    beats = head["model"] is not None and head["naive"] is not None and head["model"] > head["naive"]
    block = {"test_year": year, "rows": rows, "naive": naive, "beats_naive": beats}
    return with_v1_lift({**block, "text": _earlier_versions_text(block)}, report.get("v1_comparison"))


def with_v1_lift(block: dict | None, v1_comparison: Mapping | None) -> dict | None:
    """An `earlier_versions` block whose v1 row has the Lift and size-matched Lift the v1 comparison measured for v1
    (on the same test year). A block saved before v1's row carried them is filled when read; others are unchanged."""
    lifts = {m["metric"]: m.get("v1") for m in (v1_comparison or {}).get("secondary", [])
             if m.get("metric") in ("lift", "size_matched_lift")}
    if block is None or not lifts:
        return block
    rows = [{**row, **{k: v for k, v in lifts.items() if row.get(k) is None}} if row["name"] == "v1" else row
            for row in block["rows"]]
    return {**block, "rows": rows}


def large_markets_beside_earlier(report: Mapping) -> list[dict] | None:
    """The tie-proof UK ranking on the largest importing Markets, this version beside the earlier versions re-scored
    from their saved scores (`rescore_saved` with Flows) and this run's Naive Baseline: one row per version, `name`,
    `markets` (count of Markets -> figure, None where not measured) and `columns` (the same figures by their column
    label, `large_markets_label`, for every table that shows them). None when the report has no `large_markets`."""
    large = report.get("large_markets")
    if not large:
        return None
    this = {row["markets"]: row["model"] for row in large["rows"]}
    naive = {row["markets"]: row.get("naive") for row in large["rows"]}
    rows = [(version_name(report.get("model_form")), this)]
    rescored = report.get("rescored_earlier_versions") or {}
    order = [name for name in EARLIER_FINGERPRINTS if name in rescored] + [n for n in rescored if n not in EARLIER_FINGERPRINTS]
    for name, figures in ((name, rescored[name]) for name in order):  # newest first, as `earlier_versions` lists them
        measured = {row["markets"]: row["model"] for row in figures.get("large_markets") or []}
        rows.append((name, {n: measured.get(n) for n in this}))
    return [
        {"name": name, "markets": markets, "columns": {large_markets_label(n): v for n, v in markets.items()}}
        for name, markets in [*rows, (NAIVE_THIS_RUN, naive)]
    ]


def large_markets_label(markets: int) -> str:
    """The column label of the measure on the `markets` largest importing Markets."""
    return f"{markets} largest Markets"


def _earlier_versions_text(block: dict) -> str:
    """The `earlier_versions` sentence, worded from the block's rows: on both measures where it holds them."""
    this, *others = block["rows"]
    if "uk_ranking_tie_proof" not in this:
        return _saved_earlier_versions_text(block)
    naive = {row["name"]: row for row in block["naive"]}
    verdict = "beats the Naive Baseline" if block["beats_naive"] else "does not beat the Naive Baseline"
    reported = naive.get(NAIVE_AS_REPORTED_FOR_V2)
    this_run = naive[NAIVE_THIS_RUN]

    def figure(value: float | None) -> str:
        return "not yet re-scored" if value is None else f"{value:.3f}"

    old = ", ".join(f"{row['name']} {figure(row['uk_ranking'])}" for row in others)
    tie_proof = ", ".join(f"{row['name']} {figure(row['uk_ranking_tie_proof'])}" for row in others)
    return (
        f"On {block['test_year']}, read once, the UK ranking of this version is {figure(this['uk_ranking_tie_proof'])} "
        f"on the tie-proof Selection Measure and {figure(this['uk_ranking'])} on the old measure (where tied "
        f"predictions share their average rank): it {verdict} ({figure(this_run['uk_ranking_tie_proof'])} tie-proof "
        f"and {figure(this_run['uk_ranking'])} old, from every Exporter's Flows"
        + (f"; {reported['uk_ranking']:.3f} old, as reported for v2" if reported is not None else "")
        + ")."
        + (f" Earlier versions on the old measure: {old}; on the tie-proof measure, from their saved scores: "
           f"{tie_proof}." if others else "")
        + " These are associations on one test year, not a forecast."
    )


def _saved_earlier_versions_text(block: dict) -> str:
    """The sentence of a block saved before the two measures were told apart (one figure per row)."""
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
    names and the saved report is not changed. v1's row takes the Lift its saved comparison holds (`with_v1_lift`).
    `comparison.relabelled` also rewords the v1 comparison."""
    saved = report.get("headline")
    if saved and saved.get("model") is not None and report.get("test_year") is not None:
        worded = headline({f"model_{HEADLINE_METRIC}": saved["model"], f"naive_{HEADLINE_METRIC}": saved["naive"]},
                          report["test_year"])
        report = {**report, "headline": {**saved, "text": worded["text"]}}
    report = _with_served_uk_pair_coverage(_with_nested_size_matched_lift(report))
    block = report.get("earlier_versions")
    if not block:
        return report
    this, *others = block["rows"]
    if this["name"].startswith(_SAVED_THIS_VERSION):
        this = {**this, "name": version_name(report.get("model_form"))}
    naive = [{**row, "name": _SAVED_NAMES.get(row["name"], row["name"])} for row in block["naive"]]
    block = {**block, "rows": [this, *others], "naive": naive}
    block = with_v1_lift(block, report.get("v1_comparison"))
    return {**report, "earlier_versions": {**block, "text": _earlier_versions_text(block)}}


def _with_nested_size_matched_lift(report: dict) -> dict:
    """`market_reach_effect` with Lift's size-matched pair nested (`size_matched`: with, without), as reports saved
    before it was (version 20261120T221726Z: `size_matched_with`, `size_matched_without`) are read."""
    effect = report.get("market_reach_effect")
    lift = (effect or {}).get("lift") or {}
    if "size_matched_with" not in lift:
        return report
    nested = {"with": lift["with"], "without": lift["without"],
              "size_matched": {"with": lift["size_matched_with"], "without": lift["size_matched_without"]}}
    return {**report, "market_reach_effect": {**effect, "lift": nested}}


def _with_served_uk_pair_coverage(report: dict) -> dict:
    """`market_reach_effect` with the Potential Range's UK-pair coverage the report serves, worded again. `rereach`
    rewrites the Potential Range's figure for the corrected Market Reach but not the block saved with the test-year
    read, so a card would state both (#162). The test-year figures stay as saved."""
    effect = report.get("market_reach_effect")
    served = (report.get("potential_range") or {}).get("uk_pair_coverage")
    if not effect or "uk_pair_coverage" not in effect or not served:
        return report
    coverage = {"with": served["with_market_reach"], "without": served["without_market_reach"]}
    if coverage["without"] is None:
        coverage["without"] = effect["uk_pair_coverage"]["without"]
    block = {key: value for key, value in effect.items() if key != "text"} | {"uk_pair_coverage": coverage}
    return {**report, "market_reach_effect": {**block, "text": market_reach_effect_text(block, report["test_year"])}}


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
    reach_source: pl.DataFrame | None = None,
) -> dict:
    """The evaluation report for the test year.

    `test_rows` are the test year's training rows (every non-zero Flow plus sampled zeros),
    `test_grid` its full UK grid, both with features untransformed. `baseline` holds the
    Baseline year's UK scores (`gap_period.baseline_year_scores`) and `baseline_models` /
    `baseline_transform` the models that scored them, used for the stability check. `settings` (validation year, params, sampling rates, ...)
    is recorded in the report. `flows` are the test year's Flows the naive baseline is computed
    from (default: the non-zero Flows among `test_rows`); `form` is the model form the test-year
    models are scored in (a blend takes that naive baseline). A form with Market Reach is fitted
    again for the test year (ADR 0010): its table from `reach_source`, the UK grid of the year
    before the test year, and its rescale constant on `test_grid`, for the test-year models and,
    for the stability check, for the Baseline models.
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
    previous_form = form
    if form.market_reach is not None:
        if reach_source is None:
            raise ValueError("A Model Form with Market Reach needs the UK grid of the year before the test year")
        beta = form.market_reach.beta
        fitted = {
            name: market_reach.fit(
                form, models, transform.apply(test_grid), reach_source, scored_year=year, uk=uk, beta=beta
            )
            for name, models, transform in (
                ("test", test_models.models, test_models.transform),
                ("previous", baseline_models, baseline_transform),
            )
        }
        form, previous_form = fitted["test"], fitted["previous"]
    scored_rows = score_rows(test_rows, test_models.models, test_models.transform, uk, form=form)
    scored_grid = score_rows(test_grid, test_models.models, test_models.transform, uk, form=form)
    without = {}
    if form.market_reach is not None:  # the kept change's effect on the ranking (`market_reach_effect`)
        plain = score_rows(test_grid, test_models.models, test_models.transform, uk, form=form.without_market_reach())
        without["without_market_reach"] = _ranking_figures(drop_excluded(plain), flows, year)
    accuracy_by_scope = {
        "all_exporters": accuracy(scored_rows, markets),
        "uk_out_of_fold": accuracy(scored_grid, markets),
    }

    previous = _uk_table(
        score_out_of_fold(baseline_transform.apply(test_grid), list(baseline_models), uk=uk, form=previous_form)
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
        # The same on the old measure, for continuity with v1, v2 and the Blend (`earlier_versions`); nothing is chosen on it.
        "old_measure": old_measure(scored_grid),
        "large_markets": large_markets_report(drop_excluded(scored_grid), flows, year, predictors=_PREDICTORS),
        "tariff_split_text": tariff_split_text(accuracy_by_scope, year),
        "stability": stable,
        **without,
    }


def _ranking_figures(scored_grid: pl.DataFrame, flows: pl.DataFrame, year: int) -> dict:
    """The model's headline (tie-proof), old-measure and large-Market figures on a scored UK grid of the test year."""
    column = _PREDICTORS["model"]
    large = large_markets_report(scored_grid, flows, year, predictors={"model": column, "naive": _PREDICTORS["naive"]})
    return {
        "headline_model": selection_measure(scored_grid, column, market=_ROW_MARKET),
        "old_measure_model": legacy_selection_measure(scored_grid, column),
        "large_markets": [{"markets": row["markets"], "model": row["model"]} for row in large["rows"]],
    }


# Why the Scenario Effect comparisons measure the model without Market Reach (ADR 0010).
SCENARIO_EFFECTS_UNCHANGED = (
    "Market Reach leaves Scenario Effects unchanged: it scales a pair's Export Potential with and without a Scenario "
    "alike, so the Event Backtests and the PPML benchmark compare the model's Scenario Effects without it (ADR 0010)."
)


def market_reach_effect(report: dict) -> dict | None:
    """What Market Reach, the change this version keeps (ADR 0010), does to the test-year figures: the served form
    beside the same models without it, on the headline (tie-proof and old measure), the large-Market measures, Lift
    (`lift` against `lift_without_market_reach`) and the Potential Range's coverage of UK pairs; Scenario Effects are
    unchanged by it (`SCENARIO_EFFECTS_UNCHANGED`). None for a version without Market Reach."""
    without = report.get("without_market_reach")
    if without is None:
        return None
    large = {row["markets"]: row["model"] for row in (report.get("large_markets") or {}).get("rows", [])}
    block: dict = {
        "selection_measure": {"with": report["headline"]["model"], "without": without["headline_model"]},
        "old_measure": {"with": (report.get("old_measure") or {}).get("model"), "without": without["old_measure_model"]},
        "large_markets": [
            {"markets": row["markets"], "with": large.get(row["markets"]), "without": row["model"]}
            for row in without["large_markets"]
        ],
    }
    lift, plain = report.get("lift"), report.get("lift_without_market_reach")
    if lift and plain:
        block["lift"] = {
            "with": lift["lift"], "without": plain["lift"],
            "size_matched": {"with": lift["size_matched"]["lift"], "without": plain["size_matched"]["lift"]},
        }
    coverage = (report.get("potential_range") or {}).get("uk_pair_coverage")
    if coverage:
        block["uk_pair_coverage"] = {"with": coverage["with_market_reach"], "without": coverage["without_market_reach"]}
    block["scenario_effects"] = SCENARIO_EFFECTS_UNCHANGED
    return {**block, "text": market_reach_effect_text(block, report["test_year"])}


def market_reach_effect_text(block: dict, test_year: int) -> str:
    """The `market_reach_effect` sentences, worded from its figures."""

    def pair(figures: dict, fmt: str = ".3f") -> str:
        return f"{number_text(figures['with'], fmt)} with it against {number_text(figures['without'], fmt)} without"

    large = "; ".join(f"on the {row['markets']} largest Markets {pair(row)}" for row in block["large_markets"])
    parts = [
        f"Market Reach is the change this version keeps (ADR 0010). On {test_year}, the UK ranking is "
        f"{pair(block['selection_measure'])} on the tie-proof Selection Measure ({pair(block['old_measure'])} on the "
        f"old measure)" + (f"; {large}" if large else "") + ".",
    ]
    if "lift" in block:
        lift = block["lift"]
        parts.append(
            f"Lift is {pair(lift, '.2f')} (size-matched {number_text(lift['size_matched']['with'], '.2f')} against "
            f"{number_text(lift['size_matched']['without'], '.2f')}), each held-out Exporter scaled by its own Market Reach."
        )
    if "uk_pair_coverage" in block:
        parts.append(
            f"The Potential Range holds {pair(block['uk_pair_coverage'], '.0%')} of UK pairs' Gap Period Actual Exports."
        )
    parts.append(block["scenario_effects"])
    parts.append("These are associations on one test year, not a forecast.")
    return " ".join(parts)


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
    """Where real test-year models are cached under the data root, keyed by their training settings and feature set."""
    from exportspredictor.features import features_dir

    settings = {**report_settings(metadata), "model_params": metadata["model_params"], "test_year": metadata["baseline_year"] - 1}
    if metadata.get("feature_set"):  # trained on another feature set's files; the original set keeps its keys
        settings["feature_set"] = metadata["feature_set"]
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
        reach_source=pl.read_parquet(folder / f"uk_grid_{year - market_reach.LAG_YEARS}.parquet"),
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
    folder = store.copy_version(source, artefacts_root, metadata=metadata, evaluation=report)
    if test_year_ledger is not None:  # the figures are saved: the read is completed (ADR 0009)
        ledger.complete_read(test_year_ledger, ledger.fingerprint(metadata))
    return folder


if __name__ == "__main__":
    import sys

    command = ledger.Ledger.for_command("python -m exportspredictor.evaluation", sys.argv[1:])
    print(add_to_latest(test_year_ledger=command), flush=True)
