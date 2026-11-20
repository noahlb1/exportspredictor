"""Governance for the write-up: the model card, data lineage and drift checks.

- `model_card` builds the card from an artefact (its metadata and evaluation report), so the
  numbers on it are the numbers that were evaluated, and `write_model_card` saves it as
  `model_card.md` next to the models it describes (course: generate cards from code and
  version them with the model; print UNASSIGNED rather than omit a section).
- `data_lineage` fingerprints every raw download with its size and SHA-256 (course: data
  lineage and reproducible artefacts).
- `psi` / `feature_drift` compare a feature's distribution in two years with the Population
  Stability Index, rated with the course's thresholds (below 0.10 stable, 0.10–0.20 watch,
  above 0.20 act).
"""

import hashlib
import math
from pathlib import Path

import polars as pl

from exportspredictor import comparison
from exportspredictor.artefacts import Artefacts
from exportspredictor.conformal import COVERAGE_TOLERANCE, coverage_text
from exportspredictor.evaluation import (
    LIMITATIONS,
    parts_not_run,
    range_limitations,
    result_limitations,
    size_coverage_limitations,
)
from exportspredictor.illustrative import scenario_note
from exportspredictor.model_choice import (
    model_form_summary,
    ranking_summary,
    round_count,
    selection_rule_text,
    traded_away,
)
from exportspredictor.paths import data_dir
from exportspredictor.wording import number_text, rate_text, years_text

OWNER = "UNASSIGNED"
MODEL_CARD_FILE = "model_card.md"

INTENDED_USE = [
    ("Help government export Advisors decide where to look: rank UK Product–Market pairs by Export Gap "
    "(Opportunities) or by the Scenario Effect of adding a Trade Agreement (Blocked Opportunities)."),
    ("Explore how the model associates Trade Agreement and Trading Bloc Scenarios with UK Export Potential, "
    "as an order of magnitude and a starting point for policy analysis."),
    "Explain, for one pair, which fundamentals raised or lowered its Export Potential (the Why? page).",
    "Every output is a lead for a person to check against sector knowledge, not a decision.",
]

OUT_OF_SCOPE = [
    ("Forecasting next year's UK exports, or any one pair's future exports: Lift is a tendency across many "
    "pairs, not a forecast."),
    ("Quoting a Scenario Effect as what a Trade Agreement would cause: Scenario Effects are associations, and the "
    "Event Backtests found them several times larger than the changes observed."),
    ("Decisions about individual firms or people (grants, credit, eligibility). The model holds no data on firms "
    "or people, and such uses would change its risk class (EU AI Act, intended purpose)."),
    ("Services trade, sanctions, quotas and product standards: none of them is a model input. The Applied Tariff "
    "is, but it is carried forward from 2021, so tariff changes after 2021 are invisible to the model."),
    "Markets under sanctions or embargo: their Export Gaps reflect fundamentals the sanctions override.",
]

# Decisions a reader of the model needs, in substance. The full log, with the options
# considered, is in the ADRs and in the project's decision log.
DECISIONS = [
    ("UK pairs are scored out-of-fold (ADR 0001)",
     ("UK Product–Market pairs are split into five folds by a hash of the pair, all years together. Each pair's "
     "Export Potential, Scenario Effect and explanation come from the fold model that never saw it, so the "
     "Export Gap is not shrunk by memorisation. UK supply features leave the target Market out.")),
    ("One monotone-constrained XGBoost model, blended with the Naive Baseline (ADR 0002, ADR 0008)",
     ("A Tweedie-loss gradient-boosted model serves Opportunities and Scenarios. Every Trade Agreement and "
     "Trading Bloc feature has a +1 monotone constraint and the Applied Tariff a −1 constraint, so adding an "
     "agreement or cutting a tariff never lowers Export Potential; it does not make the size of the effect right. "
     "Since ADR 0008 the form is chosen on the UK ranking alone: the model summary gives the served form and its "
     "weights. A blend with the Naive Baseline ranked UK Markets better than the model alone on all five folds of "
     "2022. The Naive Baseline has no policy features, so Scenario Effects come from the model's share only.")),
    ("BACI's HS 2012 revision (ADR 0003)",
     ("A longer history (2012–2024) gives the model more agreements that changed during training, and covers the "
     "EU–Japan Event Backtest; the cost is 2012-vintage Product codes.")),
    ("Excluded Products, Flagged Products and Re-export Hubs (ADR 0005)",
     ("Products whose Flows follow pipelines, bullion markets or licences (Sector 27 except 2712; silver, gold, "
     "platinum, precious-metal scrap and coins; antiques; radioactive elements) are dropped from training, scoring, "
     "rankings and metrics, though every total still counts them. The rest of Sector 71 is a Flagged Product, and "
     "eight entrepôt Markets are Re-export Hubs: both are ranked with a caveat.")),
    ("Applied Tariffs carried forward from 2021 and zeroed under Trade Agreements (ADR 0006)",
     ("UNCTAD TRAINS ends in 2021, so 2021 rates are carried forward to 2022–2024 and marked as imputed. A pair "
     "whose Trade Agreement entered force after 2021 gets a tariff of 0. Adding an agreement in a Scenario also "
     "sets the tariff to 0, and removing one restores the 2021 MFN rate. Tariffs beyond the training range are "
     "clipped, so the Scenario Effect of a very large hike flattens.")),
    ("Potential Range calibrated on other Exporters, never the UK (ADR 0007)",
     ("An 80% split-conformal range, calibrated within 20 bands of Export Potential (the top band split at "
     "$10m, $100m and $1bn) on other top-100 Exporters' 2024 rows. UK residuals are the Export Gaps the model exists to find, so calibrating on them would make every "
     "gap look normal. An Opportunity whose Actual Exports lie inside its range is an Uncertain Opportunity.")),
    ("Features kept by an ablation on 2022",
     ("Relatedness, Market import growth, the re-export ratio and the Applied Tariff were each removed in turn from "
     "a model on a Sector subset and judged on 2022; a feature was dropped when the model without it had both a lower deviance and a better UK ranking. "
     "A learning curve over the sampling rates, also on 2022, set the rates.")),
    ("Hyperparameters chosen by a Tree-structured Parzen Estimator search on 2022",
     ("Optuna's TPE with a median pruner replaced v1's five hand-picked configurations, with v1's configuration as "
     "trial 0, judged on 2022 Tweedie deviance and guarded by the UK ranking. The fold models reuse the winner, "
     "with its round count scaled by the ratio of training rows (a heuristic).")),
    ("The Export Gap is a Gap Period mean (2022–2024)",
     ("Each year's Export Potential minus Actual Exports, averaged over three years, steadies the ranking against "
     "one-off shipments and leaves out the COVID and Brexit-transition years. Scenarios and Blocked Opportunity "
     "values start from the 2024 Baseline.")),
    ("No features from the pair's trade in other Products",
     ("Total bilateral trade would absorb the policy effect, so a Scenario that flips only the policy columns would "
     "understate it.")),
    ("Policy features from the WTO RTA-IS",
     ("An agreement counts in a year if in force before 1 July. Partial-scope agreements are excluded. The UK leaves "
     "the EU Trading Bloc in 2020 (31 January) but stays in the Customs Union and EU agreements for 2020.")),
    ("Bloc Scenarios change only member Markets' rows",
     ("EU re-join sets the Trade Agreement, Customs Union and EU flags on EU Markets' rows; CPTPP sets the Trade "
     "Agreement and CPTPP flags on members' rows; each added agreement zeroes the tariff. A Blocked Opportunity "
     "gains at least 20% and $1M.")),
    ("Headline metric: rank correlation within each Product",
     ("Advisors read the model as a ranking of Markets for one Product, so the headline is the within-Product "
     "Spearman correlation on the UK's full 2023 grid, out-of-fold, against a naive market-share baseline.")),
    ("Lift and Event Backtests",
     ("Lift ranks held-out Exporters' pairs by their 2015–2017 Export Gap and counts a pair as grown if its 2023 "
     "Flow is at least 50% and $100,000 above its 2015–2017 mean, with a size-matched Lift beside it. Event "
     "Backtests compare predicted with observed difference-in-differences against the largest high-income Control "
     "Markets with no agreement, with the tariff zeroed when the agreement is on.")),
    ("A pre-registered comparison with v1",
     ("v2 is said to beat v1 only if, on the same 2023 UK rows, its within-Product rank correlation is higher with a "
     "paired-bootstrap interval above 0, it beats the Naive Baseline by more than v1 did, and no secondary metric is "
     "worse beyond its interval. The rule was fixed before 2023 was read; a gain ladder splits the change.")),
    ("A PPML gravity benchmark for the Trade Agreement Scenario Effect",
     ("A 3-way fixed-effects Poisson pseudo-maximum-likelihood gravity model, at Sector level on the same Flows and "
     "scope, gives an independent estimate of what a Trade Agreement is associated with. It is fitted once and "
     "saved with the artefact (`python -m exportspredictor.ppml`), beside the model's own association.")),
    ("Values in nominal US dollars",
     ("Every value is in nominal USD, BACI's unit, so figures match international trade sources and need no "
     "conversion. The Bank of England's annual averages (XUAAUSS) only convert HMRC's £ for the cross-check.")),
    ("Missing GDP and population stay missing after a stated chain of sources",
     ("WDI, then IMF WEO, then CEPII Gravity, then interpolation; 21 small territories (0.1% of value) stay "
     "missing and XGBoost routes them by learned defaults.")),
]

DATA_SOURCES = [
    ("BACI HS12 V202601 (CEPII)", ("Every non-zero Flow, 2012–2024, reconciled from exporter and importer reports; "
     "the model's target and supply and demand features."), "raw/baci"),
    ("CEPII Gravity V202211", ("Distance, contiguity, common language, colonial ties; cross-check of agreements "
     "to 2021."), "raw/gravity"),
    ("World Bank WDI and country list", "GDP and population; regions and income levels.", "raw/wdi"),
    ("IMF World Economic Outlook", "GDP and population where WDI has none (e.g. Taiwan).", "raw/imf_weo"),
    ("WTO Regional Trade Agreements database (RTA-IS)", ("Trade Agreements, Customs Unions and Trading Bloc "
     "memberships with per-member dates: the policy features."), "raw/wto"),
    ("Bank of England, series XUAAUSS", "USD → GBP annual average rates, used only to convert HMRC's £ for the cross-check.", "src/exportspredictor/currency.py"),
    ("HMRC Overseas Trade Statistics, 2023 (Open Government Licence v3.0)", ("Cross-check of BACI's UK totals "
     "only; never a model input."), "raw/hmrc"),
    ("UN Statistics Division HS 2022 → HS 2012 conversion table", ("Puts HMRC's 2023 codes on BACI's revision "
     "for the cross-check."), "raw/hmrc"),
    ("UNCTAD TRAINS via the WITS API (reported rates, 2012–2021)", ("HS6 MFN and preferential rates: the "
     "Applied Tariff, carried forward from 2021 to 2022–2024."), "raw/trains"),
    ("UN Statistics Division HS 2017 and HS 2007 → HS 2012 correlation tables", ("Put TRAINS reporter-years "
     "on BACI's revision."), "raw/unsd"),
]

# Raw folders under the data root and the source each belongs to, for `data_lineage`.
_RAW_SOURCES = {
    "baci": "BACI HS12 V202601 (CEPII)",
    "gravity": "CEPII Gravity V202211",
    "wdi": "World Bank WDI",
    "worldbank": "World Bank country list",
    "imf_weo": "IMF World Economic Outlook",
    "wto": "WTO RTA-IS",
    "hmrc": "HMRC OTS 2023 and UN HS concordance",
    "trains": "UNCTAD TRAINS",
    "unsd": "UN HS concordances",
}


# --- Model card ----------------------------------------------------------------------------


def model_card(artefacts: Artefacts) -> dict:
    """The model card of one artefact version, built from its metadata, evaluation report and
    saved PPML benchmark (`Artefacts.ppml_benchmark`; listed under the parts not run where absent).
    """
    # Read in today's words: the saved report's names and the v1 comparison's, which predate the Blend.
    metadata, report = artefacts.metadata, comparison.relabelled(artefacts.evaluation or {})
    return {
        "version": artefacts.version,
        "owner": OWNER,
        "model": _model_summary(metadata),
        "model_form": model_form_summary(metadata),
        "round_count": round_count(metadata),
        "illustrative": scenario_note(artefacts),
        "intended_use": INTENDED_USE,
        "out_of_scope": OUT_OF_SCOPE,
        "training_scope": _training_scope(metadata),
        "features": len(metadata.get("features") or []),
        "dropped_features": _dropped_features(metadata),
        "excluded_products": _excluded_products(artefacts.excluded_products),
        "metrics": _metrics(report),
        "not_run": parts_not_run(report, artefacts.ppml_benchmark) if report else [],
        "v1_comparison": _comparison_summary(report.get("v1_comparison"), comparison.version_name(report)),
        "earlier_versions": report.get("earlier_versions"),
        "traded_away": traded_away(report.get("earlier_versions")),
        "ranking_experiments": ranking_summary(metadata.get("ranking_experiments")),
        "potential_range": _potential_range(report.get("potential_range")),
        "range_coverage_by_exporter": _range_review(report.get("potential_range")),
        "limitations": [
            *result_limitations(metadata, report), *_benchmark_limitations(artefacts.ppml_benchmark),
            *range_limitations(artefacts.potential_range_bands),
            *size_coverage_limitations(report.get("potential_range")), *LIMITATIONS,
        ],
        "data_sources": [{"name": n, "use": u, "location": loc} for n, u, loc in DATA_SOURCES],
        "source_versions": metadata.get("sources", []),  # `sources.source_versions`, if recorded
        "decisions": [{"title": t, "text": x} for t, x in [_sampling_decision(metadata), *DECISIONS]],
    }


def _model_summary(metadata: dict) -> str:
    params = metadata.get("params", {})
    return (
        f"XGBoost, Tweedie objective (variance power {number_text(params.get('tweedie_variance_power'), '.2f')}), depth "
        f"{params.get('max_depth', '?')}, {number_text(metadata.get('rounds'), ',')} rounds; {metadata.get('n_folds', '?')} "
        f"out-of-fold models trained on {years_text(metadata.get('training_years'))}, scoring the "
        f"{metadata.get('baseline_year', '?')} Baseline. Source: {metadata.get('source', '?')}."
    )


def _headline_lines(card: dict) -> list[str]:
    """The test-year UK ranking beside the Naive Baseline and earlier versions, else the report's headline."""
    if card.get("earlier_versions"):
        return _earlier_lines(card["earlier_versions"])
    headline = next((m for m in card["metrics"] if m["name"] == "UK ranking (headline)"), None)
    return [headline["text"]] if headline else ["The test year has not been evaluated for this version."]


def _illustrative_lines(note: dict) -> list[str]:
    """The Scenario note (`illustrative.scenario_note`): the label, then each benchmark part, run or not."""
    parts = [note["benchmark"], note["model_association"], note["event_backtest"], note["blend"]]
    return [f"**{note['label']}.** {note['text']}", "", *[f"- {part['text']}" for part in parts if part]]


def _ranking_lines(summary: dict | None) -> list[str]:
    if summary is None:
        return ["No experiment loop is recorded for this version."]
    lines = [
        selection_rule_text(summary),
        "",
        (f"Each row is one decision of the rule, labelled with its experiment's number in "
         f"docs/research/opportunity-ranking-experiments.md: {len(summary['experiments'])} decisions for "
         f"{summary['n_experiments']} experiments."),
        "",
        "| Experiment | Best Arm | Selection Measure | Change on the best so far | Kept |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {e['label']} | {e['best_arm'] or 'n/a'} | {number_text(e['selection_measure'], '.4f')} | "
            f"{number_text(e['change'], '+.4f')} | {e['kept'] or 'no'} |"
            for e in summary["experiments"]
        ],
    ]
    confirmation = summary["confirmation"]
    if confirmation:
        lines += ["", (
            f"Confirmed on all {confirmation['n_folds']} folds of {summary['validation_year']}: the chosen form "
            f"{number_text(confirmation['winner'], '.4f')}, v2's form (the model alone) {number_text(confirmation['v2'], '.4f')}, "
            f"the Naive Baseline {number_text(confirmation['naive'], '.4f')}; difference "
            f"{number_text(confirmation['difference'], '+.4f')} against a threshold of "
            f"{number_text(confirmation['win_threshold'], '.4f')} ("
            f"{'confirmed' if confirmation['beats_v2'] else 'not confirmed'})."
        )]
    return lines


def _earlier_lines(block: dict | None) -> list[str]:
    if block is None:
        return ["The test year has not been set beside earlier versions for this version."]
    def cell(row, key, fmt):
        value = row.get(key)
        return "n/a" if value is None else number_text(value, fmt)
    if "uk_ranking_tie_proof" not in block["rows"][0]:  # saved before the two measures were told apart
        return [
            f"**{block['text']}**",
            "",
            "| Version | UK ranking | Size-matched Lift | Lift | Tweedie deviance, all Exporters | Tweedie deviance, UK pairs |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                f"| {row['name']} | {cell(row, 'uk_ranking', '.3f')} | {cell(row, 'size_matched_lift', '.2f')} | "
                f"{cell(row, 'lift', '.2f')} | {cell(row, 'tweedie_deviance_all_exporters', ',.1f')} | "
                f"{cell(row, 'tweedie_deviance_uk', ',.1f')} |"
                for row in block["rows"]
            ],
            *[f"| {row['name']} | {number_text(row['uk_ranking'], '.3f')} | | | | |" for row in block["naive"]],
        ]
    return [
        f"**{block['text']}**",
        "",
        "| Version | UK ranking, old measure | UK ranking, tie-proof | Size-matched Lift | Lift | "
        "Tweedie deviance, all Exporters | Tweedie deviance, UK pairs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            f"| {row['name']} | {cell(row, 'uk_ranking', '.3f')} | {cell(row, 'uk_ranking_tie_proof', '.3f')} | "
            f"{cell(row, 'size_matched_lift', '.2f')} | {cell(row, 'lift', '.2f')} | "
            f"{cell(row, 'tweedie_deviance_all_exporters', ',.1f')} | {cell(row, 'tweedie_deviance_uk', ',.1f')} |"
            for row in block["rows"]
        ],
        *[f"| {row['name']} | {cell(row, 'uk_ranking', '.3f')} | {cell(row, 'uk_ranking_tie_proof', '.3f')} | | | | |"
          for row in block["naive"]],
    ]


def _training_scope(metadata: dict) -> dict:
    """Who the models train on and at what sampling rates, as recorded in the metadata (ADR 0004)."""
    scope = metadata.get("exporter_scope") or []
    non_uk = metadata.get("non_uk_sampling_rate")
    zeros = metadata.get("zero_sampling_rate")
    uk_zeros = metadata.get("uk_zero_sampling_rate", 1.0)
    text = (
        f"The models train on Flows and Zero Cells of {len(scope)} Exporters (the top economies by mean GDP, the "
        f"UK {'included' if 'GBR' in scope else 'NOT included'}) into every Market. Every UK row is kept, with "
        f"{number_text(uk_zeros, '.0%')} of UK Zero Cells. Other Exporters' Zero Cells are kept at {rate_text(zeros)}, and their "
        f"rows are then kept at {rate_text(non_uk)}; each sampled row is weighted by the inverse of its rate "
        f"(Horvitz–Thompson), so weighted totals stay unbiased."
    )
    return {
        "exporters": len(scope),
        "uk_in_scope": "GBR" in scope,
        "non_uk_sampling_rate": non_uk,
        "non_uk_sampling_rate_derived": metadata.get("non_uk_sampling_rate_derived"),
        "zero_sampling_rate": zeros,
        "uk_zero_sampling_rate": uk_zeros,
        "text": text,
    }


def _dropped_features(metadata: dict) -> list[str]:
    """Features the 2022 ablation dropped: removed one at a time, and absent from the kept list."""
    experiments = metadata.get("experiments") or {}
    kept = set(experiments.get("kept_features") or metadata.get("features") or [])
    return [row["dropped"] for row in experiments.get("ablation", []) if row["dropped"] and row["dropped"] not in kept]


def _feature_name(feature: str) -> str:
    return feature.replace("_", " ").capitalize()


def _excluded_products(table: pl.DataFrame) -> list[dict]:
    """The Excluded Products of the artefact, one group per reason (ADR 0005)."""
    groups = table.sort("hs6").group_by("reason", maintain_order=True).agg(pl.col("hs6"))
    return [{"reason": row["reason"], "products": row["hs6"]} for row in groups.iter_rows(named=True)]


# The row budget spec #39 planned around; the measured one (`scripts/gpu_check.py`) is in the metadata.
SPEC_ROW_BUDGET_M = 110


def _sampling_decision(metadata: dict) -> tuple[str, str]:
    scope = _training_scope(metadata)
    budget = metadata.get("row_budget")
    measured = (
        f" The budget measured on the RTX 3080 (`scripts/gpu_check.py`) is {budget:,} rows, about "
        f"{1 - budget / (SPEC_ROW_BUDGET_M * 1e6):.0%} below the about {SPEC_ROW_BUDGET_M} M rows the spec planned "
        "around, so the non-UK rate is lower than planned."
        if budget
        else ""
    )
    return (
        f"Train on {scope['exporters']} Exporters into every Market, keeping every UK row (ADR 0004)",
        (f"The full HS6 world does not fit the GPU's 10 GB. {scope['text']} The non-UK rate is the largest the "
         f"VRAM row budget allows, because the learning curve was still falling at the highest rate tried.{measured}"),
    )


def _metrics(report: dict) -> list[dict]:
    rows = []
    headline = report.get("headline")
    if headline:
        rows.append({"name": "UK ranking (headline)", "model": headline["model"], "naive": headline["naive"],
                     "text": headline["text"]})
    for scope, label in (("all_exporters", "all Exporters"), ("uk_out_of_fold", "UK pairs, out-of-fold")):
        overall = report.get("accuracy", {}).get(scope, {}).get("overall")
        if overall:
            rows.append({"name": f"Tweedie deviance, {label}", "model": overall["model_tweedie_deviance"],
                         "naive": overall["naive_tweedie_deviance"], "text": "lower is better"})
    lift = report.get("lift")
    if lift:
        rows.append({"name": "Lift", "model": lift["lift"], "naive": 1.0, "text": f"95% interval {_ci(lift['ci'])}"})
        rows.append({"name": "Lift against pairs of similar size", "model": lift["size_matched"]["lift"],
                     "naive": 1.0, "text": f"95% interval {_ci(lift['size_matched']['ci'])}"})
    # The Event Backtests measure the model alone; a served blend scales Scenario Effects by 1 − w.
    alone = "; the model alone, before the blend" if (report.get("model_form") or {}).get("naive_weight") else ""
    for event in report.get("event_backtest", {}).get("events", []):
        for direction in event["directions"]:
            pooled = direction["pooled"]
            rows.append({"name": f"Event Backtest: {direction['label']}", "model": pooled["predicted"],
                         "naive": pooled["observed"], "percent": True,
                         "text": f"model's Scenario Effect against the observed change (95% interval "
                         f"{_ci(pooled['observed_ci'], percent=True)}){alone}"})
    uk = report.get("accuracy", {}).get("uk_out_of_fold", {})
    for group in uk.get("by_income", []):
        rows.append({"name": f"UK ranking, {group['group']} Markets", "model": group["model_spearman_within_product"],
                     "naive": group["naive_spearman_within_product"],
                     "text": f"{group['non_zero_flows']:,} UK Flows in {group['rows']:,} pairs"})
    return rows


def _comparison_summary(block: dict | None, name: str) -> dict | None:
    """The pre-registered v1 comparison in brief: the verdict, the primary metric and the gain ladder;
    `name` is what the words call the version compared with v1 (`comparison.version_name`)."""
    if not block:
        return None
    primary = block["primary"]
    ladder = block.get("ladder")
    return {
        "name": name,
        "verdict": block["verdict"]["text"],
        "v2_beats_v1": block["verdict"]["v2_beats_v1"],
        "primary": {key: primary[key] for key in ("label", "v1", "v2", "naive", "difference", "ci")},
        "rows": block["rows"],
        "ladder": [
            {"from": step["from"], "to": step["to"], "label": step["label"], "change": step["change"], "ci": step["ci"]}
            for step in ladder["steps"]
        ] if ladder else [],
        "ladder_note": ladder["note"] if ladder else None,
    }


def _ci(interval, *, percent: bool = False) -> str:
    low, high = interval or (None, None)
    spec = "+.0%" if percent else ".2f"
    return f"{number_text(low, spec)} to {number_text(high, spec)}"


# A per-Exporter coverage more than this far from the target is reported as too wide or too narrow.
RANGE_COVERAGE_TOLERANCE = COVERAGE_TOLERANCE


def range_coverage_by_exporter(potential_range: dict, tolerance: float = RANGE_COVERAGE_TOLERANCE) -> pl.DataFrame:
    """Held-out Potential Range coverage per calibration Exporter, lowest first, rated against the target.

    The bias review's check of ADR 0007's exchangeability assumption: an Exporter whose pairs fall
    inside their range far less often than the target is one whose variation the range understates
    ("too narrow"); far more often, one it overstates ("too wide").
    """
    target = potential_range.get("target", 0.8)
    difference = pl.col("coverage") - target
    return (
        pl.DataFrame(potential_range.get("by_exporter", []),
                     schema={"exporter": pl.String, "rows": pl.Int64, "coverage": pl.Float64})
        .with_columns(difference=difference)
        .with_columns(
            status=pl.when(difference > tolerance).then(pl.lit("too wide"))
            .when(difference < -tolerance).then(pl.lit("too narrow"))
            .otherwise(pl.lit("near target"))
        )
        .sort("coverage")
    )


def _potential_range(potential_range: dict | None) -> dict | None:
    """The share of Opportunities whose UK Actual Exports lie inside their Potential Range (Uncertain
    Opportunities), reported as a share and never per row (ADR 0007). None where not recorded."""
    if not potential_range or potential_range.get("uncertain_opportunity_share") is None:
        return None
    keys = ("coverage", "target", "opportunities", "uncertain_opportunities", "uncertain_opportunity_share")
    return {key: potential_range[key] for key in keys} | {
        "text": coverage_text(potential_range), "by_size": potential_range.get("by_size"),
    }


def _size_lines(by_size: list[dict] | None) -> list[str]:
    """The held-out coverage by size of Export Potential as a Markdown table; none for older versions."""
    if not by_size:
        return []
    return ["", "| Size of Export Potential | Held-out rows | Coverage |", "| --- | --- | --- |", *[
        f"| {row['size']} | {row['rows']:,} | {number_text(row['coverage'], '.0%')} |" for row in by_size
    ]]


def _range_review(potential_range: dict | None) -> dict | None:
    if not potential_range or not potential_range.get("by_exporter"):
        return None
    table = range_coverage_by_exporter(potential_range)
    lowest = table.row(0, named=True)
    by_excess = table.sort("difference", descending=True)
    return {
        "target": potential_range.get("target", 0.8),
        "coverage": potential_range.get("coverage"),
        "exporters": table.height,
        "near_target": table.filter(pl.col("status") == "near target").height,
        "lowest": {"exporter": lowest["exporter"], "coverage": lowest["coverage"]},
        "too_wide": by_excess.filter(pl.col("status") == "too wide")["exporter"].to_list(),
        "too_narrow": table.filter(pl.col("status") == "too narrow")["exporter"].to_list(),
    }


def _range_review_lines(review: dict | None) -> list[str]:
    if review is None:
        return ["The artefact holds no Potential Range coverage by Exporter."]
    tolerance = f"{RANGE_COVERAGE_TOLERANCE * 100:.0f} percentage points"
    lines = [(
        f"On a held-out half of the calibration rows, {number_text(review['coverage'], '.0%')} of Actual Exports fell inside "
        f"their Potential Range (target {review['target']:.0%}). Per Exporter, {review['near_target']} of "
        f"{review['exporters']} are within {tolerance} of the target; the lowest is "
        f"{review['lowest']['exporter']} at {review['lowest']['coverage']:.0%}."
    )]
    if review["too_narrow"]:
        lines += ["", f"Too narrow (the range understates their variation): {_some(review['too_narrow'])}."]
    if review["too_wide"]:
        lines += ["", f"Too wide (the range overstates their variation), largest excess first: {_some(review['too_wide'])}."]
    lines += ["", ("The UK is never a calibration Exporter (ADR 0007), so its own coverage cannot be measured: the "
                   "range assumes UK pairs vary as these Exporters' pairs do.")]
    return lines


def _some(names: list[str], limit: int = 10) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


def _benchmark_limitations(benchmark: dict | None) -> list[dict]:
    if not benchmark:
        return []
    ppml, model, published = benchmark["ppml"], benchmark["model"], benchmark["published_range"]
    low, high = published["low_pct"] / 100, published["high_pct"] / 100
    return [{
        "title": "Scenario Effects are far above a gravity benchmark",
        "text": f"Over the UK pairs with no Trade Agreement, the model associates adding one (with its tariff "
        f"removed) with {model['effect_pct']:+.1f}% Export Potential, Baseline-weighted. A 3-way fixed-effects PPML "
        f"gravity model on the same Flows associates a Trade Agreement with {ppml['effect_pct']:+.1f}% (95% interval "
        f"{_signed_percent_with_minus_sign(ppml['effect_low_pct'])} to {_signed_percent_with_minus_sign(ppml['effect_high_pct'])}). Published estimates span "
        f"about {low:+.0%} to {high:+.0%}, but the Event Backtests point the same way as the benchmark: the model overstates the "
        f"association. Read Scenario Effects as an upper end.",
    }]


def _signed_percent_with_minus_sign(value: float) -> str:
    return f"{value:+.1f}%".replace("-", "−")


def render_model_card(card: dict) -> str:
    """The model card as Markdown. It leads with what the notebook leads with (the headline, the Model
    Form and the Illustrative label), and ends with the frozen v1 comparison as a short reported result."""
    lines = [f"# Model card: UK Exports Predictor, version {card['version']}", "",
             f"**Owner:** {card['owner']}", ""]
    lines += ["## Headline", "", *_headline_lines(card), ""]
    if card.get("traded_away"):
        lines += [card["traded_away"]["text"], ""]
    lines += ["## Model Form", "", card["model_form"]["text"], "", f"**Model:** {card['model']}", "",
              card["round_count"]["text"], ""]
    lines += ["## Illustrative Scenario Effects", "", *_illustrative_lines(card["illustrative"]), ""]
    lines += ["## How the Model Form was chosen", "", *_ranking_lines(card.get("ranking_experiments")), ""]
    lines += ["## Potential Range", "", (card["potential_range"] or {}).get(
        "text", "The artefact records no Potential Range coverage for this version."),
        *_size_lines((card["potential_range"] or {}).get("by_size")), ""]
    lines += ["## Metrics", "", "| Metric | Model | Reference (Naive Baseline, chance, or observed) | Note |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {m['name']} | {_number(m['model'], m.get('percent', False))} | "
        f"{_number(m['naive'], m.get('percent', False))} | {m['text']} |"
        for m in card["metrics"]
    ]
    lines += [""]
    if card.get("not_run"):
        lines += ["## Parts not run", "", *[f"- {part['text']}" for part in card["not_run"]], ""]
    lines += ["## Limitations", "", *[f"- **{x['title']}.** {x['text']}" for x in card["limitations"]], ""]
    lines += ["## Intended use", "", *[f"- {item}" for item in card["intended_use"]], ""]
    lines += ["## Out-of-scope use", "", *[f"- {item}" for item in card["out_of_scope"]], ""]
    lines += ["## Training scope and sampling", "", card["training_scope"]["text"], ""]
    dropped = card["dropped_features"]
    lines += ["## Features", "", (
        f"The model uses {card['features']} features. The 2022 ablation dropped "
        f"{', '.join(_feature_name(f) for f in dropped)}: the model without it had both a lower 2022 deviance and a better UK ranking, so it is "
        "still built but not trained on." if dropped else
        f"The model uses {card['features']} features; the ablation dropped none of the candidates."), ""]
    lines += ["## Excluded Products", "", (
        "These Products are left out of training, scoring and every ranking and metric (ADR 0005). Country and "
        "world totals still count them."), "", *[
        f"- **{group['reason']}** {len(group['products'])} Product{'s' if len(group['products']) != 1 else ''}: {', '.join(group['products'])}."
        for group in card["excluded_products"]], ""]
    lines += ["## Bias review: Potential Range coverage by Exporter", "",
              *_range_review_lines(card["range_coverage_by_exporter"]), ""]
    lines += ["## Data sources", "", *[f"- **{s['name']}** ({s['location']}): {s['use']}" for s in card["data_sources"]], ""]
    if card["source_versions"]:
        lines += ["## Source versions", "", *[
            f"- **{s['source']}**: {s['version']}; downloaded {s['downloaded'] or 'not found'}"
            for s in card["source_versions"]
        ], ""]
    lines += ["## Decisions behind the model", "", *[f"- **{d['title']}.** {d['text']}" for d in card["decisions"]], ""]
    lines += ["## Frozen results", "", ("Reported once and no longer extended: they decided v2, not the served "
              "Model Form (ADR 0008)."), "", "### Comparison with v1", "", *_comparison_lines(card["v1_comparison"]), ""]
    return "\n".join(lines)


def _comparison_lines(summary: dict | None) -> list[str]:
    if summary is None:
        return ["The pre-registered comparison with v1 has not been run for this version."]
    primary, name = summary["primary"], summary["name"]
    lines = [
        f"**Verdict.** {summary['verdict']}",
        "",
        (
            f"Primary metric ({primary['label']}): v1 {number_text(primary['v1'], '.3f')}, {name} {number_text(primary['v2'], '.3f')}, "
            f"Naive Baseline {number_text(primary['naive'], '.3f')}; {name} − v1 {number_text(primary['difference'], '+.3f')} (95% interval "
            f"{number_text(primary['ci'][0], '+.3f')} to {number_text(primary['ci'][1], '+.3f')})."
        ),
    ]
    if summary["ladder"]:
        lines += ["", "Gain ladder (change in the primary metric at each cumulative step):", ""]
        lines += [
            f"- {step['from']} → {step['to']} ({step['label']}): {number_text(step['change'], '+.3f')} (95% interval "
            f"{number_text(step['ci'][0], '+.3f')} to {number_text(step['ci'][1], '+.3f')})"
            for step in summary["ladder"]
        ]
        lines += ["", summary["ladder_note"]]
    return lines


def _number(value, percent: bool = False) -> str:
    if percent:
        return number_text(value, "+.0%")
    return number_text(value, ",.0f" if value is not None and abs(value) >= 100 else ".2f")


def write_model_card(artefacts: Artefacts, folder: Path) -> Path:
    """Write `model_card.md` into an artefact folder, next to the models it describes."""
    path = Path(folder) / MODEL_CARD_FILE
    path.write_text(render_model_card(model_card(artefacts)), encoding="utf-8")
    return path


# --- Data lineage ---------------------------------------------------------------------------


def data_lineage(root: Path | None = None) -> pl.DataFrame:
    """Size and SHA-256 of every raw download under the data root: source, file, bytes, sha256."""
    root = Path(root or data_dir())
    rows = []
    for folder, source in _RAW_SOURCES.items():
        base = root / "raw" / folder
        if not base.is_dir():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and not p.name.endswith(".part")):
            rows.append({"source": source, "file": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return pl.DataFrame(rows, schema={"source": pl.String, "file": pl.String, "bytes": pl.Int64, "sha256": pl.String})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- Drift ---------------------------------------------------------------------------------

PSI_BINS = 10
PSI_FLOOR = 0.005  # bins floored at 0.5% so an empty bin does not explode the index
PSI_WATCH = 0.10
PSI_ACT = 0.20


def psi(reference: pl.Series, current: pl.Series, bins: int = PSI_BINS) -> float:
    """Population Stability Index of `current` against `reference`, binned on the reference's quantiles.

    Missing values form their own bin. PSI = Σ (actual − expected) × ln(actual / expected).
    """
    reference, current = reference.cast(pl.Float64), current.cast(pl.Float64)
    edges = sorted({reference.drop_nulls().quantile(q / bins, "linear") for q in range(1, bins)})
    expected, actual = _shares(reference, edges), _shares(current, edges)
    return float(sum((a - e) * math.log(a / e) for e, a in zip(expected, actual, strict=True)))


def _shares(values: pl.Series, edges: list[float]) -> list[float]:
    present = values.drop_nulls().sort()
    index = present.search_sorted(pl.Series(edges, dtype=pl.Float64), side="right") if edges else None
    counts, previous = [], 0
    for position in ([*index.to_list()] if index is not None else []):
        counts.append(position - previous)
        previous = position
    counts.append(present.len() - previous)
    counts.append(values.null_count())
    total = max(values.len(), 1)
    return [max(count / total, PSI_FLOOR) for count in counts]


def feature_drift(reference: pl.DataFrame, current: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """PSI of each feature between two tables, largest first, with the course's status: stable, watch or act."""
    rows = [{"feature": f, "psi": psi(reference[f], current[f])} for f in features]
    return (
        pl.DataFrame(rows, schema={"feature": pl.String, "psi": pl.Float64})
        .with_columns(
            status=pl.when(pl.col("psi") > PSI_ACT).then(pl.lit("act"))
            .when(pl.col("psi") >= PSI_WATCH).then(pl.lit("watch"))
            .otherwise(pl.lit("stable"))
        )
        .sort("psi", descending=True)
    )


if __name__ == "__main__":
    # Write the model card into the newest artefact folder (a document, not a table: no new version).
    from exportspredictor import artefacts as store
    from exportspredictor.paths import artefacts_dir

    folder = store.latest_folder(artefacts_dir())
    print(write_model_card(store.load(folder), folder))
