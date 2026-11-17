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

from exportspredictor.artefacts import Artefacts
from exportspredictor.evaluation import LIMITATIONS
from exportspredictor.paths import data_dir

OWNER = "UNASSIGNED"
MODEL_CARD_FILE = "model_card.md"

INTENDED_USE = [
    "Help government export Advisors decide where to look: rank UK Product–Market pairs by Export Gap "
    "(Opportunities) or by the Scenario Effect of adding a Trade Agreement (Blocked Opportunities).",
    "Explore how the model associates Trade Agreement and Trading Bloc Scenarios with UK Export Potential, "
    "as an order of magnitude and a starting point for policy analysis.",
    "Explain, for one pair, which fundamentals raised or lowered its Export Potential (the Why? page).",
    "Every output is a lead for a person to check against sector knowledge, not a decision.",
]

OUT_OF_SCOPE = [
    "Forecasting next year's UK exports, or any one pair's future exports: Lift is a tendency across many "
    "pairs, not a forecast.",
    "Quoting a Scenario Effect as what a Trade Agreement would cause: Scenario Effects are associations, and the "
    "Event Backtests found them several times larger than the changes observed.",
    "Decisions about individual firms or people (grants, credit, eligibility). The model holds no data on firms "
    "or people, and such uses would change its risk class (EU AI Act, intended purpose).",
    "Services trade, tariff changes, sanctions, quotas and product standards: none of them is a model input.",
    "Markets under sanctions or embargo: their Export Gaps reflect fundamentals the sanctions override.",
]

# Decisions a reader of the model needs, in substance. The full log, with the options
# considered, is in the ADRs and in the project's decision log.
DECISIONS = [
    ("UK pairs are scored out-of-fold (ADR 0001)",
     "UK Product–Market pairs are split into five folds by a hash of the pair, all years together. Each pair's "
     "Export Potential, Scenario Effect and explanation come from the fold model that never saw it, so the "
     "Export Gap is not shrunk by memorisation. UK supply features leave the target Market out."),
    ("One monotone-constrained XGBoost model (ADR 0002)",
     "A single Tweedie-loss gradient-boosted model serves Opportunities and Scenarios. Every policy feature has a "
     "+1 monotone constraint, so adding an agreement never lowers Export Potential; it does not make the size "
     "of the effect right."),
    ("BACI's HS 2012 revision (ADR 0003)",
     "A longer history (2012–2024) gives the model more agreements that changed during training, and covers the "
     "EU–Japan Event Backtest; the cost is 2012-vintage Product codes."),
    ("No features from the pair's trade in other Products",
     "Total bilateral trade would absorb the policy effect, so a Scenario that flips only the policy columns would "
     "understate it."),
    ("Training rows: non-zero Flows plus 2% of relevant zero cells",
     "Zero cells where both sides trade the Product, but not with each other, are sampled at 2% and weighted by 50 "
     "(Horvitz–Thompson), so totals stay unbiased."),
    ("Only 30% of other Exporters' rows are trained on",
     "All 152 M training rows filled the GPU's memory, so the real models keep a reweighted 30% hash sample of "
     "non-UK rows and every UK row. Estimates stay unbiased but noisier, most for small Exporters."),
    ("Policy features from the WTO RTA-IS",
     "An agreement counts in a year if in force before 1 July. Partial-scope agreements are excluded. The UK leaves "
     "the EU Trading Bloc in 2020 (31 January) but stays in the Customs Union and EU agreements for 2020."),
    ("Bloc Scenarios change only member Markets' rows",
     "EU re-join sets the Trade Agreement, Customs Union and EU flags on EU Markets' rows; CPTPP sets the Trade "
     "Agreement and CPTPP flags on members' rows. A Blocked Opportunity gains at least 20% and $1M."),
    ("Headline metric: rank correlation within each Product",
     "Advisors read the model as a ranking of Markets for one Product, so the headline is the within-Product "
     "Spearman correlation on the UK's full 2023 grid, out-of-fold, against a naive market-share baseline."),
    ("Lift and Event Backtests",
     "Lift counts a pair as grown if its Flow rose at least 50% and $100,000 from 2017 to 2023, for Exporters the "
     "model never saw, with a size-matched Lift beside it. Event Backtests compare predicted with observed "
     "difference-in-differences against the 10 largest high-income Control Markets with no agreement."),
    ("Values in £ at one stated rate",
     "USD values are converted at the Bank of England's annual average for the Baseline year (XUAAUSS)."),
    ("Missing GDP and population stay missing after a stated chain of sources",
     "WDI, then IMF WEO, then CEPII Gravity, then interpolation; 21 small territories (0.1% of value) stay "
     "missing and XGBoost routes them by learned defaults."),
]

DATA_SOURCES = [
    ("BACI HS12 V202601 (CEPII)", "Every non-zero Flow, 2012–2024, reconciled from exporter and importer reports; "
     "the model's target and supply and demand features.", "raw/baci"),
    ("CEPII Gravity V202211", "Distance, contiguity, common language, colonial ties; cross-check of agreements "
     "to 2021.", "raw/gravity"),
    ("World Bank WDI and country list", "GDP and population; regions and income levels.", "raw/wdi"),
    ("IMF World Economic Outlook", "GDP and population where WDI has none (e.g. Taiwan).", "raw/imf_weo"),
    ("WTO Regional Trade Agreements database (RTA-IS)", "Trade Agreements, Customs Unions and Trading Bloc "
     "memberships with per-member dates: the policy features.", "raw/wto"),
    ("Bank of England, series XUAAUSS", "USD → GBP annual average rates.", "src/exportspredictor/currency.py"),
    ("HMRC Overseas Trade Statistics, 2023 (Open Government Licence v3.0)", "Cross-check of BACI's UK totals "
     "only; never a model input.", "raw/hmrc"),
    ("UN Statistics Division HS 2022 → HS 2012 conversion table", "Puts HMRC's 2023 codes on BACI's revision "
     "for the cross-check.", "raw/hmrc"),
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
}


# --- Model card ----------------------------------------------------------------------------


def model_card(artefacts: Artefacts) -> dict:
    """The model card of one artefact version, built from its metadata and evaluation report."""
    metadata, report = artefacts.metadata, artefacts.evaluation or {}
    return {
        "version": artefacts.version,
        "owner": OWNER,
        "model": _model_summary(metadata),
        "intended_use": INTENDED_USE,
        "out_of_scope": OUT_OF_SCOPE,
        "metrics": _metrics(report),
        "limitations": [*_result_limitations(metadata, report), *LIMITATIONS],
        "data_sources": [{"name": n, "use": u, "location": loc} for n, u, loc in DATA_SOURCES],
        "decisions": [{"title": t, "text": x} for t, x in DECISIONS],
    }


def _model_summary(metadata: dict) -> str:
    params = metadata.get("params", {})
    return (
        f"XGBoost, Tweedie objective (variance power {params.get('tweedie_variance_power', '?')}), depth "
        f"{params.get('max_depth', '?')}, {metadata.get('rounds', '?')} rounds; {metadata.get('n_folds', '?')} "
        f"out-of-fold models trained on {_years(metadata.get('training_years'))}, scoring the "
        f"{metadata.get('baseline_year', '?')} Baseline. Source: {metadata.get('source', '?')}."
    )


def _years(years) -> str:
    return f"{min(years)}–{max(years)}" if years else "?"


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
    for event in report.get("event_backtest", {}).get("events", []):
        for direction in event["directions"]:
            pooled = direction["pooled"]
            rows.append({"name": f"Event Backtest: {direction['label']}", "model": pooled["predicted"],
                         "naive": pooled["observed"], "percent": True,
                         "text": f"model's Scenario Effect against the observed change (95% interval "
                         f"{_ci(pooled['observed_ci'], percent=True)})"})
    uk = report.get("accuracy", {}).get("uk_out_of_fold", {})
    for group in uk.get("by_income", []):
        rows.append({"name": f"UK ranking, {group['group']} Markets", "model": group["model_spearman_within_product"],
                     "naive": group["naive_spearman_within_product"],
                     "text": f"{group['non_zero_flows']:,} UK Flows in {group['rows']:,} pairs"})
    return rows


def _ci(interval, *, percent: bool = False) -> str:
    low, high = interval or (None, None)
    spec = "+.0%" if percent else ".2f"
    return f"{_f(low, spec)} to {_f(high, spec)}"


def _f(value, spec: str) -> str:
    """Format a number that may be missing (tiny fixture worlds leave some results undefined)."""
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def _result_limitations(metadata: dict, report: dict) -> list[dict]:
    limits = []
    headline = report.get("headline")
    if headline:
        limits.append({"title": "Only slightly better than a naive ranking for UK pairs",
                       "text": f"Within a Product, the model ranks UK Markets with a rank correlation of "
                       f"{_f(headline['model'], '.2f')}, against {_f(headline['naive'], '.2f')} for the UK's market share times "
                       f"each Market's imports. Its value is mostly in the size of Export Potential, not its order."})
    lift = report.get("lift")
    if lift:
        limits.append({"title": "Large Export Gaps mostly mark large pairs",
                       "text": f"Lift is {_f(lift['lift'], '.1f')}, but against pairs of similar Export Potential it is "
                       f"{_f(lift['size_matched']['lift'], '.2f')}: an Opportunity is worth a look, not more likely to grow "
                       f"than its peers of the same size."})
    events = report.get("event_backtest", {}).get("events", [])
    if events:
        pooled = events[0]["directions"][0]["pooled"]
        limits.append({"title": "Scenario Effects are larger than observed changes",
                       "text": f"For {events[0]['title'][0].lower() + events[0]['title'][1:]}, the model associated the agreement with "
                       f"{_f(pooled['predicted'], '+.0%')}, against an observed {_f(pooled['observed'], '+.0%')}. Read Scenario "
                       f"Effects as an upper-end association."})
    rate = metadata.get("non_uk_sampling_rate")
    if rate is not None:
        limits.append({"title": "Trained on a sample of other Exporters' Flows",
                       "text": f"The models train on every UK row but a reweighted {rate:.0%} sample of other "
                       f"Exporters' rows, to fit the GPU's memory. Unbiased, but noisier than training on all rows."})
    return limits


def render_model_card(card: dict) -> str:
    """The model card as Markdown."""
    lines = [f"# Model card: UK Exports Predictor, version {card['version']}", "",
             f"**Owner:** {card['owner']}", "", f"**Model:** {card['model']}", ""]
    lines += ["## Intended use", "", *[f"- {item}" for item in card["intended_use"]], ""]
    lines += ["## Out-of-scope use", "", *[f"- {item}" for item in card["out_of_scope"]], ""]
    lines += ["## Metrics", "", "| Metric | Model | Reference (naive baseline, chance, or observed) | Note |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {m['name']} | {_number(m['model'], m.get('percent', False))} | "
        f"{_number(m['naive'], m.get('percent', False))} | {m['text']} |"
        for m in card["metrics"]
    ]
    lines += ["", "## Limitations", "", *[f"- **{x['title']}.** {x['text']}" for x in card["limitations"]], ""]
    lines += ["## Data sources", "", *[f"- **{s['name']}** ({s['location']}): {s['use']}" for s in card["data_sources"]], ""]
    lines += ["## Decisions behind the model", "", *[f"- **{d['title']}.** {d['text']}" for d in card["decisions"]], ""]
    return "\n".join(lines)


def _number(value, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if percent:
        return f"{value:+.0%}"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}"


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
    edges = sorted(set(reference.drop_nulls().quantile(q / bins, "linear") for q in range(1, bins)))
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
