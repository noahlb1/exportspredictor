"""The Potential Range (ADR 0007): an 80% split-conformal range for each UK pair's Actual Exports.

Calibration rows are other Exporters' Baseline-year Flows and Zero Cells (the Baseline
year's training rows: within the training scope, Excluded Products left out). **No UK row
is ever a calibration row**: a UK residual is its Export Gap with the sign flipped, so a
range calibrated on it would call every gap normal. The fold models train only on years
before the Baseline year, so these residuals are out of sample. Each row is scored by the
fold model its hash assigns it to (`folds.assign_folds` keyed on Exporter, Product and
Market), as UK pairs are.

Calibration is Mondrian: rows are split into `N_BANDS` equal-count bands of Export
Potential, because Tweedie errors grow with the prediction, and a band with fewer than
`MIN_BAND_ROWS` rows merges with its smaller neighbour (`fit_bands`). Each band's 10th and
90th percentiles of signed residuals are its offsets; sampled Zero Cells count by their
weight. The range is the offsets around Export Potential, clipped at $0 (`potential_range`).
For a UK pair they go around the Gap Period mean Export Potential, and an Opportunity
whose Gap Period mean Actual Exports lie inside its range is an Uncertain Opportunity
(`with_potential_range`).

The Potential Range describes how much comparable pairs vary around their Export
Potential. It is not a confidence interval for Export Potential itself.
"""

from itertools import pairwise

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.currency import format_usd
from exportspredictor.exclusions import drop_excluded
from exportspredictor.features import FeatureTransform
from exportspredictor.folds import EXPORTER_PAIR_KEYS, assign_folds
from exportspredictor.prediction import MODEL_ALONE, ModelForm, export_potential


def score_calibration(
    rows: pl.DataFrame,
    models: list[xgb.Booster],
    transform: FeatureTransform,
    *,
    uk: str,
    form: ModelForm = MODEL_ALONE,
) -> pl.DataFrame:
    """The calibration rows of a Baseline-year training table, scored by their hash-assigned fold models.

    They are scored in the served model `form` (a blend needs `naive_usd` on the rows), so the
    Potential Range is calibrated on the Export Potential the app shows (ADR 0007).

    `rows` is a feature table from `features.training_rows` (untransformed). UK rows are
    dropped. Returns exporter, hs6, iso3 (strings), weight, fold, actual_exports_usd,
    export_potential_usd and residual_usd (Actual Exports minus Export Potential).
    """
    calibration = (
        transform.apply(rows.filter(pl.col("exporter").cast(pl.String) != uk))
        .pipe(drop_excluded)
        .with_columns(
            exporter=pl.col("exporter").cast(pl.String),
            hs6=pl.col("hs6").cast(pl.String),
            iso3=pl.col("importer").cast(pl.String),
        )
        .pipe(assign_folds, len(models), EXPORTER_PAIR_KEYS)
    )
    return calibration.select(
        "exporter",
        "hs6",
        "iso3",
        "weight",
        "fold",
        actual_exports_usd="value_usd",
        export_potential_usd=export_potential(calibration, models, form=form),
    ).with_columns(residual_usd=pl.col("actual_exports_usd") - pl.col("export_potential_usd"))


def potential_range_text(low_usd: float, high_usd: float, actual_exports_usd: float, *, uncertain: bool) -> str:
    """One pair's Potential Range in plain words, for its detail view (never a confidence interval)."""
    text = (
        f"**Potential Range {format_usd(low_usd)} to {format_usd(high_usd)}.** For comparable pairs, those "
        "with about the same Export Potential, Actual Exports fall in this range 80% of the time. It shows how "
        "much such pairs vary around their Export Potential, not how precisely Export Potential is known. "
    )
    if uncertain:
        return text + (
            f"UK Actual Exports of {format_usd(actual_exports_usd)} lie inside it, so this Export Gap is within "
            "the variation normally seen among comparable pairs: an Uncertain Opportunity."
        )
    where = "below" if actual_exports_usd < low_usd else "above" if actual_exports_usd > high_usd else "inside"
    return text + f"UK Actual Exports of {format_usd(actual_exports_usd)} lie {where} it."


N_BANDS = 20
MIN_BAND_ROWS = 1000  # a parameter: the fixture world has far fewer calibration rows
LOW_QUANTILE, HIGH_QUANTILE = 0.1, 0.9  # an 80% Potential Range
COVERAGE = HIGH_QUANTILE - LOW_QUANTILE


def fit_bands(scored: pl.DataFrame, *, n_bands: int = N_BANDS, min_rows: int = MIN_BAND_ROWS) -> pl.DataFrame:
    """The band offsets from scored calibration rows (`score_calibration`).

    Rows are sorted by Export Potential and cut into `n_bands` equal-count bands; while a
    band holds fewer than `min_rows` rows it merges with its smaller neighbour. Returns one
    row per band, by Export Potential: band, min_export_potential_usd (its lower edge; the
    first band starts at $0), rows, low_offset_usd and high_offset_usd (the weighted 10th
    and 90th percentiles of its residuals).
    """
    if scored.is_empty():
        raise ValueError("No calibration rows: the Potential Range needs other Exporters' Baseline-year rows")
    rows = scored.sort("export_potential_usd")
    n = rows.height
    # Band boundaries as row positions: [starts[b], starts[b + 1]).
    starts = sorted({b * n // n_bands for b in range(min(n_bands, n))}) + [n]
    while len(starts) > 2:
        sizes = np.diff(starts)
        smallest = int(np.argmin(sizes))
        if sizes[smallest] >= min_rows:
            break
        if smallest == 0:
            neighbour = 1
        elif smallest == len(sizes) - 1:
            neighbour = smallest - 1
        else:
            neighbour = smallest - 1 if sizes[smallest - 1] <= sizes[smallest + 1] else smallest + 1
        del starts[max(smallest, neighbour)]
    potential = rows["export_potential_usd"].to_numpy()
    residual = rows["residual_usd"].to_numpy()
    weight = rows["weight"].to_numpy()
    bands = []
    for b, (start, end) in enumerate(pairwise(starts)):
        low, high = _weighted_quantiles(residual[start:end], weight[start:end], (LOW_QUANTILE, HIGH_QUANTILE))
        bands.append((b, 0.0 if b == 0 else float(potential[start]), end - start, low, high))
    schema = {
        "band": pl.Int16,
        "min_export_potential_usd": pl.Float64,
        "rows": pl.Int64,
        "low_offset_usd": pl.Float64,
        "high_offset_usd": pl.Float64,
    }
    return pl.DataFrame(bands, schema=schema, orient="row")


def potential_range(
    table: pl.DataFrame, bands: pl.DataFrame, *, potential: str = "export_potential_usd"
) -> pl.DataFrame:
    """`table` with `band`, `potential_range_low_usd` and `potential_range_high_usd`.

    Each row takes the offsets of the band its `potential` column falls in, added to that
    Export Potential and clipped at $0. Row order is kept.
    """
    edges = bands.select("band", "min_export_potential_usd", "low_offset_usd", "high_offset_usd")
    indexed = table.with_row_index("_row")
    banded = (
        indexed.select("_row", pl.col(potential).alias("_potential"))
        .sort("_potential")
        .join_asof(edges, left_on="_potential", right_on="min_export_potential_usd", strategy="backward")
        # Below the first edge ($0) only if negative; Tweedie predictions never are.
        .with_columns(pl.col("band", "low_offset_usd", "high_offset_usd").fill_null(strategy="backward"))
        .select(
            "_row",
            "band",
            potential_range_low_usd=(pl.col("_potential") + pl.col("low_offset_usd")).clip(lower_bound=0.0),
            potential_range_high_usd=(pl.col("_potential") + pl.col("high_offset_usd")).clip(lower_bound=0.0),
        )
    )
    return indexed.join(banded, on="_row", how="left").sort("_row").drop("_row")


def with_potential_range(baseline: pl.DataFrame, bands: pl.DataFrame) -> pl.DataFrame:
    """The Baseline with each UK pair's Potential Range and its `uncertain` flag.

    The offsets go around the Gap Period mean Export Potential (`export_potential_usd`). A
    pair is `uncertain` (an Uncertain Opportunity) when it is an Opportunity (a positive
    Export Gap) and its Gap Period mean Actual Exports lie inside the range, edges included.
    """
    ranged = potential_range(baseline, bands).drop("band")
    actual = pl.col("actual_exports_usd")
    return ranged.with_columns(
        uncertain_opportunity=(pl.col("export_gap_usd") > 0)
        & (actual >= pl.col("potential_range_low_usd"))
        & (actual <= pl.col("potential_range_high_usd"))
    )


def uncertain_share(baseline: pl.DataFrame) -> dict:
    """How many Opportunities (positive Export Gap) there are and the share that are Uncertain Opportunities."""
    opportunities = baseline.filter(pl.col("export_gap_usd") > 0)
    uncertain = int(opportunities["uncertain_opportunity"].sum())
    return {
        "opportunities": opportunities.height,
        "uncertain_opportunities": uncertain,
        "uncertain_opportunity_share": uncertain / opportunities.height if opportunities.height else 0.0,
    }


_HALF_SALT = "potential-range-held-out"  # keeps the halves independent of the fold hash


def coverage_report(
    scored: pl.DataFrame, *, n_bands: int = N_BANDS, min_rows: int = MIN_BAND_ROWS
) -> dict:
    """Coverage of the Potential Range on a held-out half of the calibration rows (JSON-ready).

    The rows are halved by a hash of Exporter, Product and Market; bands are fitted on one
    half (with `min_rows` halved too, so the half keeps as many bands as the full set) and
    the share of the other half's rows whose Actual Exports fall inside their range is the
    coverage, weighted as the bands are. Reported overall, per band and per Exporter, with
    the `target` (80%). It checks the assumption that comparable pairs vary alike across
    Exporters (ADR 0007).
    """
    keyed = scored.with_columns(_salt=pl.lit(_HALF_SALT)).rename({"fold": "_scoring_fold"})
    halves = assign_folds(keyed, 2, ("_salt", *EXPORTER_PAIR_KEYS)).rename({"fold": "_half", "_scoring_fold": "fold"})
    calibration = halves.filter(pl.col("_half") == 0).drop("_salt", "_half")
    held_out = halves.filter(pl.col("_half") == 1).drop("_salt", "_half")
    bands = fit_bands(calibration, n_bands=n_bands, min_rows=max(1, min_rows // 2))
    actual = pl.col("actual_exports_usd")
    checked = potential_range(held_out, bands).with_columns(
        covered=(actual >= pl.col("potential_range_low_usd")) & (actual <= pl.col("potential_range_high_usd"))
    )

    def summary(rows: pl.DataFrame) -> dict:
        weight = rows["weight"].sum()
        return {
            "rows": rows.height,
            "coverage": float((rows["covered"] * rows["weight"]).sum() / weight) if weight else float("nan"),
        }

    return {
        "target": COVERAGE,
        "n_bands": n_bands,
        "min_band_rows": min_rows,
        "calibration_rows": calibration.height,
        "held_out_rows": held_out.height,
        **{key: value for key, value in summary(checked).items() if key == "coverage"},
        "by_band": [
            {"band": band, "min_export_potential_usd": edge, **summary(checked.filter(pl.col("band") == band))}
            for band, edge in bands.select("band", "min_export_potential_usd").iter_rows()
        ],
        "by_exporter": [
            {"exporter": exporter, **summary(checked.filter(pl.col("exporter") == exporter))}
            for exporter in sorted(checked["exporter"].unique())
        ],
    }


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles) -> list[float]:
    """Weighted quantiles (inverted CDF): the first sorted value whose cumulative weight reaches q of the total."""
    order = np.argsort(values, kind="stable")
    values, cumulative = values[order], np.cumsum(weights[order])
    total = cumulative[-1]
    return [float(values[min(np.searchsorted(cumulative, q * total, side="left"), len(values) - 1)]) for q in quantiles]


def coverage_text(report: dict) -> str:
    """The Potential Range's held-out coverage and the share of Uncertain Opportunities, in one paragraph.

    The Uncertain Opportunity is reported only as this share, never as a badge per row (ADR 0007).
    """
    return (
        f"On a held-out half of the calibration rows ({report['held_out_rows']:,} other Exporters' rows), "
        f"{report['coverage']:.0%} of their Actual Exports fell inside their Potential Range, against a target of "
        f"{report['target']:.0%}. {report['uncertain_opportunities']:,} of the {report['opportunities']:,} "
        f"Opportunities ({report['uncertain_opportunity_share']:.0%}) have UK Actual Exports inside their "
        "Potential Range (Uncertain Opportunities): most Export Gaps are within the variation normally seen among "
        "comparable pairs, so an Export Gap ranks a lead rather than proving one."
    )
