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
`MIN_BAND_ROWS` rows merges with its smaller neighbour (`fit_bands`). A band spanning more
than tenfold is split at $10m, $100m and $1bn where each part keeps the minimum rows, so a
large pair is calibrated on pairs of similar size. Residuals are on the
log scale, log(1 + Actual Exports) − log(1 + Export Potential), so the range is a ratio of
Export Potential, not a fixed number of dollars: dollar offsets gave every pair in the top
band, from about $2m to over $1bn, the same width (ADR 0007, amended). Each band's 10th and
90th percentiles of signed log residuals are its offsets; sampled Zero Cells count by their
weight. The range is exp(log(1 + Export Potential) + offset) − 1, clipped at $0 (`potential_range`).
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
    export_potential_usd and residual_log (`log_residual`).
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
    ).with_columns(residual_log=log_residual(pl.col("actual_exports_usd"), pl.col("export_potential_usd")))


def log_residual(actual: pl.Expr, potential: pl.Expr) -> pl.Expr:
    """log(1 + Actual Exports) − log(1 + Export Potential): a ratio, so a band's offsets scale with the pair."""
    return actual.log1p() - potential.log1p()


def potential_range_text(low_usd: float, high_usd: float, actual_exports_usd: float, *, uncertain: bool) -> str:
    """One pair's Potential Range in plain words, for its detail view (never a confidence interval)."""
    text = (
        f"**Potential Range {format_usd(low_usd)} to {format_usd(high_usd)}.** For comparable pairs, those "
        "with about the same Export Potential, Actual Exports fall in this range 80% of the time. It shows how "
        "much such pairs vary around their Export Potential, not how precisely Export Potential is known. "
    )
    if uncertain:
        return text + (
            f"UK Actual Exports of {format_usd(actual_exports_usd)} lie inside it, within the variation "
            "normally seen among comparable pairs: an Uncertain Opportunity."
        )
    where = "below" if actual_exports_usd < low_usd else "above" if actual_exports_usd > high_usd else "inside"
    return text + f"UK Actual Exports of {format_usd(actual_exports_usd)} lie {where} it."


N_BANDS = 20
MIN_BAND_ROWS = 1000  # a parameter: the fixture world has far fewer calibration rows
LOW_QUANTILE, HIGH_QUANTILE = 0.1, 0.9  # an 80% Potential Range
COVERAGE = HIGH_QUANTILE - LOW_QUANTILE
SCALE = "log"  # residuals and offsets are log(1 + dollars); earlier versions recorded none and used dollars
# A band spanning more than tenfold in Export Potential is split at these edges (ADR 0007, amended),
# so a $1bn pair is calibrated on large pairs, not on the $2m pairs at the bottom of the top band.
MAX_BAND_SPAN = 10.0
SPLIT_EDGES_USD = (1e7, 1e8, 1e9)
# Held-out coverage is also reported by these sizes of Export Potential: (label, lower edge).
SIZE_GROUPS = (("under $2m", 0.0), ("$2m–10m", 2e6), ("$10m–100m", 1e7), ("over $100m", 1e8))
COVERAGE_TOLERANCE = 0.05  # a size group further than this from the target is a known limitation


def fit_bands(scored: pl.DataFrame, *, n_bands: int = N_BANDS, min_rows: int = MIN_BAND_ROWS) -> pl.DataFrame:
    """The band offsets from scored calibration rows (`score_calibration`).

    Rows are sorted by Export Potential and cut into `n_bands` equal-count bands; while a
    band holds fewer than `min_rows` rows it merges with its smaller neighbour. A band whose
    Export Potential spans more than tenfold is then split at $10m, $100m and $1bn, as long as
    each part keeps `min_rows` rows (`_split`); a split band's lower edge is the split edge. Returns one
    row per band, by Export Potential: band, min_export_potential_usd (its lower edge; the
    first band starts at $0), rows, low_offset_log and high_offset_log (the weighted 10th
    and 90th percentiles of its log residuals, `log_residual`).
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
    edges = {start: float(potential[start]) for start in starts[:-1]}
    for start, end in pairwise(list(starts)):
        edges |= _split(potential, start, end, min_rows)
    starts = [*sorted(edges), n]
    residual = rows["residual_log"].to_numpy()
    weight = rows["weight"].to_numpy()
    bands = []
    for b, (start, end) in enumerate(pairwise(starts)):
        low, high = _weighted_quantiles(residual[start:end], weight[start:end], (LOW_QUANTILE, HIGH_QUANTILE))
        bands.append((b, 0.0 if b == 0 else edges[start], end - start, low, high))
    schema = {
        "band": pl.Int16,
        "min_export_potential_usd": pl.Float64,
        "rows": pl.Int64,
        "low_offset_log": pl.Float64,
        "high_offset_log": pl.Float64,
    }
    return pl.DataFrame(bands, schema=schema, orient="row")


def _split(potential: np.ndarray, start: int, end: int, min_rows: int) -> dict[int, float]:
    """Extra band starts, {row position: lower edge}, that split rows [start, end) at `SPLIT_EDGES_USD`.

    Only a band whose Export Potential spans more than `MAX_BAND_SPAN` is split. A new band with
    fewer than `min_rows` rows merges with the band above it; the topmost, with none above, merges
    with the band below. `potential` is sorted.
    """
    if potential[end - 1] <= MAX_BAND_SPAN * potential[start]:
        return {}
    cuts = {int(np.searchsorted(potential, edge, side="left")): edge for edge in SPLIT_EDGES_USD}
    cuts = {position: edge for position, edge in cuts.items() if start < position < end}
    while cuts:
        bounds = [start, *sorted(cuts), end]
        sizes = np.diff(bounds)
        small = next((i for i, size in enumerate(sizes) if size < min_rows), None)
        if small is None:
            break
        # A band merging upwards loses its upper boundary; the topmost band loses its lower one.
        del cuts[bounds[small + 1] if small < len(sizes) - 1 else bounds[small]]
    return cuts


def potential_range(
    table: pl.DataFrame, bands: pl.DataFrame, *, potential: str = "export_potential_usd"
) -> pl.DataFrame:
    """`table` with `band`, `potential_range_low_usd` and `potential_range_high_usd`.

    Each row takes the log offsets of the band its `potential` column falls in, applied to
    log(1 + Export Potential) and turned back into dollars, clipped at $0. Row order is kept.
    """
    edges = bands.select("band", "min_export_potential_usd", "low_offset_log", "high_offset_log")
    indexed = table.with_row_index("_row")
    banded = (
        indexed.select("_row", pl.col(potential).alias("_potential"))
        .sort("_potential")
        .join_asof(edges, left_on="_potential", right_on="min_export_potential_usd", strategy="backward")
        # Below the first edge ($0) only if negative; Tweedie predictions never are.
        .with_columns(pl.col("band", "low_offset_log", "high_offset_log").fill_null(strategy="backward"))
        .select(
            "_row",
            "band",
            potential_range_low_usd=_shifted("low_offset_log"),
            potential_range_high_usd=_shifted("high_offset_log"),
        )
    )
    return indexed.join(banded, on="_row", how="left").sort("_row").drop("_row")


def _shifted(offset: str) -> pl.Expr:
    """exp(log(1 + Export Potential) + offset) − 1, clipped at $0."""
    return (pl.col("_potential").log1p() + pl.col(offset)).exp().sub(1.0).clip(lower_bound=0.0)


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
    coverage, weighted as the bands are. Reported overall, per band, per Exporter and by size of
    Export Potential (`SIZE_GROUPS`; a group with no held-out rows has coverage None), with the
    `target` (80%). It checks the assumption that comparable pairs vary alike across Exporters,
    and that the range fits pairs of every size (ADR 0007).
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
            "coverage": float((rows["covered"] * rows["weight"]).sum() / weight) if weight else None,
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
        "by_size": [
            {"size": label, "min_export_potential_usd": edge, **summary(checked.filter(in_group))}
            for (label, edge), in_group in zip(SIZE_GROUPS, _size_filters(), strict=True)
        ],
    }


def _size_filters() -> list[pl.Expr]:
    """For each of `SIZE_GROUPS`, whether a row's Export Potential lies in it."""
    potential = pl.col("export_potential_usd")
    edges = [edge for _, edge in SIZE_GROUPS]
    return [
        (potential >= low) & (potential < high) if high is not None else potential >= low
        for low, high in zip(edges, [*edges[1:], None], strict=True)
    ]


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles) -> list[float]:
    """Weighted quantiles (inverted CDF): the first sorted value whose cumulative weight reaches q of the total."""
    order = np.argsort(values, kind="stable")
    values, cumulative = values[order], np.cumsum(weights[order])
    total = cumulative[-1]
    return [float(values[min(np.searchsorted(cumulative, q * total, side="left"), len(values) - 1)]) for q in quantiles]


def coverage_text(report: dict) -> str:
    """The Potential Range's held-out coverage and the share of Uncertain Opportunities, in one paragraph.

    The Uncertain Opportunity is reported only as this share, never as a badge per row (ADR 0007). A report with
    `uk_pair_coverage` (in newer reports) also says how many UK pairs' Actual Exports lie inside their
    range, with Market Reach and without it.
    """
    text = (
        f"On a held-out half of the calibration rows ({report['held_out_rows']:,} other Exporters' rows), "
        f"{report['coverage']:.0%} of their Actual Exports fell inside their Potential Range, against a target of "
        f"{report['target']:.0%}. {report['uncertain_opportunities']:,} of the {report['opportunities']:,} "
        f"Opportunities ({report['uncertain_opportunity_share']:.0%}) have UK Actual Exports inside their "
        "Potential Range (Uncertain Opportunities): most Export Gaps are within the variation normally seen among "
        "comparable pairs, so an Export Gap says where to look rather than proving an Opportunity."
    )
    uk = report.get("uk_pair_coverage")
    if not uk or uk.get("with_market_reach") is None:
        return text
    without = uk.get("without_market_reach")
    return text + (
        f" Over every UK pair, {uk['with_market_reach']:.0%} of UK pairs have Gap Period Actual Exports inside their "
        f"Potential Range" + ("" if without is None else f" ({without:.0%} without Market Reach)") + "."
    )


def pair_coverage(baseline: pl.DataFrame) -> float | None:
    """The share of Baseline pairs whose Gap Period mean Actual Exports lie inside their Potential Range.

    For UK pairs this is not the range's coverage (UK rows are never calibration rows, and a UK residual is its
    Export Gap): it is a check that a change to UK Export Potential, such as Market Reach (ADR 0010), does not move
    the range away from comparable UK-like pairs. None for an empty Baseline.
    """
    if baseline.is_empty():
        return None
    inside = (pl.col("actual_exports_usd") >= pl.col("potential_range_low_usd")) & (
        pl.col("actual_exports_usd") <= pl.col("potential_range_high_usd")
    )
    return float(baseline.select(inside.mean()).item())
