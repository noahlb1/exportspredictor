"""Market Reach (ADR 0010): how widely the UK sends its Products to a Market, against how widely expected.

The ranking does not know where the UK already sells: the open review of 2022
(`docs/research/open-review-2022.md`, idea 1) found that most of the Selection Measure's loss
is in which Markets get any UK trade. Market Reach is a per-Market offset learned from the
UK's *other* Products in the same Market, and Export Potential becomes

    Export Potential × exp(β · offset) × c,

with c one constant so that the UK's summed Export Potential over the grid is unchanged. Only
that grand total is kept: each Market's dollar level moves with its multiplier, so every
dollar figure built on Export Potential (a Scenario Effect in dollars, a Blocked Opportunity
value and the $1M floor it must clear) follows Market Reach. The offset is the same for every
Product of a Market, so a Scenario's relative effect is unchanged.

**The form (the open review's share form, nested).** For fold k and scored year t,
the offset of Market m is learned from the UK pairs *outside* fold k in year t − `LAG_YEARS`:

    offset = log((sent + a) / (imported + 2a)) − log((Σ sent + a) / (Σ imported + 2a)),

where `imported` counts the Market's source pairs whose Product the Market imports from any
other Exporter (`log_demand_loo` > 0), `sent` those of them that also have a UK Flow, the sums
run over every Market's source pairs, and a is `PSEUDO_COUNT`. The first term is the share of
the Products the Market imports from others that the UK sends there too, shrunk towards a half;
the second is the UK's pooled share, the share expected of every Market, so an offset of 0
means "as expected". `products_expected` is that pooled share times `imported`. A Market with
no source pairs gets 0.

`sent` is nested in `imported`, so the share is at most 1 and the offset is bounded. The first
form ("share") counted every UK Flow as sent. A Market that reports no imports of its own
is seen in BACI only through its Exporters' declarations, so the UK could send it many Products
"no other Exporter" supplies: the Falkland Islands' share was about 3, its offset about 2.1
(Ireland, the next, 0.92) and its multiplier about 230×, which made it the fourth Market by
Blocked Opportunity value. The nested form ("nested_share") was re-checked on 2022 under the
same pre-registered procedure (`docs/research/market-reach-2022.md`, ADR 0010).

**Leakage (ADR 0001).** A pair is held out with its fold, so its own Flows (any year) never
enter its offset; and only year t − 1 is read, so no UK Flow of the scored year does. The
offset is post-processing in the Model Form, not a model feature: it is built from UK Flows
into the target Market in other Products, which a feature may not use.

**The adoption gate.** β comes from `BETA_GRID`, chosen on folds 0–2 of 2022 under the
tie-proof Selection Measure, and is kept only if it beats the Win Threshold on the pooled
five-fold 2022 grid (`scripts/research/market_reach_2022/`). `FORM`, `BETA_GRID`,
`PSEUDO_COUNT` and `LAG_YEARS` were committed before any result was read (commit ed982b0);
the experiment chose and adopted `CHOSEN_BETA` (`docs/research/market-reach-2022.md`). The
nested form changed only which pairs count as sent, and was re-run through the same script.

The model associates broad UK reach into a Market with higher Export Potential there; it
does not say that reach causes exports.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.features import FEATURES
from exportspredictor.folds import N_FOLDS, score_out_of_fold, with_folds
from exportspredictor.prediction import MarketReach, ModelForm

FORM = "nested_share"  # the open review's share form with sent nested in imported (the first "share" was not)
BETA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
PSEUDO_COUNT = 5.0
LAG_YEARS = 1
# Chosen from BETA_GRID on folds 0–2 of 2022 and adopted on the five-fold check (+0.0406 against a Win
# Threshold of 0.008869; chosen again for the nested form, +0.0449): docs/research/market-reach-2022.md.
CHOSEN_BETA = 4.0
DEMAND = "log_demand_loo"  # the Market's imports of the Product from Exporters other than the UK (log1p)
TABLE_COLUMNS = ("fold", "importer", "products_sent", "products_imported", "products_expected", "offset")


def reach_table(
    uk_rows: pl.DataFrame, scored_year: int, *, n_folds: int = N_FOLDS, pseudo_count: float = PSEUDO_COUNT
) -> pl.DataFrame:
    """The Market Reach table for `scored_year`: one row per fold and Market (`TABLE_COLUMNS`).

    `uk_rows` are UK grid rows (`year`, `fold`, `hs6`, `importer`, `value_usd`, `log_demand_loo`);
    only year `scored_year` − `LAG_YEARS` is read. Fold k's offsets come from the pairs outside fold k.
    Rows with a `weight` column (training rows, whose Zero Cells are sampled) are counted by their
    weight, so the counts stay unbiased (`exporter_reach`); the UK grid has every pair, weight 1.
    """
    count = pl.col("weight").cast(pl.Float64) if "weight" in uk_rows.columns else pl.lit(1, pl.Int64)
    source = uk_rows.filter(pl.col("year") == scored_year - LAG_YEARS).select(
        "fold",
        pl.col("importer").cast(pl.String),
        sent=pl.when((pl.col("value_usd") > 0) & (pl.col(DEMAND) > 0)).then(count).otherwise(0),
        imported=pl.when(pl.col(DEMAND) > 0).then(count).otherwise(0),
    )
    a = pseudo_count
    tables = []
    for k in range(n_folds):
        outside = source.filter(pl.col("fold") != k)
        sent, imported = outside["sent"].sum(), outside["imported"].sum()
        pooled = sent / imported if imported else 0.0
        log_pooled = np.log((sent + a) / (imported + 2 * a))
        tables.append(
            outside.group_by("importer")
            .agg(products_sent=pl.col("sent").sum(), products_imported=pl.col("imported").sum())
            .select(
                fold=pl.lit(k, pl.Int8),
                importer="importer",
                products_sent="products_sent",
                products_imported="products_imported",
                products_expected=pl.col("products_imported") * pooled,
                offset=((pl.col("products_sent") + a) / (pl.col("products_imported") + 2 * a)).log() - log_pooled,
            )
        )
    return pl.concat(tables).sort("fold", "importer")


def apply(
    potential: np.ndarray, rows: pl.DataFrame, table: pl.DataFrame, *, beta: float, constant: float,
    cap: float | None = None,
) -> np.ndarray:
    """Export Potential × exp(β · offset) × `constant` for `rows` (`fold`, `importer`), in their order.

    A row whose fold and Market have no offset (another Exporter's row, a Market without source
    pairs) is multiplied by the constant alone. With `cap`, the multiplier exp(β · offset) is held at
    or below it before the constant (a final-round experiment); None caps nothing.
    """
    return np.asarray(potential, dtype=np.float64) * _multiplier(rows, table, beta, cap) * constant


def rescale_constant(
    potential: np.ndarray, rows: pl.DataFrame, table: pl.DataFrame, *, beta: float, weight: np.ndarray | None = None,
    cap: float | None = None,
) -> float:
    """c such that `apply` with it (and the same `cap`) keeps the summed Export Potential of `rows` (weighted by
    `weight`) unchanged."""
    potential = np.asarray(potential, dtype=np.float64)
    if weight is not None:
        potential = potential * np.asarray(weight, dtype=np.float64)
    return float(potential.sum() / (potential * _multiplier(rows, table, beta, cap)).sum())


def exporter_reach(
    potential: np.ndarray, rows: pl.DataFrame, source: pl.DataFrame, *, scored_year: int, beta: float,
    n_folds: int = N_FOLDS,
) -> np.ndarray:
    """`potential` with Market Reach applied to every Exporter's own rows, as the served Model Form applies the
    UK's to UK pairs: Lift measures the served form on held-out Exporters (`backtests.lift_rows`), not on the UK.

    `rows` are one year's training rows (`exporter`, `hs6`, `importer`, `weight`), in `potential`'s order;
    `source` the same Exporters' training rows of `scored_year` − `LAG_YEARS` (`year`, `exporter`, `hs6`,
    `importer`, `value_usd`, `log_demand_loo`, `weight`). Each Exporter gets its own table (`reach_table`,
    counted by weight, since other Exporters' Zero Cells are sampled) from its pairs outside each pair's
    fold, so a pair's own Flows never reach its offset (ADR 0001), and its own c, which keeps its weighted
    summed Export Potential unchanged.
    """
    potential = np.asarray(potential, dtype=np.float64)
    out = potential.copy()
    exporters = rows["exporter"].cast(pl.String).to_numpy()
    keyed = with_folds(rows.select("hs6", pl.col("importer").cast(pl.String)), n_folds)
    for exporter in np.unique(exporters):
        mask = exporters == exporter
        own = source.filter(pl.col("exporter").cast(pl.String) == exporter).select(
            "year", "hs6", pl.col("importer").cast(pl.String), "value_usd", DEMAND, "weight"
        )
        table = reach_table(with_folds(own, n_folds), scored_year, n_folds=n_folds)
        pairs = keyed.filter(pl.Series(mask))
        weight = rows.filter(pl.Series(mask))["weight"].to_numpy()
        constant = rescale_constant(potential[mask], pairs, table, beta=beta, weight=weight)
        out[mask] = apply(potential[mask], pairs, table, beta=beta, constant=constant)
    return out


def uk_keys(rows: pl.DataFrame, *, fold: pl.Expr, uk: str) -> pl.DataFrame:
    """The keys a Market Reach offset is looked up by, one row per row of `rows`: `fold` (the `fold` expression on
    UK rows, null on other Exporters' rows, which take no offset), `importer` (the row's `importer`, or `iso3`) and
    `uk` (whether the row is a UK pair: rows without an `exporter` column are). The served form
    (`prediction.reach_multiplier`) and the experiment loop's Arms (`loop.runner`) both key their rows here."""
    market = "importer" if "importer" in rows.columns else "iso3"
    is_uk = pl.col("exporter").cast(pl.String) == uk if "exporter" in rows.columns else pl.lit(True)
    return rows.select(
        fold=pl.when(is_uk).then(fold).cast(pl.Int8), importer=pl.col(market).cast(pl.String), uk=is_uk
    )


def _multiplier(rows: pl.DataFrame, table: pl.DataFrame, beta: float, cap: float | None = None) -> np.ndarray:
    offsets = (
        rows.select(pl.col("fold").cast(pl.Int8), pl.col("importer").cast(pl.String))
        .join(table.select("fold", "importer", "offset"), on=["fold", "importer"], how="left", maintain_order="left")
        ["offset"]
        .fill_null(0.0)
        .to_numpy()
    )
    multiplier = np.exp(beta * offsets)
    return multiplier if cap is None else np.minimum(multiplier, cap)


def settings(beta: float = CHOSEN_BETA) -> dict:
    """Market Reach's configuration as the artefact metadata records it (a test-year fingerprint key, ADR 0009)."""
    return {"form": FORM, "beta": beta, "pseudo_count": PSEUDO_COUNT, "lag_years": LAG_YEARS}


def fit(
    form: ModelForm,
    models: Sequence[xgb.Booster],
    grid: pl.DataFrame,
    source: pl.DataFrame,
    *,
    scored_year: int,
    uk: str,
    beta: float = CHOSEN_BETA,
    features: Sequence[str] = FEATURES,
) -> ModelForm:
    """`form` with Market Reach fitted for `scored_year`: its table and its rescale constant.

    `grid` is the scored year's UK grid as the models see it (transformed, with `naive_usd` for a
    blend); `source` is the UK grid of the reference year, `scored_year` − `LAG_YEARS` (untransformed:
    `year`, `hs6`, `importer`, `value_usd`, `log_demand_loo`). Folds are the pairs' hash folds, so a
    pair's own Flows never reach the offsets of its own fold. c is computed once, on `grid` scored by
    `form` without Market Reach, so the UK's summed Export Potential over the grid is unchanged; every
    Scenario then reuses the same c.
    """
    table = reach_table(with_folds(source, len(models)), scored_year, n_folds=len(models))
    plain = form.without_market_reach()
    scored = score_out_of_fold(grid, list(models), uk, inputs=features, form=plain)
    rows = scored.select("fold", importer=pl.col("importer").cast(pl.String))
    constant = rescale_constant(scored["export_potential_usd"].to_numpy(), rows, table, beta=beta)
    reach = MarketReach(
        beta=beta, form=FORM, pseudo_count=PSEUDO_COUNT, lag_years=LAG_YEARS,
        reference_year=scored_year - LAG_YEARS, constant=constant, table=table,
    )
    return replace(plain, market_reach=reach)


def by_market(table: pl.DataFrame, n_folds: int = N_FOLDS) -> pl.DataFrame:
    """Each Market's reach over every UK pair, from its per-fold rows: importer, products_sent,
    products_imported, products_expected and offset (the mean of its folds' offsets).

    Each fold's counts come from the pairs outside it, so each pair is counted in all folds but its
    own; dividing the sums by (folds − 1) gives the counts over every pair.
    """
    others = max(n_folds - 1, 1)
    return (
        table.group_by("importer")
        .agg(
            products_sent=(pl.col("products_sent").sum() / others).round().cast(pl.Int64),
            products_imported=(pl.col("products_imported").sum() / others).round().cast(pl.Int64),
            products_expected=pl.col("products_expected").sum() / others,
            offset=pl.col("offset").mean(),
        )
        .sort("importer")
    )


def market_figures(
    table: pl.DataFrame, iso3: str, *, market: str, year: int, n_folds: int = N_FOLDS
) -> dict | None:
    """One Market's reach as Advisors see it (`by_market`), with its sentence; None for a Market without one.

    Returns year (the reference year), products_sent, products_imported, products_expected, offset and text.
    """
    found = by_market(table.filter(pl.col("importer") == iso3), n_folds)
    if found.is_empty():
        return None
    row = found.row(0, named=True)
    return {
        "year": year,
        "products_sent": row["products_sent"],
        "products_imported": row["products_imported"],
        "products_expected": row["products_expected"],
        "offset": row["offset"],
        "text": reach_text(row["products_sent"], row["products_imported"], row["products_expected"], market, year),
    }


def pair_figures(
    table: pl.DataFrame, iso3: str, fold: int, *, market: str, year: int, beta: float, n_folds: int = N_FOLDS
) -> dict | None:
    """`market_figures` with the offset applied to a pair of `fold` (`pair_offset`) and its multiplier
    exp(β · offset) (`pair_multiplier`, before the rescale constant); None for a Market without a reach."""
    figures = market_figures(table, iso3, market=market, year=year, n_folds=n_folds)
    if figures is None:
        return None
    applied = table.filter((pl.col("fold") == fold) & (pl.col("importer") == iso3))["offset"]
    offset = float(applied.item()) if applied.len() else 0.0
    return {**figures, "pair_offset": offset, "pair_multiplier": float(np.exp(beta * offset))}


def reach_text(sent: int, imported: int, expected: float, market: str, year: int) -> str:
    """A Market's reach in one sentence, so a Market the UK reaches less widely than expected stays visible beside
    Product-level Export Gaps. `sent` counts only Products the Market also imports from other Exporters, so it
    is at most `imported`."""
    return (
        f"In {year} {market} imported {imported:,} Products from other Exporters and the UK sent {sent:,} of them; "
        f"at the UK's usual reach across Markets, about {expected:,.0f} were expected."
    )
