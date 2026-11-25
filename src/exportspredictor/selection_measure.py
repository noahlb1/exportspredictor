"""The Selection Measure (ADR 0008): the mean over Products of a tie-proof rank correlation across Markets.

Within each Product, Export Potential is ranked against Actual Exports across the
Product's Markets, and the correlations are averaged over Products (each Product with at
least `MIN_RANKED_ROWS` rows and a defined correlation). Every choice the project makes is
made on it, and the evaluation's headline is the same measure on the test year.

**Ties.** Ties in Export Potential are broken in a fixed pseudo-random order, a stable hash
of the Product and the Market (`TIE_KEY`), so the result is deterministic and does not depend
on row order. Actual Exports keep average ranks for their ties (the zeros). With average
ranks on both sides, tying the bottom of a ranking (flooring Export Potential at $1,000)
lined the tied predictions up with the tied zeros and raised the measure without ranking
anything better: on 2022 it lifted the Blend from 0.5635 to 0.6062 (open review of 2022). A
tied block broken at random has, on average, the correlation of a randomly ordered block, so
tying can only lose information. Without ties the measure is the ordinary Spearman mean.

`legacy_selection_measure` is the old measure (average ranks on both sides), kept for
continuity tables only: earlier figures (v1, v2, the Blend's 2023 read) were made with it.
Nothing is chosen on it.

Every within-Product rank correlation in the package goes through
`within_product_correlations`: the experiment loop's selection and round-by-round early
stopping, the evaluation's headline, breakdowns and stability check, and the v1 comparison.
"""

import zlib
from collections.abc import Sequence

import polars as pl

# A Product (or Product within a group) needs this many rows for its rank correlation.
MIN_RANKED_ROWS = 3
# The pseudo-random order that breaks ties in Export Potential (`with_tie_key`).
TIE_KEY = "_tie_key"
# Park–Miller's prime and multiplier: the tie key stays below 2**31, so no step overflows Int64.
_PRIME = 2_147_483_647
_MULTIPLIER = 48_271
_SHIFT = 1_000_003


def selection_measure(
    rows: pl.DataFrame,
    prediction: str,
    *,
    target: str = "value_usd",
    product: str = "hs6",
    market: str | Sequence[str] = "importer",
) -> float | None:
    """The Selection Measure of `prediction`: the mean over Products of its tie-proof rank correlation with `target`.

    `rows` hold one row per Product and Market (`product`, `market`); `market` may name
    several columns (exporter and importer, for rows of every Exporter). None when no
    Product can be ranked.
    """
    within = within_product_correlations(rows, prediction, target=target, product=product, market=market)
    return _mean(within["rho"])


def legacy_selection_measure(rows: pl.DataFrame, prediction: str, *, target: str = "value_usd") -> float | None:
    """The measure before it was made tie-proof, with average ranks for ties on both sides: for continuity tables only."""
    within = within_product_correlations(rows, prediction, target=target, tie_proof=False)
    return _mean(within["rho"])


def within_product_correlations(
    rows: pl.DataFrame,
    prediction: str,
    *,
    target: str = "value_usd",
    product: str = "hs6",
    market: str | Sequence[str] = "importer",
    by: Sequence[str] = (),
    tie_proof: bool = True,
) -> pl.DataFrame:
    """Per Product (within each `by` group), the rank correlation `rho` of `prediction` with `target`.

    Only Products with at least `MIN_RANKED_ROWS` rows and a defined correlation (neither
    side constant) are returned, with their row count `rows`. `tie_proof` breaks ties in
    `prediction` by the tie key; False gives the legacy average ranks, and then `market`
    is not read.
    """
    groups = [*by, product]
    if tie_proof:
        keyed = rows if TIE_KEY in rows.columns else with_tie_key(rows, product=product, market=market)
        ranked = keyed.sort(*groups, prediction, TIE_KEY).select(
            *groups,
            _p=pl.int_range(pl.len()).over(groups).cast(pl.Float64),
            _t=pl.col(target).rank("average").over(groups).cast(pl.Float64),
        )
        rho = pl.corr("_p", "_t")
    else:
        ranked, rho = rows, pl.corr(prediction, target, method="spearman")
    return (
        ranked.group_by(groups)
        .agg(rows=pl.len(), rho=rho)
        .filter((pl.col("rows") >= MIN_RANKED_ROWS) & pl.col("rho").is_not_null() & pl.col("rho").is_not_nan())
    )


def with_tie_key(rows: pl.DataFrame, *, product: str = "hs6", market: str | Sequence[str] = "importer") -> pl.DataFrame:
    """`rows` with `TIE_KEY`: a stable pseudo-random number from the Product and Market of each row.

    It is the same on every machine and run (CRC-32 of each value, mixed by integer
    arithmetic). Adding it once lets a caller that measures the same rows many times (the
    loop's early stopping) skip rehashing. A Product code is read as six digits, so an
    integer and a string column give the same key.
    """
    key = pl.lit(0, pl.Int64)
    for column in [product, *([market] if isinstance(market, str) else market)]:
        values = pl.col(column).cast(pl.String)
        if column == product:
            values = values.str.zfill(6)
        unique = rows.select(values.unique())[column].to_list()
        crc = {value: zlib.crc32(str(value).encode()) % _PRIME for value in unique}
        part = values.replace_strict(crc, return_dtype=pl.Int64)
        key = (((key * _SHIFT + part) % _PRIME) * _MULTIPLIER) % _PRIME
    return rows.with_columns(((key * _MULTIPLIER) % _PRIME).alias(TIE_KEY))


def _mean(values: pl.Series) -> float | None:
    return float(values.mean()) if values.len() else None  # type: ignore[arg-type]
