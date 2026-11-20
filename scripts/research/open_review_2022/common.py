"""Shared loading for the open review of the 2022 held-out ranking (read-only; 2022 only).

Every UK Product–Market pair of 2022 with its held-out scores (`loop_V202601/confirmation`),
the 2022 UK grid features, the fold, Product and Market lookups. Nothing here reads 2023.
"""

import os
from pathlib import Path

import numpy as np
import polars as pl

from exportspredictor.prediction import blend
from exportspredictor.selection_measure import legacy_selection_measure

DATA = Path(os.environ.get("EXPORTSPREDICTOR_DATA_DIR", r"C:\CodingProjectsC\exportspredictor\data"))
ARTEFACTS = Path(os.environ.get("EXPORTSPREDICTOR_ARTEFACTS_DIR", r"C:\CodingProjectsC\exportspredictor\artefacts"))
LOOP = DATA / "derived" / "loop_V202601"
YEAR = 2022
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "frame_2022.parquet"


def latest_artefact() -> Path:
    return max(p for p in ARTEFACTS.iterdir() if p.is_dir())


def frame() -> pl.DataFrame:
    """One row per UK pair of 2022: scores, fold, features, Sector, Market name and region."""
    if CACHE.exists():
        return pl.read_parquet(CACHE)
    folds = pl.concat(
        pl.read_parquet(LOOP / "confirmation" / f"fold_{k}.parquet")
        .with_columns(pl.col("hs6").cast(pl.String), fold=pl.lit(k, pl.Int8))
        for k in range(5)
    )
    grid = pl.read_parquet(DATA / "derived" / "features_V202601" / f"uk_grid_{YEAR}.parquet").with_columns(
        pl.col("hs6").cast(pl.String)
    ).drop("exporter", "year", "value_usd", "weight")
    art = latest_artefact()
    products = pl.read_parquet(art / "products.parquet").select("hs6", "description")
    markets = pl.read_parquet(art / "markets.parquet").rename({"iso3": "importer", "name": "market"})
    df = (
        folds.join(grid, on=["hs6", "importer"], how="left")
        .join(products, on="hs6", how="left")
        .join(markets, on="importer", how="left")
        .with_columns(hs2=pl.col("hs6").str.slice(0, 2), hs4=pl.col("hs6").str.slice(0, 4))
    )
    assert df.height == folds.height
    df.write_parquet(CACHE)
    return df


def measure(df: pl.DataFrame, col: str) -> float:
    """The Selection Measure of `col` on `df` as the review measured it, before ties were broken (ticket #115)."""
    return legacy_selection_measure(df, col)


def per_product(df: pl.DataFrame, col: str, target: str = "value_usd") -> pl.DataFrame:
    """Per Product (>= 3 rows): Spearman of `col` with `target`, and the Product's row and UK totals."""
    return (
        df.group_by("hs6")
        .agg(
            n=pl.len(),
            rho=pl.corr(col, target, method="spearman"),
            uk_usd=pl.col("value_usd").sum(),
            nonzero=(pl.col("value_usd") > 0).sum(),
        )
        .filter((pl.col("n") >= 3) & pl.col("rho").is_not_nan() & pl.col("rho").is_not_null())
    )


def log_blend(model_log: np.ndarray, naive_usd: np.ndarray, w: float = 0.3) -> np.ndarray:
    return (1 - w) * model_log + w * np.log(np.maximum(naive_usd, 1.0))


def served_blend(df: pl.DataFrame, model_col: str = "v2", w: float = 0.3) -> np.ndarray:
    return blend(df[model_col].to_numpy(), df["naive_usd"].to_numpy(), w)


def dollar_spread(df: pl.DataFrame, col: str, floor_usd: float = 1e7) -> dict:
    """10th–90th spread of log(1+actual) − log(1+col) on pairs with Actual Exports above `floor_usd`."""
    big = df.filter(pl.col("value_usd") > floor_usd)
    r = np.log1p(big["value_usd"].to_numpy()) - np.log1p(big[col].to_numpy())
    p10, p50, p90 = np.percentile(r, [10, 50, 90])
    return {"n": big.height, "p10": p10, "median": p50, "p90": p90, "spread": p90 - p10}


def fit_check(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Folds 0–2 to fit, folds 3–4 to check."""
    return df.filter(pl.col("fold") <= 2), df.filter(pl.col("fold") >= 3)


# --- Fit/check harness for post-hoc adjustments -------------------------------------------------
# An adjustment adds `beta * offset` to log Export Potential. Offsets are learned from UK pairs of
# *other* folds only (pairs are held out together, as ADR 0001 holds them out), so no pair ever sees
# its own Actual Exports. beta is chosen by leave-one-fold-out inside folds 0–2, then checked on 3–4.

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def apply_offsets(target: pl.DataFrame, source: pl.DataFrame, offsets_fn, beta: float, base: str = "winner") -> pl.DataFrame:
    off = offsets_fn(source)
    keys = [c for c in off.columns if c != "offset"]
    out = target.join(off, on=keys, how="left").with_columns(pl.col("offset").fill_null(0.0))
    return out.with_columns(adj=(pl.col(base).clip(1.0).log() + beta * pl.col("offset")).exp())


def choose_beta(df: pl.DataFrame, offsets_fn, base: str = "winner", betas=BETAS) -> tuple[float, dict]:
    """beta maximising the mean, over folds 0–2, of the measure on one fold with offsets from the other two."""
    scores = {}
    for b in betas:
        vals = []
        for k in (0, 1, 2):
            src = df.filter(pl.col("fold").is_in([j for j in (0, 1, 2) if j != k]))
            tgt = df.filter(pl.col("fold") == k)
            vals.append(measure(apply_offsets(tgt, src, offsets_fn, b, base), "adj"))
        scores[b] = float(np.mean(vals))
    return max(scores, key=scores.get), scores


def fit_check_test(df: pl.DataFrame, offsets_fn, base: str = "winner", label: str = "", betas=BETAS) -> dict:
    """beta from folds 0–2; check on folds 3–4 (offsets from 0–2); and the cross-fitted five-fold pooled figure."""
    beta, scores = choose_beta(df, offsets_fn, base, betas)
    fit, check = fit_check(df)
    adj = apply_offsets(check, fit, offsets_fn, beta, base)
    base_check = measure(check, base)
    parts = [apply_offsets(df.filter(pl.col("fold") == k), df.filter(pl.col("fold") != k), offsets_fn, beta, base)
             for k in range(5)]
    pooled = pl.concat(parts)
    res = {
        "label": label, "beta": beta, "check_base": base_check, "check_adj": measure(adj, "adj"),
        "check_gain": measure(adj, "adj") - base_check,
        "pooled_base": measure(df, base), "pooled_adj": measure(pooled, "adj"),
        "pooled_gain": measure(pooled, "adj") - measure(df, base),
        "spread_base": dollar_spread(df, base)["spread"], "spread_adj": dollar_spread(pooled, "adj")["spread"],
        "median_base": dollar_spread(df, base)["median"], "median_adj": dollar_spread(pooled, "adj")["median"],
        "beta_scores": {k: round(v, 4) for k, v in scores.items()},
    }
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    return res
