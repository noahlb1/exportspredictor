"""Idea: a UK×Market affinity learned from the UK's other Products into the same Market (cross-fitted by fold)."""
import json

import polars as pl
from common import OUT, fit_check_test

SHRINK = 200  # pairs: offsets of Markets with few pairs are pulled to 0


def rank_offset(keys, shrink=SHRINK):
    """Mean within-Product rank residual (Actual rank − Export Potential rank) of the source pairs, shrunk to 0."""
    def fn(src):
        return src.group_by(keys).agg(s=(pl.col("ra") - pl.col("rp")).sum(), n=pl.len()).select(
            *keys, offset=pl.col("s") / (pl.col("n") + shrink))
    return fn


def dollar_offset(keys, shrink_usd=1e6):
    """log of (Actual + c) / (Export Potential + c) summed over the source pairs."""
    def fn(src):
        return src.group_by(keys).agg(a=pl.col("value_usd").sum(), p=pl.col("winner").sum()).select(
            *keys, offset=((pl.col("a") + shrink_usd) / (pl.col("p") + shrink_usd)).log())
    return fn


def nz_offset(keys, shrink=SHRINK):
    """log-odds of the UK exporting at all to the Market, observed against expected from the rank bins."""
    def fn(src):
        return src.group_by(keys).agg(o=(pl.col("value_usd") > 0).sum(), n=pl.len(), e=pl.col("rp").sum()).select(
            *keys, offset=((pl.col("o") + 5) / (pl.col("n") + 10)).log() - ((pl.col("e") + 5) / (pl.col("n") + 10)).log())
    return fn



def main():
    df = pl.read_parquet(OUT / "frame_ranked.parquet")
    res = []
    RB = (0.0, 1, 2, 4, 6, 8, 12, 16)
    res.append(fit_check_test(df, rank_offset(["importer"]), label="rank offset by Market", betas=RB))
    res.append(fit_check_test(df, rank_offset(["importer", "hs2"], 50), label="rank offset by Market x Sector", betas=RB))
    res.append(fit_check_test(df, dollar_offset(["importer"]), label="dollar offset by Market"))
    res.append(fit_check_test(df, dollar_offset(["importer", "hs2"]), label="dollar offset by Market x Sector"))
    res.append(fit_check_test(df, nz_offset(["importer"]), label="any-Flow offset by Market"))
    (OUT / "s04_market_affinity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
