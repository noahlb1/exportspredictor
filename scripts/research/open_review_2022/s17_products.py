"""Candidate Flagged Products found in the case review: their 2022 concentration and ranking, with training-year volatility."""
import polars as pl
from common import OUT

pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_rows(30)
df = pl.read_parquet(OUT / "frame_ranked21.parquet")
pv = pl.read_parquet(OUT / "product_volatility.parquet").select("hs6", "sd", "top")
CANDIDATES = ["490700", "284390", "120190", "880230", "880240", "890520", "890190", "930190"]
t = df.filter(pl.col("hs6").is_in(CANDIDATES)).group_by("hs6", "description").agg(
    uk_m=pl.col("value_usd").sum() / 1e6, markets=(pl.col("value_usd") > 0).sum(),
    top_market=pl.col("importer").sort_by("value_usd").last(), top_share=pl.col("value_usd").max() / pl.col("value_usd").sum(),
    rho=pl.corr("winner", "value_usd", method="spearman")).join(pv, on="hs6", how="left").sort("uk_m", descending=True)
print(t)
print("volatility quintile cut-offs (sd of yearly log change of UK exports, 2012-2021):", pv["sd"].quantile(0.8))
