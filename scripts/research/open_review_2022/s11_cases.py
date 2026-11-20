"""The biggest misses, case by case: large UK Flows ranked low, and confident leads with no Flow (2022)."""
import polars as pl
from common import OUT

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(34)
df = pl.read_parquet(OUT / "frame_ranked21.parquet").with_columns(
    rank_ep=pl.col("winner").rank("ordinal", descending=True).over("hs6"),
    rank_actual=pl.col("value_usd").rank("min", descending=True).over("hs6"),
    a_m=pl.col("value_usd") / 1e6, ep_m=pl.col("winner") / 1e6, naive_m=pl.col("naive_usd") / 1e6, v2_m=pl.col("v2") / 1e6,
    a21_m=pl.col("value_2021") / 1e6, share=pl.col("value_usd") / pl.col("value_usd").sum().over("hs6"))
cols = ["hs6", "description", "importer", "a_m", "a21_m", "ep_m", "v2_m", "naive_m", "rank_actual", "rank_ep", "share"]
print("Large UK Flows (> $20m) ranked outside the Product's top 20 Markets by Export Potential:")
miss = df.filter((pl.col("value_usd") > 2e7) & (pl.col("rank_ep") > 20)).sort("value_usd", descending=True)
print("count", miss.height, "of", df.filter(pl.col("value_usd") > 2e7).height); print(miss.select(cols).head(30))
print(miss.group_by("importer").len().sort("len", descending=True).head(12))
print("\nConfident leads with no UK Flow: Export Potential rank 1–3 in a Product the UK sends to >= 20 Markets:")
lead = df.filter((pl.col("value_usd") == 0) & (pl.col("rank_ep") <= 3) & (pl.col("uk_nonzero") >= 20)).sort("winner", descending=True)
print("count", lead.height); print(lead.select(cols).head(30))
print(lead.group_by("importer").len().sort("len", descending=True).head(15))
miss.select(cols).write_csv(OUT / "cases_missed_flows.csv"); lead.select(cols).write_csv(OUT / "cases_false_leads.csv")
