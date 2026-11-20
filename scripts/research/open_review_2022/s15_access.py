"""Markets the model cannot know are closed: sanctions (2022 onwards) and alcohol prohibition. 2022 figures and the served view."""
import polars as pl
from common import OUT, latest_artefact, measure

pl.Config.set_tbl_rows(30); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(30)
df = pl.read_parquet(OUT / "frame_ranked21.parquet").with_columns(rank_ep=pl.col("winner").rank("ordinal", descending=True).over("hs6"))
SANCTIONED = ["RUS", "BLR"]  # broad UK export bans from 2022
PROHIBITION = ["SAU", "KWT", "LBY", "IRN"]  # alcohol import bans
ALCOHOL = ["220300", "220421", "220429", "220830", "220860", "220820", "220850", "220870", "220890"]
print("2022 Selection Measure, all Markets", round(measure(df, "winner"), 4), "without RUS, BLR:",
      round(measure(df.filter(~pl.col("importer").is_in(SANCTIONED)), "winner"), 4))
r = df.filter(pl.col("importer").is_in(SANCTIONED)).group_by("importer").agg(
    uk_2021_m=pl.col("value_2021").sum() / 1e6, uk_2022_m=pl.col("value_usd").sum() / 1e6, ep_2022_m=pl.col("winner").sum() / 1e6,
    products_2021=(pl.col("value_2021") > 0).sum(), products_2022=(pl.col("value_usd") > 0).sum(), top10_ranks=(pl.col("rank_ep") <= 10).sum())
print(r)
a = df.filter(pl.col("hs6").is_in(ALCOHOL) & pl.col("importer").is_in(PROHIBITION)).select(
    "hs6", "importer", "value_usd", "winner", "rank_ep", "n_markets")
print("alcohol in prohibition Markets, 2022 (rank of Export Potential among the Product's Markets):"); print(a.sort("rank_ep"))
art = latest_artefact()
b = pl.read_parquet(art / "baseline.parquet").with_columns(rank_gap=pl.col("export_gap_usd").rank("ordinal", descending=True))
print("served artefact", art.name)
print(b.filter(pl.col("hs6").is_in(ALCOHOL) & pl.col("iso3").is_in(PROHIBITION)).select(
    "hs6", "iso3", "export_potential_usd", "actual_exports_usd", "export_gap_usd", "opportunity_type", "rank_gap", "blocked_opportunity_value_usd").sort("rank_gap"))
s = b.filter(pl.col("iso3").is_in(SANCTIONED) & (pl.col("export_gap_usd") > 0))
print("served Opportunities in RUS/BLR:", s.height, "total gap $m", round(s["export_gap_usd"].sum() / 1e6), "in top 1000:", s.filter(pl.col("rank_gap") <= 1000).height,
      "top 10000:", s.filter(pl.col("rank_gap") <= 10000).height)
