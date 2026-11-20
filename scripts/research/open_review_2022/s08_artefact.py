"""Qualitative review of the served artefact as an Advisor sees it (Baseline-year tables; no metric, no 2023 rows)."""
import polars as pl
from common import latest_artefact

pl.Config.set_tbl_rows(45); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(38); pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
art = latest_artefact()
print("artefact", art.name)
b = pl.read_parquet(art / "baseline.parquet")
p = pl.read_parquet(art / "products.parquet").select("hs6", "description")
m = pl.read_parquet(art / "markets.parquet").select("iso3", "name")
b = b.join(p, on="hs6", how="left").join(m, on="iso3", how="left").with_columns(
    pl.col("export_potential_usd", "actual_exports_usd", "export_gap_usd", "blocked_opportunity_value_usd") / 1e6)
cols = ["hs6", "description", "name", "export_potential_usd", "actual_exports_usd", "export_gap_usd", "opportunity_type", "reexport_hub", "flagged_product"]
opp = b.filter(pl.col("export_gap_usd") > 0).sort("export_gap_usd", descending=True)
print("Top 40 Opportunities by Export Gap ($m):"); print(opp.select(cols).head(40))
top1000 = opp.head(1000)
print("Top-1000 Opportunities by Market:"); print(top1000.group_by("name").agg(n=pl.len(), gap=pl.col("export_gap_usd").sum()).sort("n", descending=True).head(25))
print("Top-1000 by type:"); print(top1000.group_by("opportunity_type").len())
print("Top-1000 by Sector:"); print(top1000.group_by(pl.col("hs6").str.slice(0, 2)).agg(n=pl.len(), gap=pl.col("export_gap_usd").sum()).sort("n", descending=True).head(12))
SANCTIONED = ["RUS", "BLR", "IRN", "PRK", "SYR", "VEN", "CUB", "AFG", "MMR"]
print("Sanctioned/conflict Markets among the top 1000 / 10000:",
      top1000.filter(pl.col("iso3").is_in(SANCTIONED)).height, opp.head(10000).filter(pl.col("iso3").is_in(SANCTIONED)).height)
print(opp.filter(pl.col("iso3").is_in(SANCTIONED)).select(cols).head(15))
blocked = b.sort("blocked_opportunity_value_usd", descending=True)
print("Top 30 Blocked Opportunities ($m):")
print(blocked.select("hs6", "description", "name", "export_potential_usd", "actual_exports_usd", "blocked_opportunity_value_usd").head(30))
print("Blocked top-500 by Market:"); print(blocked.head(500).group_by("name").agg(n=pl.len(), v=pl.col("blocked_opportunity_value_usd").sum()).sort("v", descending=True).head(20))
print("Relative Export Gap top (EP>$1m):")
print(opp.filter(pl.col("export_potential_usd") > 1).sort(["relative_export_gap", "export_gap_usd"], descending=True).select(cols).head(25))
