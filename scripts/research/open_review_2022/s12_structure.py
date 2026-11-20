"""Structural patterns behind the misses: UK-dominated Market imports, agri-food and steel false leads, lumpy Products."""
import json

import polars as pl
from common import DATA, OUT

pl.Config.set_tbl_rows(30); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_width_chars(200)
df = pl.read_parquet(OUT / "frame_ranked21.parquet")
res = {}
print("totals 2022: UK actual $bn", round(df["value_usd"].sum() / 1e9, 1), "Export Potential $bn", round(df["winner"].sum() / 1e9, 1),
      "model alone", round(df["v2"].sum() / 1e9, 1), "naive", round(df["naive_usd"].sum() / 1e9, 1))

# (1) the UK's share of the Market's imports of the Product, non-zero pairs only
nz = df.filter(pl.col("value_usd") > 0).with_columns(
    uk_share=pl.col("value_usd") / (pl.col("value_usd") + pl.col("log_demand_loo").cast(pl.Float64).exp() - 1))
t = nz.with_columns(b=pl.col("uk_share").cut([0.05, 0.2, 0.5, 0.8], labels=["<5%", "5-20%", "20-50%", "50-80%", ">=80%"])).group_by("b").agg(
    pairs=pl.len(), uk_bn=pl.col("value_usd").sum() / 1e9, d=(pl.col("rp") - pl.col("ra")).mean(),
    log_actual_over_ep=(pl.col("value_usd").log() - pl.col("winner").log()).median()).sort("b")
print("UK share of the Market's imports (non-zero UK pairs):"); print(t)
res["uk_share"] = t.to_dicts()

# (2) where the confident leads with no Flow sit, by broad group
grp = pl.when(pl.col("hs2").cast(pl.Int32) <= 24).then(pl.lit("agri-food (01-24)")).when(pl.col("hs2").is_in(["72", "73"])).then(pl.lit("iron & steel (72-73)")).otherwise(pl.lit("other"))
lead = df.with_columns(rank_ep=pl.col("winner").rank("ordinal", descending=True).over("hs6")).filter(
    (pl.col("rank_ep") <= 10) & (pl.col("uk_nonzero") >= 20))
t2 = lead.with_columns(g=grp).group_by("g").agg(top10_pairs=pl.len(), no_flow_share=(pl.col("value_usd") == 0).mean()).sort("g")
print("Top-10 Markets by Export Potential, Products the UK sends to >= 20 Markets: share with no UK Flow"); print(t2)
res["lead_groups"] = t2.to_dicts()
big = ["USA", "CHN", "JPN", "KOR", "IND", "AUS", "CAN", "BRA", "MEX", "SAU"]
t3 = lead.filter(pl.col("importer").is_in(big)).with_columns(g=grp).group_by("g").agg(pairs=pl.len(), no_flow_share=(pl.col("value_usd") == 0).mean()).sort("g")
print("same, ten large non-EU Markets:"); print(t3)
res["lead_groups_big_non_eu"] = t3.to_dicts()
rho = df.group_by("hs6").agg(n=pl.len(), r=pl.corr("winner", "value_usd", method="spearman"), b=(pl.col("value_usd") > 0).mean()).filter(pl.col("r").is_not_nan())
rho = rho.with_columns(g=pl.when(pl.col("hs6").str.slice(0, 2).cast(pl.Int32) <= 24).then(pl.lit("agri-food")).when(pl.col("hs6").str.slice(0, 2).is_in(["72", "73"])).then(pl.lit("steel")).otherwise(pl.lit("other")))
print(rho.group_by("g").agg(products=pl.len(), rho=pl.col("r").mean(), breadth=pl.col("b").mean()))

# (3) lumpy Products: year-to-year volatility of the UK's total exports of the Product over the training years 2012–2021
files = [str(DATA / "derived" / "features_V202601" / f"uk_grid_{y}.parquet") for y in range(2012, 2022)]
hist = pl.scan_parquet(files).select(pl.col("hs6").cast(pl.String), "year", "value_usd").group_by("hs6", "year").agg(v=pl.col("value_usd").sum()).collect()
vol = hist.sort("year").group_by("hs6").agg(sd=pl.col("v").clip(1).log().diff().std(), top_market_share=pl.lit(None))
# concentration: mean over 2012–2021 of the UK's top Market share in the Product
conc = pl.scan_parquet(files).select(pl.col("hs6").cast(pl.String), "year", "value_usd").collect().with_columns(
    s=pl.col("value_usd") / pl.col("value_usd").sum().over("hs6", "year")).group_by("hs6", "year").agg(top=pl.col("s").max()).group_by("hs6").agg(top=pl.col("top").mean())
pv = rho.join(vol.drop("top_market_share"), on="hs6").join(conc, on="hs6")
pv = pv.with_columns(vb=pl.col("sd").qcut(5, labels=["q1 calm", "q2", "q3", "q4", "q5 lumpy"]))
print(pv.group_by("vb").agg(products=pl.len(), rho=pl.col("r").mean(), breadth=pl.col("b").mean(), top=pl.col("top").mean()).sort("vb"))
print("partial view: within breadth decile, corr(rho, volatility):",
      pv.with_columns(bd=pl.col("b").qcut(10, allow_duplicates=True)).group_by("bd").agg(c=pl.corr("r", "sd")).select(pl.col("c").mean()).item())
pv.write_parquet(OUT / "product_volatility.parquet")
(OUT / "s12_structure.json").write_text(json.dumps(res, indent=1, default=str))
