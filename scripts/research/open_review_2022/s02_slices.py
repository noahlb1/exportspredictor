"""Failure review: where the served blend's 2022 within-Product ranking goes wrong."""
import polars as pl
from common import OUT, frame, per_product

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
df = frame()
# within-Product percentile ranks (average ties); d > 0: the pair is ranked above where its Actual Exports put it
df = df.with_columns(
    ra=(pl.col("value_usd").rank("average").over("hs6") - 1) / (pl.len().over("hs6") - 1),
    rp=(pl.col("winner").rank("average").over("hs6") - 1) / (pl.len().over("hs6") - 1),
    n_markets=pl.len().over("hs6"),
    uk_nonzero=(pl.col("value_usd") > 0).sum().over("hs6"),
).with_columns(d=pl.col("rp") - pl.col("ra"))
df.write_parquet(OUT / "frame_ranked.parquet")

pp = per_product(df, "winner").join(per_product(df, "naive_usd").select("hs6", rho_naive="rho"), on="hs6")
pp = pp.join(per_product(df, "v2").select("hs6", rho_v2="rho"), on="hs6")
pp = pp.with_columns(hs2=pl.col("hs6").str.slice(0, 2), share_nonzero=pl.col("nonzero") / pl.col("n"))
print("ranked products", pp.height, "mean rho", pp["rho"].mean())
print(pp["rho"].describe())

print("\n== by share of Markets the UK exports to (bins)")
pp = pp.with_columns(nz_bin=pl.col("share_nonzero").cut([0.02, 0.05, 0.1, 0.2, 0.4, 0.6], left_closed=True))
print(pp.group_by("nz_bin").agg(products=pl.len(), rho=pl.col("rho").mean(), rho_naive=pl.col("rho_naive").mean(),
      rho_v2=pl.col("rho_v2").mean(), uk_bn=pl.col("uk_usd").sum() / 1e9).sort("nz_bin"))

print("\n== by UK export value of the Product (bins)")
pp = pp.with_columns(val_bin=pl.col("uk_usd").cut([1e5, 1e6, 1e7, 1e8, 1e9], left_closed=True))
print(pp.group_by("val_bin").agg(products=pl.len(), rho=pl.col("rho").mean(), rho_naive=pl.col("rho_naive").mean(),
      share_nz=pl.col("share_nonzero").mean()).sort("val_bin"))

print("\n== worst and best Sectors (HS2) by mean rho, with gain over naive")
sec = pp.group_by("hs2").agg(products=pl.len(), rho=pl.col("rho").mean(), gain_vs_naive=(pl.col("rho") - pl.col("rho_naive")).mean(),
      share_nz=pl.col("share_nonzero").mean(), uk_bn=pl.col("uk_usd").sum() / 1e9)
print(sec.sort("rho").head(15)); print(sec.sort("rho").tail(10))
print("sector-level corr(rho, share_nz):", sec.select(pl.corr("rho", "share_nz")).item())
print("product-level corr(rho, share_nz):", pp.select(pl.corr("rho", "share_nonzero", method="spearman")).item())

print("\n== Markets: mean over Products of d (over-ranked +), and contribution")
mk = df.group_by("importer", "market", "region").agg(
    pairs=pl.len(), d=pl.col("d").mean(), uk_bn=pl.col("value_usd").sum() / 1e9,
    nz_share=(pl.col("value_usd") > 0).mean(), ep_nz_share=(pl.col("rp") > 0.5).mean(),
    sq=(pl.col("d") ** 2).sum())
print("most UNDER-ranked Markets (d<0):"); print(mk.sort("d").head(25))
print("most OVER-ranked Markets (d>0):"); print(mk.sort("d", descending=True).head(25))
print("\n== by region"); print(df.group_by("region").agg(pairs=pl.len(), d=pl.col("d").mean(), absd=pl.col("d").abs().mean()).sort("d"))
mk.write_parquet(OUT / "markets_d.parquet"); pp.write_parquet(OUT / "products_rho.parquet")
