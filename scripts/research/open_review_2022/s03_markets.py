"""Market-level misses: extensive margin (how many Products the UK sends) and intensive (dollars), robust to ties."""
import polars as pl
from common import OUT

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(22); pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
df = pl.read_parquet(OUT / "frame_ranked.parquet")
# P(UK Flow > 0) given the pair's within-Product rank of Export Potential (50 bins) and the Product's UK breadth (6 bins):
# the number of Products the UK "should" send to each Market if the ranking were right on average
df = df.with_columns(bb=(pl.col("uk_nonzero") / pl.col("n_markets")).cut([0.05, 0.1, 0.2, 0.4, 0.6]).to_physical(),
                     rpb=(pl.col("rp") * 50).floor().clip(0, 49), nz=(pl.col("value_usd") > 0).cast(pl.Float64))
df = df.with_columns(p_nz=pl.col("nz").mean().over("bb", "rpb"))
mk = df.group_by("importer", "market").agg(
    uk_products=pl.col("nz").sum(), expected=pl.col("p_nz").sum(),
    uk_musd=pl.col("value_usd").sum() / 1e6, ep_musd=pl.col("winner").sum() / 1e6,
    log_gdp=pl.col("log_gdp_d").first(), dist=pl.col("log_distance").first(), col=pl.col("colonial_tie").first(),
).with_columns(ext_ratio=pl.col("uk_products") / pl.col("expected"), dollar_ratio=pl.col("uk_musd") / pl.col("ep_musd"))
mk.write_parquet(OUT / "markets_margin.parquet")
show = mk.with_columns(pl.col(pl.Float64, pl.Float32).round(2))
print("Markets where the UK sends MORE Products than the ranking expects (>= 20 expected):")
print(show.filter(pl.col("expected") >= 20).sort("ext_ratio", descending=True).head(30))
print("Markets where the UK sends FEWER Products than expected:")
print(show.filter(pl.col("expected") >= 20).sort("ext_ratio").head(25))
print("Dollar ratio extremes (UK >= $50m):")
print(show.filter(pl.col("uk_musd") >= 50).sort("dollar_ratio").head(15))
print(show.filter(pl.col("uk_musd") >= 50).sort("dollar_ratio", descending=True).head(15))
print("Markets with no gravity data (GDP or distance missing):")
print(show.filter(pl.col("log_gdp").is_null() | pl.col("dist").is_null()))
