"""UK-dominated Markets look small to leave-one-out demand. Test UK-free demand proxies as an extra blend term:
the Market's imports from other Exporters of the Product's HS4 siblings, and its imports of the Product in its region."""
import json

import polars as pl
from common import DATA, OUT, dollar_spread, fit_check, measure

from exportspredictor.features import load_year_flows

df = pl.read_parquet(OUT / "frame_ranked21.parquet")
cmap = pl.read_parquet(DATA / "derived" / "country_map.parquet").select("baci_iso3", "iso3").unique()
flows = load_year_flows(2022, dict(cmap.iter_rows())).filter((pl.col("exporter") != "GBR") & (pl.col("value_usd") > 0)).with_columns(
    pl.col("hs6").cast(pl.String))
flows = flows.with_columns(hs4=pl.col("hs6").str.slice(0, 4))
d6 = flows.group_by("importer", "hs6").agg(d6=pl.col("value_usd").sum())
d4 = flows.group_by("importer", "hs4").agg(d4=pl.col("value_usd").sum())
df = df.join(d6, on=["importer", "hs6"], how="left").join(d4, on=["importer", "hs4"], how="left").with_columns(
    pl.col("d6", "d4").fill_null(0.0)).with_columns(sib=(pl.col("d4") - pl.col("d6")).clip(0))
# sibling demand relative to the Product's sibling demand across Markets, so a Product-level constant cancels
df = df.with_columns(lsib=(pl.col("sib") + 1).log(), lgdp=pl.col("log_gdp_d").fill_null(pl.col("log_gdp_d").min()))
print("check: d6 == exp(log_demand_loo)-1, median rel err",
      df.select(((pl.col("d6") - (pl.col("log_demand_loo").cast(pl.Float64).exp() - 1)).abs() / pl.col("d6").clip(1)).median()).item())
fit, check = fit_check(df)
res = {}
for term in ["lsib", "lgdp"]:
    sc = {}
    for g in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        sc[g] = measure(fit.with_columns(s=pl.col("winner").log() + g * pl.col(term)), "s")
    g = max(sc, key=sc.get)
    full = df.with_columns(s=(pl.col("winner").log() + g * pl.col(term)).exp())
    c = check.with_columns(s=pl.col("winner").log() + g * pl.col(term))
    res[term] = {"gamma": g, "fit_scores": sc, "check_base": measure(check, "winner"), "check": measure(c, "s"),
                 "pooled": measure(full, "s"), "spread": dollar_spread(full, "s")["spread"]}
    print(term, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[term].items()})
(OUT / "s13_demand.json").write_text(json.dumps(res, indent=1, default=float))
