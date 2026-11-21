"""Leave-one-out supply penalises the UK's own best Markets: size of the effect and a fold-level fix for the naive side."""
import json

import polars as pl
from common import OUT, dollar_spread, fit_check, measure

from exportspredictor.prediction import blend

df = pl.read_parquet(OUT / "frame_ranked.parquet")
supply = pl.col("log_supply_loo").cast(pl.Float64).exp() - 1
df = df.with_columns(supply_loo=supply, uk_total=pl.col("value_usd").sum().over("hs6"))
print("check supply_loo == UK total − own (median abs rel err):",
      df.select(((pl.col("supply_loo") - (pl.col("uk_total") - pl.col("value_usd"))).abs() / pl.col("uk_total").clip(1)).median()).item())
# the penalty: log(total) − log(supply_loo) = −log(1 − own share)
df = df.with_columns(own_share=pl.col("value_usd") / pl.col("uk_total").clip(1))
print("pairs with own share > 25%:", df.filter(pl.col("own_share") > 0.25).height,
      "> 50%:", df.filter(pl.col("own_share") > 0.5).height)
top = df.group_by("hs6").agg(top_share=pl.col("own_share").max())
print("Products whose top Market takes > 50% of UK exports:", top.filter(pl.col("top_share") > 0.5).height, "of", top.height)
print("within-Product Spearman(log_supply_loo, actual), mean over Products:",
      df.group_by("hs6").agg(r=pl.corr("log_supply_loo", "value_usd", method="spearman")).filter(pl.col("r").is_not_nan())["r"].mean())

# Fold-level supply: the UK's exports of the Product to Markets whose pair is not in the row's fold.
# It leaves out the target Market (and the rest of its fold), so it keeps the guardrail, and it is constant
# within a Product inside a fold.
fold_sum = df.group_by("hs6", "fold").agg(fs=pl.col("value_usd").sum())
df = df.join(fold_sum, on=["hs6", "fold"]).with_columns(supply_fold=pl.col("uk_total") - pl.col("fs"))
world_loo = (pl.col("supply_loo") * pl.col("log_demand_loo").cast(pl.Float64).exp().sub(1)) / pl.col("naive_usd")
# naive = supply_loo * demand_loo / world_loo  ->  world_loo recovered per row
df = df.with_columns(world_loo=pl.when(pl.col("naive_usd") > 0).then(world_loo))
df = df.with_columns(pl.col("world_loo").fill_null(strategy="max").over("hs6"))
df = df.with_columns(naive_fold=(pl.col("supply_fold") * (pl.col("log_demand_loo").cast(pl.Float64).exp() - 1) / pl.col("world_loo")).fill_null(0.0).fill_nan(0.0))
df = df.with_columns(
    blend_fold=pl.Series(blend(df["v2"].to_numpy(), df["naive_fold"].to_numpy(), 0.3)),
)
res = {}
fit, check = fit_check(df)
for c in ["naive_usd", "naive_fold", "winner", "blend_fold"]:
    res[c] = {"pooled": measure(df, c), "fit": measure(fit, c), "check": measure(check, c), **dollar_spread(df, c)}
    print(c, {k: round(float(v), 4) for k, v in res[c].items()})
# the naive weight re-chosen on fit folds for the fold-level naive side
for w in (0.2, 0.3, 0.4, 0.5):
    b = pl.Series(blend(df["v2"].to_numpy(), df["naive_fold"].to_numpy(), w))
    d2 = df.with_columns(b=b)
    f2, c2 = fit_check(d2)
    print("w", w, "fit", round(measure(f2, "b"), 4), "check", round(measure(c2, "b"), 4), "pooled", round(measure(d2, "b"), 4))
(OUT / "s06_loo_supply.json").write_text(json.dumps(res, indent=1, default=float))
df.select("hs6", "importer", "supply_fold", "naive_fold", "own_share").write_parquet(OUT / "naive_fold.parquet")
