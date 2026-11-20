"""Reproduce the pooled 2022 Selection Measure (0.5635) and the per-fold figures."""
import numpy as np
import polars as pl
from common import dollar_spread, fit_check, frame, measure, served_blend

df = frame()
print("rows", df.height, "products", df["hs6"].n_unique())
df = df.with_columns(recomputed=pl.Series(served_blend(df)))
print("max |winner - recomputed| rel:", float(((df["winner"] - df["recomputed"]).abs() / df["winner"].clip(1)).max()))
for c in ["winner", "v2", "naive_usd"]:
    print(c, "pooled", round(measure(df, c), 4),
          "fold-mean", round(np.mean([measure(df.filter(pl.col("fold") == k), c) for k in range(5)]), 4))
fit, check = fit_check(df)
for c in ["winner", "v2", "naive_usd"]:
    print(c, "fit(0-2)", round(measure(fit, c), 4), "check(3-4)", round(measure(check, c), 4))
for c in ["winner", "v2", "naive_usd"]:
    print(c, dollar_spread(df, c))
print("null features:", df.select(pl.col("log_supply_loo").null_count(), pl.col("market").null_count()))
