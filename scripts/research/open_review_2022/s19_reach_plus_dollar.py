"""The Market reach offset (any-Flow, beta 4) plus the dollar Market offset, beta of the second chosen on folds 0–2."""
import json

import polars as pl
from common import OUT, apply_offsets, dollar_spread, measure
from s04_market_affinity import dollar_offset, nz_offset

df = pl.read_parquet(OUT / "frame_ranked21.parquet")


def both(target, source, b2):
    a = apply_offsets(target, source, nz_offset(["importer"]), 4.0).drop("offset").rename({"adj": "reach"})
    return apply_offsets(a, source, dollar_offset(["importer"]), b2, base="reach")


fit = df.filter(pl.col("fold") <= 2)
scores = {}
for b2 in (0.0, 0.25, 0.5, 0.75, 1.0):
    scores[b2] = sum(measure(both(fit.filter(pl.col("fold") == k), fit.filter(pl.col("fold") != k), b2), "adj") for k in (0, 1, 2)) / 3
b2 = max(scores, key=scores.get)
check = df.filter(pl.col("fold") >= 3)
pooled = pl.concat(both(df.filter(pl.col("fold") == k), df.filter(pl.col("fold") != k), b2) for k in range(5))
res = {"beta_dollar": b2, "fit_scores": scores, "check_reach_only": measure(apply_offsets(check, fit, nz_offset(["importer"]), 4.0), "adj"),
       "check": measure(both(check, fit, b2), "adj"), "pooled": measure(pooled, "adj"), "dollar": dollar_spread(pooled, "adj")}
print(res)
(OUT / "s19_reach_plus_dollar.json").write_text(json.dumps(res, indent=1, default=float))
