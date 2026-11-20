"""Further checks: diagnostics of the affinity offset, a breadth-dependent naive weight, fold alignment,
the naive-zero cliff, and calibration of Export Potential's level (2022 only)."""
import json

import numpy as np
import polars as pl
from common import OUT, apply_offsets, dollar_spread, fit_check, measure
from s04_market_affinity import nz_offset

from exportspredictor.loop.selection import measures

pl.Config.set_tbl_rows(30); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_width_chars(200)
df = pl.read_parquet(OUT / "frame_ranked21.parquet")
res = {}

# (a) diagnostics of the any-Flow Market offset, cross-fitted, beta 4
parts = [apply_offsets(df.filter(pl.col("fold") == k), df.filter(pl.col("fold") != k), nz_offset(["importer"]), 4.0) for k in range(5)]
aff = pl.concat(parts)
res["diagnostics"] = {c: measures(aff, None, c) for c in ["winner", "adj"]}
for c in ["winner", "adj"]:
    print(c, {k: round(v, 4) for k, v in res["diagnostics"][c].items() if isinstance(v, float)})

# (b) naive weight by the Product's UK breadth in 2021 (a Product-level constant: no pair's own 2022 Flow)
df = df.with_columns(b21=((pl.col("value_2021") > 0).sum().over("hs6") / pl.len().over("hs6")))
df = df.with_columns(bbin=pl.col("b21").cut([0.05, 0.1, 0.2, 0.4], labels=["<5%", "5-10%", "10-20%", "20-40%", ">=40%"]))
lm = df["v2"].clip(1).log().to_numpy(); ln = df["naive_usd"].clip(1).log().to_numpy()
fit, check = fit_check(df)
best_w = {}
for b in ["<5%", "5-10%", "10-20%", "20-40%", ">=40%"]:
    f = fit.filter(pl.col("bbin") == b)
    scores = {w: measure(f.with_columns(s=(1 - w) * pl.col("v2").clip(1).log() + w * pl.col("naive_usd").clip(1).log()), "s")
              for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    best_w[b] = max(scores, key=scores.get)
    print("breadth", b, "products", f["hs6"].n_unique(), "best w", best_w[b], {k: round(v, 4) for k, v in scores.items()})
wexpr = pl.col("bbin").cast(pl.String).replace_strict(best_w, return_dtype=pl.Float64)
d2 = df.with_columns(w=wexpr).with_columns(s=((1 - pl.col("w")) * pl.col("v2").clip(1).log() + pl.col("w") * pl.col("naive_usd").clip(1).log()).exp())
f2, c2 = fit_check(d2)
res["breadth_weight"] = {"weights": best_w, "check_base": measure(check, "winner"), "check": measure(c2, "s"),
                         "pooled": measure(d2, "s"), "spread": dollar_spread(d2, "s")["spread"]}
print("breadth weight:", res["breadth_weight"])

# (c) label-free fold alignment: shift each fold model's log level so its mean log(model/naive) on its pairs matches all folds'
d3 = df.with_columns(r=(pl.col("v2").clip(1).log() - pl.col("naive_usd").clip(1).log()))
glob = d3.filter(pl.col("naive_usd") > 1)["r"].mean()
fold_mean = d3.filter(pl.col("naive_usd") > 1).group_by("fold").agg(fm=pl.col("r").mean())
print("fold mean log(model/naive):", fold_mean.sort("fold").to_dicts(), "global", glob)
d3 = d3.join(fold_mean, on="fold").with_columns(
    s=(0.7 * (pl.col("v2").clip(1).log() - pl.col("fm") + glob) + 0.3 * pl.col("naive_usd").clip(1).log()).exp())
res["fold_alignment"] = {"pooled_base": measure(df, "winner"), "pooled": measure(d3, "s")}
print("fold alignment:", res["fold_alignment"])

# (d) the naive-zero cliff: the Market imports none of the Product from anyone else
z = df.filter(pl.col("naive_usd") < 1)
print("pairs with naive < $1:", z.height, "of which UK exports:", z.filter(pl.col("value_usd") > 0).height,
      "UK $m:", round(z["value_usd"].sum() / 1e6, 1))
zz = df.filter(pl.col("log_demand_loo") == 0)
print("pairs where the Market imports nothing from anyone else:", zz.height, "UK exports in", zz.filter(pl.col("value_usd") > 0).height,
      "UK $m", round(zz["value_usd"].sum() / 1e6, 1))
res["naive_zero"] = {"pairs": z.height, "uk_nonzero": z.filter(pl.col("value_usd") > 0).height}

# (e) level calibration: Actual / Export Potential by Export Potential decile (conditioned on the prediction, not the outcome)
cal = df.with_columns(dec=pl.col("winner").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True)).group_by("dec").agg(
    ep_min=pl.col("winner").min(), ep_max=pl.col("winner").max(), pairs=pl.len(),
    actual_over_ep=pl.col("value_usd").sum() / pl.col("winner").sum(), share_nonzero=(pl.col("value_usd") > 0).mean()).sort("dec")
print(cal)
top = df.filter(pl.col("winner") > 1e7)
print("EP > $10m: pairs", top.height, "sum actual / sum EP", top["value_usd"].sum() / top["winner"].sum(),
      "median log(1+a)-log(1+EP)", float(np.median(np.log1p(top["value_usd"].to_numpy()) - np.log1p(top["winner"].to_numpy()))))
res["calibration"] = cal.to_dicts()
(OUT / "s10_more.json").write_text(json.dumps(res, indent=1, default=float))
