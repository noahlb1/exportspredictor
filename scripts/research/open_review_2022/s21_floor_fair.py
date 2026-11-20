"""Compare forms at their own best floor: is a gain information, or ties at the bottom? Floor chosen on folds 0–2, checked on 3–4."""
import json

import numpy as np
import polars as pl
from common import OUT, apply_offsets, fit_check, measure
from s04_market_affinity import nz_offset

from exportspredictor.loop.selection import measures

df = pl.read_parquet(OUT / "frame_ranked21.parquet")
fit, check = fit_check(df)
# the reach offset (beta 1 and 4), offsets for fit folds from the other fit folds, for check folds from folds 0–2
for beta in (1.0, 4.0):
    f = pl.concat(apply_offsets(fit.filter(pl.col("fold") == k), fit.filter(pl.col("fold") != k), nz_offset(["importer"]), beta) for k in (0, 1, 2))
    c = apply_offsets(check, fit, nz_offset(["importer"]), beta)
    fit = fit.join(f.select("hs6", "importer", pl.col("adj").alias(f"reach{int(beta)}")), on=["hs6", "importer"])
    check = check.join(c.select("hs6", "importer", pl.col("adj").alias(f"reach{int(beta)}")), on=["hs6", "importer"])
FLOORS = [1, 30, 100, 300, 1_000, 2_000, 3_000, 5_000, 10_000]
res = {}
for col in ["v2", "naive_usd", "winner", "reach1", "reach4"]:
    # a floor as a share of pairs would be fairer across forms whose levels differ: use quantiles of the column itself
    qs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    sc = {}
    for q in qs:
        thr = float(fit[col].quantile(q)) if q else 0.0
        sc[q] = (thr, measure(fit.with_columns(pl.col(col).clip(thr)), col))
    q = max(sc, key=lambda k: sc[k][1])
    thr = sc[q][0]
    chk = check.with_columns(pl.col(col).clip(thr))
    m = measures(chk, None, col)
    res[col] = {"check_raw": measure(check, col), "best_tied_share": q, "floor_usd": thr, "check_floored": m["selection_measure"],
                "non_zero": m["non_zero_spearman"], "zero_vs_non_zero": m["zero_vs_non_zero_spearman"]}
    print(col, {k: round(v, 4) for k, v in res[col].items()}, flush=True)
for a, b in [("winner", "v2"), ("winner", "naive_usd"), ("reach1", "winner"), ("reach4", "winner")]:
    print(f"{a} - {b}: raw {res[a]['check_raw'] - res[b]['check_raw']:+.4f}, each at its best floor {res[a]['check_floored'] - res[b]['check_floored']:+.4f}")
(OUT / "s21_floor_fair.json").write_text(json.dumps(res, indent=1, default=float))
_ = np
