"""Does the Selection Measure reward tying low Export Potentials? Floor the served blend at $X and re-measure (2022)."""
import json

import polars as pl
from common import OUT, apply_offsets, fit_check, measure
from s04_market_affinity import nz_offset

df = pl.read_parquet(OUT / "frame_ranked21.parquet")
fit, check = fit_check(df)
res = {}
for floor in (1, 10, 100, 1_000, 3_000, 10_000, 30_000):
    d = df.with_columns(f=pl.col("winner").clip(floor))
    f2, c2 = fit_check(d)
    res[floor] = {"pooled": measure(d, "f"), "fit": measure(f2, "f"), "check": measure(c2, "f"), "tied_share": float((d["f"] == floor).mean())}
    print(floor, {k: round(v, 4) for k, v in res[floor].items()}, flush=True)
reach = apply_offsets(check, fit, nz_offset(["importer"]), 4.0)
print("reach offset, beta 4, check: unfloored", round(measure(reach, "adj"), 4), "floored at $1",
      round(measure(reach.with_columns(pl.col("adj").clip(1)), "adj"), 4), "share below $1", round(float((reach["adj"] < 1).mean()), 3))
res["reach_check"] = {"raw": measure(reach, "adj"), "floor_1": measure(reach.with_columns(pl.col("adj").clip(1)), "adj")}
(OUT / "s20_floor.json").write_text(json.dumps(res, indent=1, default=float))
