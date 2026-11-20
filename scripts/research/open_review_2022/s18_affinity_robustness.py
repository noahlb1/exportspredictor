"""Robustness of the any-Flow Market offset: each check fold on its own, and the lagged (2021) breadth variant."""
import json

import polars as pl
from common import OUT, apply_offsets, measure
from s04_market_affinity import nz_offset
from s09_affinity_variants import breadth_offset

df = pl.read_parquet(OUT / "frame_ranked21.parquet")
fit = df.filter(pl.col("fold") <= 2)
res = []
for name, fn, beta in [("any-Flow offset, 2022", nz_offset(["importer"]), 1.0), ("any-Flow offset, 2022", nz_offset(["importer"]), 4.0),
                       ("breadth share, 2021 (lagged)", breadth_offset("value_2021", "log_demand_2021"), 4.0)]:
    for k in (3, 4):
        tgt = df.filter(pl.col("fold") == k)
        adj = apply_offsets(tgt, fit, fn, beta)
        row = {"variant": name, "beta": beta, "fold": k, "base": measure(tgt, "winner"), "adj": measure(adj, "adj")}
        row["gain"] = row["adj"] - row["base"]; res.append(row); print({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()})
(OUT / "s18_affinity_robustness.json").write_text(json.dumps(res, indent=1))
