"""The any-Flow Market offset: wider beta grid, Sector detail, where the gain comes from."""
import json

import polars as pl
from common import OUT, apply_offsets, fit_check_test, measure
from s04_market_affinity import nz_offset

df = pl.read_parquet(OUT / "frame_ranked.parquet")
WIDE = (0.0, 1, 2, 3, 4, 5, 6, 8)
res = [fit_check_test(df, nz_offset(["importer"]), label="any-Flow offset by Market (wide)", betas=WIDE)]
res.append(fit_check_test(df, nz_offset(["importer", "hs2"], 50), label="any-Flow offset by Market x Sector", betas=WIDE))
(OUT / "s05_anyflow.json").write_text(json.dumps(res, indent=1, default=float))

# where does the gain come from? cross-fitted five-fold scores at the chosen beta, measured on Market subsets
beta = res[0]["beta"]
parts = [apply_offsets(df.filter(pl.col("fold") == k), df.filter(pl.col("fold") != k), nz_offset(["importer"]), beta)
         for k in range(5)]
pooled = pl.concat(parts)
pooled.select("hs6", "importer", "fold", "offset", "adj").write_parquet(OUT / "anyflow_scores.parquet")
imports = pooled.group_by("importer").agg(dem=(pl.col("log_demand_loo").exp() - 1).sum()).sort("dem", descending=True)
top = imports["importer"].to_list()
for n in (30, 60, 100, 150, len(top)):
    sub = pooled.filter(pl.col("importer").is_in(top[:n]))
    print(f"top {n} Markets by imports: base {measure(sub, 'winner'):.4f} adj {measure(sub, 'adj'):.4f} gain {measure(sub, 'adj') - measure(sub, 'winner'):+.4f}")
print(pooled.group_by("importer").agg(o=pl.col("offset").first()).join(
    pooled.select("importer", "market").unique(), on="importer").sort("o").head(12))
print(pooled.group_by("importer").agg(o=pl.col("offset").first()).join(
    pooled.select("importer", "market").unique(), on="importer").sort("o").tail(12))
