"""Does the leave-one-out penalty on UK supply hurt the model itself? Re-predict (CPU, saved fold models) with
supply features made constant within a Product inside each fold (the UK's exports to Markets outside the row's fold)."""
import json

import polars as pl
from common import OUT, dollar_spread, fit_check, measure
from cpu_models import predict

from exportspredictor.prediction import blend

df = pl.read_parquet(OUT / "frame_ranked.parquet").join(
    pl.read_parquet(OUT / "naive_fold.parquet"), on=["hs6", "importer"])
# rca_loo = (supply_loo / UK total exports loo) / (world_p / world): rescale by the supply ratio, keeping the rest
ratio = (pl.col("supply_fold") + 1) / (pl.col("log_supply_loo").cast(pl.Float64).exp())
pert = df.with_columns(
    log_supply_loo=(pl.col("supply_fold") + 1).log().cast(pl.Float32),
    rca_loo=(pl.col("rca_loo") * ratio).cast(pl.Float32),
)
df = df.with_columns(v2_fold=pl.Series(predict(pert)))
df = df.with_columns(blend_fold_model=pl.Series(blend(df["v2_fold"].to_numpy(), df["naive_usd"].to_numpy(), 0.3)),
                     blend_fold_both=pl.Series(blend(df["v2_fold"].to_numpy(), df["naive_fold"].to_numpy(), 0.3)))
fit, check = fit_check(df)
res = {}
for c in ["v2", "v2_fold", "winner", "blend_fold_model", "blend_fold_both"]:
    res[c] = {"pooled": measure(df, c), "fit": measure(fit, c), "check": measure(check, c), "spread": dollar_spread(df, c)["spread"]}
    print(c, {k: round(float(v), 4) for k, v in res[c].items()})
(OUT / "s07_model_supply.json").write_text(json.dumps(res, indent=1, default=float))
