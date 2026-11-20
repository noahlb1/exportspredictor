"""The affinity offset's beta curve, and the affinity combined with the UK-free demand terms (fit 0–2, check 3–4)."""
import json

import polars as pl
from common import DATA, OUT, apply_offsets, dollar_spread, fit_check, measure
from s04_market_affinity import nz_offset

from exportspredictor.loop.selection import measures

df = pl.read_parquet(OUT / "frame_ranked21.parquet")
d4 = None
from exportspredictor.features import load_year_flows

cmap = pl.read_parquet(DATA / "derived" / "country_map.parquet").select("baci_iso3", "iso3").unique()
flows = load_year_flows(2022, dict(cmap.iter_rows())).filter((pl.col("exporter") != "GBR") & (pl.col("value_usd") > 0)).with_columns(
    pl.col("hs6").cast(pl.String)).with_columns(hs4=pl.col("hs6").str.slice(0, 4))
df = df.join(flows.group_by("importer", "hs4").agg(d4=pl.col("value_usd").sum()), on=["importer", "hs4"], how="left").with_columns(
    sib=(pl.col("d4").fill_null(0.0) - (pl.col("log_demand_loo").cast(pl.Float64).exp() - 1)).clip(0))
df = df.with_columns(lsib=(pl.col("sib") + 1).log(), lgdp=pl.col("log_gdp_d").fill_null(pl.col("log_gdp_d").min()))
# centre the extra terms within each Product, so a Product's dollar level is unchanged on average
df = df.with_columns((pl.col(c) - pl.col(c).mean().over("hs6")).alias(c) for c in ["lsib", "lgdp"])


def cross_fitted(frame, beta):
    """Every fold scored with the any-Flow Market offset learned from the other folds (or, for checks, from 0–2)."""
    return pl.concat(apply_offsets(frame.filter(pl.col("fold") == k), frame.filter(pl.col("fold") != k), nz_offset(["importer"]), beta)
                     for k in range(5))


res = {"beta_curve": []}
fit, check = fit_check(df)
for beta in (0.0, 1.0, 2.0, 3.0, 4.0):
    c = apply_offsets(check, fit, nz_offset(["importer"]), beta)
    p = cross_fitted(df, beta)
    m = measures(p, None, "adj")
    row = {"beta": beta, "check": measure(c, "adj"), "pooled": m["selection_measure"], "non_zero": m["non_zero_spearman"],
           "zero_vs_non_zero": m["zero_vs_non_zero_spearman"], "value_weighted": m["value_weighted_spearman"],
           "top_ten": m["top_ten_share"], **{f"dollar_{k}": v for k, v in dollar_spread(p, "adj").items()}}
    res["beta_curve"].append(row); print({k: round(float(v), 4) for k, v in row.items()}, flush=True)

# joint: affinity (beta chosen on fit folds' leave-one-fold-out, = 4) plus a demand term with gamma chosen on folds 0–2
fit_aff = pl.concat(apply_offsets(fit.filter(pl.col("fold") == k), fit.filter(pl.col("fold") != k), nz_offset(["importer"]), 4.0) for k in (0, 1, 2))
chk_aff = apply_offsets(check, fit, nz_offset(["importer"]), 4.0)
all_aff = cross_fitted(df, 4.0)
for term in ["lsib", "lgdp"]:
    sc = {g: measure(fit_aff.with_columns(s=pl.col("adj").log() + g * pl.col(term)), "s") for g in (0.0, 0.1, 0.2, 0.3, 0.5)}
    g = max(sc, key=sc.get)
    row = {"term": term, "gamma": g, "check_aff": measure(chk_aff, "adj"),
           "check": measure(chk_aff.with_columns(s=pl.col("adj").log() + g * pl.col(term)), "s"),
           "pooled": measure(all_aff.with_columns(s=pl.col("adj").log() + g * pl.col(term)), "s"),
           "spread": dollar_spread(all_aff.with_columns(s=(pl.col("adj").log() + g * pl.col(term)).exp()), "s")["spread"]}
    res[f"affinity_plus_{term}"] = row; print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
# the demand terms alone, centred (so their dollar effect is fair)
for term in ["lsib", "lgdp"]:
    sc = {g: measure(fit.with_columns(s=pl.col("winner").log() + g * pl.col(term)), "s") for g in (0.0, 0.1, 0.2, 0.3, 0.5)}
    g = max(sc, key=sc.get)
    full = df.with_columns(s=(pl.col("winner").log() + g * pl.col(term)).exp())
    row = {"term": term, "gamma": g, "check": measure(check.with_columns(s=pl.col("winner").log() + g * pl.col(term)), "s"),
           "pooled": measure(full, "s"), "spread": dollar_spread(full, "s")["spread"]}
    res[f"{term}_centred"] = row; print(row, flush=True)
(OUT / "s14_combined.json").write_text(json.dumps(res, indent=1, default=float))
