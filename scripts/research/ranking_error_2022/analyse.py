"""Stage 2: where the blend's 2022 within-Product ranking fails, and what candidate features add.

Reads grid.parquet (stage 1, `build.py`). Prints every table the research note quotes.
"""
import json

import numpy as np
import polars as pl

from exportspredictor.paths import data_dir

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(60)
HERE = data_dir() / "derived" / "ranking_error_2022"
MIN_ROWS = 3
VALUE = pl.col("value_usd")  # the default ranking target: UK Actual Exports
THRESHOLD = 0.00887
g = pl.read_parquet(HERE / "grid.parquet").with_columns(
    blend=pl.col("winner"), lb=pl.col("winner").clip(1.0).log(), ln=pl.col("naive_usd").clip(1.0).log(),
    lv=pl.col("v2").clip(1.0).log(), la=pl.col("value_usd").log1p(), traded=(pl.col("value_usd") > 0),
)
g = g.with_columns(resid=pl.col("la") - pl.col("winner").log1p())
results: dict = {}


def within(df: pl.DataFrame, pred: str | pl.Expr, target: pl.Expr = VALUE, by="hs6") -> pl.DataFrame:
    p = pl.col(pred) if isinstance(pred, str) else pred
    return (
        df.with_columns(_p=p, _t=target).group_by(by)
        .agg(_n=pl.len(), _rho=pl.corr("_p", "_t", method="spearman"))
        .filter((pl.col("_n") >= MIN_ROWS) & pl.col("_rho").is_not_nan() & pl.col("_rho").is_not_null())
    )


def measure(df, pred, target=VALUE) -> float:
    return float(within(df, pred, target)["_rho"].mean())


# --- 0. Reproduce the confirmation's figures --------------------------------------------------
base = {name: measure(g, col) for name, col in [("blend", "winner"), ("v2", "v2"), ("naive", "naive_usd")]}
per_fold = {name: float(np.mean([measure(g.filter(pl.col("fold") == k), col) for k in range(5)]))
            for name, col in [("blend", "winner"), ("v2", "v2"), ("naive", "naive_usd")]}
print("pooled", base, "\nmean of folds", per_fold)
results["pooled"], results["per_fold_mean"] = base, per_fold

# --- 1. Where it fails ------------------------------------------------------------------------
nz = measure(g.filter("traded"), "winner")
zv = measure(g, "winner", pl.col("traded").cast(pl.Float64))
nz_naive = measure(g.filter("traded"), "naive_usd")
zv_naive = measure(g, "naive_usd", pl.col("traded").cast(pl.Float64))
# Oracle: Markets with UK trade first (presence known), then the blend's order.
oracle_presence = measure(g, pl.col("traded").cast(pl.Float64) * 1e3 + pl.col("lb"))
# Oracle: presence as the blend has it, but each traded Market's value known: the blend's ranks among
# untraded Markets kept, traded Markets ordered by actual value inside the slots the blend gave them.
slots = g.with_columns(_r=pl.col("lb").rank("ordinal").over("hs6"))
# Within each Product the k-th smallest traded value takes the k-th smallest slot the blend gave traded Markets.
traded = slots.filter("traded")
by_value = traded.sort("hs6", "value_usd").with_columns(_i=pl.int_range(pl.len()).over("hs6")).drop("_r")
by_slot = traded.sort("hs6", "_r").with_columns(_i=pl.int_range(pl.len()).over("hs6")).select("hs6", "_i", "_r")
reordered = by_value.join(by_slot, on=["hs6", "_i"]).drop("_i")
oracle_order = measure(
    pl.concat([slots.filter(~pl.col("traded")), reordered], how="diagonal"), pl.col("_r").cast(pl.Float64)
)
k = g.group_by("hs6").agg(_k=pl.col("traded").sum())
topk = (
    g.join(k, on="hs6").with_columns(_rank=pl.col("winner").rank("ordinal", descending=True).over("hs6"))
    .with_columns(_in=pl.col("_rank") <= pl.col("_k"))
)
precision = float(topk.filter("_in")["traded"].mean())
missed_value = float(topk.filter(~pl.col("_in"))["value_usd"].sum() / topk["value_usd"].sum())
phantom = topk.filter(pl.col("_in") & ~pl.col("traded"))
results["decomposition"] = {
    "non_zero": nz, "non_zero_naive": nz_naive, "presence": zv, "presence_naive": zv_naive,
    "oracle_presence": oracle_presence, "oracle_order_among_traded": oracle_order,
    "top_k_precision": precision, "missed_value_share": missed_value,
    "phantom_pairs": phantom.height, "phantom_potential_bn": float(phantom["winner"].sum() / 1e9),
    "missed_pairs": int(topk.filter(~pl.col("_in") & pl.col("traded")).height),
}
print(json.dumps(results["decomposition"], indent=1))

prod = (
    g.group_by("hs6").agg(
        n=pl.len(), k=pl.col("traded").sum(), uk_usd=pl.col("value_usd").sum(),
        rho=pl.corr("winner", "value_usd", method="spearman"),
        rho_naive=pl.corr("naive_usd", "value_usd", method="spearman"),
        sector=pl.col("sector").first(), description=pl.col("description").first(),
    ).filter(pl.col("rho").is_not_nan() & pl.col("rho").is_not_null())
)
by_k = (
    prod.with_columns(markets=pl.col("k").cut([2, 10, 30, 80], labels=["1-2", "3-10", "11-30", "31-80", "81+"]))
    .group_by("markets").agg(products=pl.len(), blend=pl.col("rho").mean(), naive=pl.col("rho_naive").mean(),
                             uk_bn=pl.col("uk_usd").sum() / 1e9).sort("markets")
)
print("By number of Markets the UK sells the Product to\n", by_k)
results["by_markets"] = by_k.to_dicts()
by_value = (
    prod.with_columns(size=pl.col("uk_usd").qcut(5, labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"]))
    .group_by("size").agg(products=pl.len(), blend=pl.col("rho").mean(), naive=pl.col("rho_naive").mean(),
                          median_markets=pl.col("k").median()).sort("size")
)
print("By UK export value of the Product (quintiles)\n", by_value)
results["by_value"] = by_value.to_dicts()
sector = (
    prod.group_by("sector").agg(products=pl.len(), blend=pl.col("rho").mean(), naive=pl.col("rho_naive").mean(),
                                uk_bn=pl.col("uk_usd").sum() / 1e9)
    .filter(pl.col("products") >= 15).with_columns(margin=pl.col("blend") - pl.col("naive")).sort("blend")
)
print("Weakest HS2 chapters (>= 15 Products)\n", sector.head(10), "\nStrongest\n", sector.tail(5))
print("Chapters where the blend loses to naive\n", sector.filter(pl.col("margin") < 0).sort("margin").head(10))
results["sector_weakest"], results["sector_strongest"] = sector.head(10).to_dicts(), sector.tail(5).to_dicts()
results["sector_losing"] = sector.filter(pl.col("margin") < 0).sort("margin").to_dicts()
# Region: rank correlation within each Product over that region's Markets only.
region = (
    g.group_by("hs6", "region").agg(_n=pl.len(), _k=pl.col("traded").sum(),
                                     _rho=pl.corr("winner", "value_usd", method="spearman"),
                                     _rn=pl.corr("naive_usd", "value_usd", method="spearman"))
    .filter((pl.col("_n") >= MIN_ROWS) & pl.col("_rho").is_not_nan() & pl.col("_rn").is_not_nan())
    .group_by("region").agg(products=pl.len(), blend=pl.col("_rho").mean(), naive=pl.col("_rn").mean())
    .sort("blend")
)
print("By Market region (within Product, within region)\n", region)
results["region"] = region.to_dicts()
# Which kind of Market the blend over- and under-rates: mean within-Product rank residual by region.
g = g.with_columns(
    _pr=(pl.col("winner").rank("average").over("hs6") / pl.len().over("hs6")),
    _ar=(pl.col("value_usd").rank("average").over("hs6") / pl.len().over("hs6")),
)
bias = (g.group_by("region").agg(pairs=pl.len(), rank_bias=(pl.col("_pr") - pl.col("_ar")).mean())
        .sort("rank_bias"))
print("Mean (predicted rank - actual rank) share by region: >0 means over-rated\n", bias)
results["region_rank_bias"] = bias.to_dicts()
mbias = (g.group_by("importer", "name").agg(pairs=pl.len(), rank_bias=(pl.col("_pr") - pl.col("_ar")).mean())
         .sort("rank_bias"))
print("Most under-rated Markets\n", mbias.head(10), "\nMost over-rated\n", mbias.tail(10))
results["market_under"], results["market_over"] = mbias.head(10).to_dicts(), mbias.tail(10).to_dicts()

# --- 2. Worked examples ----------------------------------------------------------------------
show = ["hs6", "name", "value_usd", "winner", "naive_usd", "n_suppliers", "hhi", "top_share", "similar_share",
        "sibling_imports", "uk_market_hs4_other", "rca_loo", "log_demand_loo", "log_distance", "eu_d"]
worst = (prod.filter((pl.col("k") >= 15) & (pl.col("uk_usd") >= 5e7)).sort("rho").head(10)
         .select("hs6", "description", "k", "uk_usd", "rho", "rho_naive"))
print("Worst-ranked Products (>= 15 UK Markets, >= $50m)\n", worst)
results["worst_products"] = worst.to_dicts()
over = g.with_columns(miss=pl.col("winner") - pl.col("value_usd")).sort("miss", descending=True).head(10)
under = g.with_columns(miss=pl.col("value_usd") - pl.col("winner")).sort("miss", descending=True).head(10)
print("Largest over-predictions\n", over.select(*show))
print("Largest under-predictions\n", under.select(*show))
results["over"], results["under"] = over.select(*show).to_dicts(), under.select(*show).to_dicts()
turb = g.filter((pl.col("hs6") == "841191") & pl.col("importer").is_in(["FRA", "DEU", "USA", "SGP", "CAN"]))
print("841191\n", turb.select(*show))
results["turbines"] = turb.select(*show).to_dicts()

# --- 3. Candidate features: within-Product association with the residual, and re-ranking -------
CANDIDATES = {
    "F1 suppliers": pl.col("n_suppliers").log1p(),
    "F2 concentration (HHI)": pl.col("hhi").fill_null(1.0),
    "F2b top supplier share": pl.col("top_share").fill_null(1.0),
    "F3 HS4 sibling imports": pl.col("sibling_imports").log1p(),
    "F4 UK-like imports": pl.col("similar_imports").log1p(),
    "F4b UK-like share": pl.col("similar_share").fill_null(0.0),
    "6a UK exports to Market, other Products": pl.col("uk_market_other").log1p(),
    "6b UK exports to Market, same HS4": pl.col("uk_market_hs4_other").log1p(),
}
for name, expr in CANDIDATES.items():
    g = g.with_columns(expr.alias(name))
    m, s = g[name].mean(), g[name].std()
    g = g.with_columns(((pl.col(name) - m) / s).alias(name))
BETAS = [-2, -1, -0.6, -0.4, -0.25, -0.15, -0.08, -0.04, 0, 0.04, 0.08, 0.15, 0.25, 0.4, 0.6, 1, 2]
fit, test = g.filter(pl.col("fold") <= 2), g.filter(pl.col("fold") >= 3)
fit_base, test_base, full_base = measure(fit, "lb"), measure(test, "lb"), measure(g, "lb")


def best_beta(df, terms: dict) -> float:
    scores = {b: measure(df, pl.col("lb") + sum(v * pl.col(n) for n, v in terms.items()) + b * pl.col(cand))
              for b in BETAS}
    return max(scores, key=scores.get)


rows = []
for cand in CANDIDATES:
    assoc = float(within(g, cand, pl.col("resid"))["_rho"].mean())
    assoc_traded = float(within(g.filter("traded"), cand, pl.col("resid"))["_rho"].mean())
    b = best_beta(fit, {})
    score = pl.col("lb") + b * pl.col(cand)
    rows.append({
        "candidate": cand, "assoc_resid": assoc, "assoc_resid_traded": assoc_traded, "beta": b,
        "gain_fit": measure(fit, score) - fit_base, "gain_test": measure(test, score) - test_base,
        "gain_all": measure(g, score) - full_base,
        "presence_gain_all": measure(g, score, pl.col("traded").cast(pl.Float64)) - zv,
        "non_zero_gain_all": measure(g.filter("traded"), score) - nz,
    })
cand_table = pl.DataFrame(rows)
print(f"Base: fit {fit_base:.4f} test {test_base:.4f} all {full_base:.4f}\n", cand_table)
results["candidates"] = rows
results["candidate_base"] = {"fit": fit_base, "test": test_base, "all": full_base}

# Allowed features together (F1-F4), coordinate ascent on folds 0-2.
allowed = [c for c in CANDIDATES if not c.startswith("6")]
terms = {c: 0.0 for c in allowed}
for _ in range(2):
    for cand in allowed:
        others = {n: v for n, v in terms.items() if n != cand}
        terms[cand] = best_beta(fit, others)
score = pl.col("lb") + sum(v * pl.col(n) for n, v in terms.items())
combo = {"betas": terms, "gain_fit": measure(fit, score) - fit_base, "gain_test": measure(test, score) - test_base,
         "gain_all": measure(g, score) - full_base}
print("F1-F4 together", combo)
results["combined_allowed"] = combo

# --- 4. Pooled vs per-fold: does aligning fold levels per Product help? ------------------------
off = g.with_columns(_d=pl.col("lb") - pl.col("ln"))
fold_off = off.group_by("hs6", "fold").agg(_o=pl.col("_d").median())
prod_off = off.group_by("hs6").agg(_O=pl.col("_d").median())
aligned = (off.join(fold_off, on=["hs6", "fold"]).join(prod_off, on="hs6")
           .with_columns(_al=pl.col("lb") - (pl.col("_o") - pl.col("_O"))))
global_off = off.group_by("fold").agg(_g=pl.col("_d").median())
aligned = aligned.join(global_off, on="fold").with_columns(_alg=pl.col("lb") - pl.col("_g"))
spread = fold_off.group_by("hs6").agg(_sd=pl.col("_o").std())
align = {
    "pooled": base["blend"], "per_fold_mean": per_fold["blend"],
    "naive_pooled": base["naive"], "naive_per_fold_mean": per_fold["naive"],
    "aligned_per_product_fold": measure(aligned, "_al"),
    "aligned_per_fold_global": measure(aligned, "_alg"),
    "fold_offset_global": global_off.sort("fold").to_dicts(),
    "median_sd_of_fold_offsets_within_product": float(spread["_sd"].median()),
}
print(json.dumps(align, indent=1))
results["alignment"] = align
(HERE / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
print("saved results.json")
