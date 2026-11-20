"""Stage 3: wider beta grid for the candidates, combinations, and a blend weight by Product breadth.

Reads grid.parquet (stage 1, `build.py`); fits on folds 0-2, checks on folds 3-4. Appends to results2.json.
"""
import json

import numpy as np
import polars as pl

from exportspredictor.paths import data_dir

HERE = data_dir() / "derived" / "ranking_error_2022"
MIN_ROWS = 3
VALUE = pl.col("value_usd")  # the default ranking target: UK Actual Exports
g = pl.read_parquet(HERE / "grid.parquet").with_columns(
    lb=pl.col("winner").clip(1.0).log(), lm=pl.col("v2").clip(1.0).log(), ln=pl.col("naive_usd").clip(1.0).log(),
    traded=pl.col("value_usd") > 0,
)
# Breadth of the Product's trade from non-UK information only: Markets importing it from other Exporters.
breadth = g.group_by("hs6").agg(importing=(pl.col("n_suppliers") > 0).sum())
g = g.join(breadth, on="hs6")


def measure(df, pred, target=VALUE) -> float:
    p = pl.col(pred) if isinstance(pred, str) else pred
    w = (df.with_columns(_p=p, _t=target).group_by("hs6")
         .agg(_n=pl.len(), _rho=pl.corr("_p", "_t", method="spearman"))
         .filter((pl.col("_n") >= MIN_ROWS) & pl.col("_rho").is_not_nan() & pl.col("_rho").is_not_null()))
    return float(w["_rho"].mean())


CANDIDATES = {
    "F1 suppliers": pl.col("n_suppliers").log1p(),
    "F2 concentration (HHI)": pl.col("hhi").fill_null(1.0),
    "F3 HS4 sibling imports": pl.col("sibling_imports").log1p(),
    "F4 UK-like imports": pl.col("similar_imports").log1p(),
    "F4b UK-like share": pl.col("similar_share").fill_null(0.0),
    "6a UK exports to Market, other Products": pl.col("uk_market_other").log1p(),
    "6b UK exports to Market, same HS4": pl.col("uk_market_hs4_other").log1p(),
}
for name, expr in CANDIDATES.items():
    g = g.with_columns(expr.alias(name))
    g = g.with_columns(((pl.col(name) - g[name].mean()) / g[name].std()).alias(name))
BETAS = [-8, -5, -3, -2, -1.4, -1, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 1, 1.4, 2, 3, 5, 8]
fit, test = g.filter(pl.col("fold") <= 2), g.filter(pl.col("fold") >= 3)
base = {"fit": measure(fit, "lb"), "test": measure(test, "lb"), "all": measure(g, "lb")}


def score(terms):
    return pl.col("lb") + sum(v * pl.col(n) for n, v in terms.items() if v)


def best(df, terms, cand):
    trials = {b: measure(df, score({**terms, cand: b})) for b in BETAS}
    return max(trials, key=trials.get)


out = {"base": base, "single": [], "combined": {}}
for cand in CANDIDATES:
    b = best(fit, {}, cand)
    s = score({cand: b})
    out["single"].append({"candidate": cand, "beta": b, "gain_fit": measure(fit, s) - base["fit"],
                          "gain_test": measure(test, s) - base["test"], "gain_all": measure(g, s) - base["all"],
                          "at_edge": b in (BETAS[0], BETAS[-1])})
    print(out["single"][-1], flush=True)


def combine(names):
    terms = {n: 0.0 for n in names}
    for _ in range(2):
        for n in names:
            terms[n] = best(fit, {k: v for k, v in terms.items() if k != n}, n)
    s = score(terms)
    return {"betas": terms, "gain_fit": measure(fit, s) - base["fit"], "gain_test": measure(test, s) - base["test"],
            "gain_all": measure(g, s) - base["all"],
            "presence_all": measure(g, s, pl.col("traded").cast(pl.Float64))
            - measure(g, "lb", pl.col("traded").cast(pl.Float64)),
            "non_zero_all": measure(g.filter("traded"), s) - measure(g.filter("traded"), "lb")}


allowed = ["F1 suppliers", "F2 concentration (HHI)", "F3 HS4 sibling imports", "F4b UK-like share"]
out["combined"]["F1-F4"] = combine(allowed)
print("F1-F4", out["combined"]["F1-F4"], flush=True)
out["combined"]["F1-F4 + 6b"] = combine(allowed + ["6b UK exports to Market, same HS4"])
print("F1-F4 + 6b", out["combined"]["F1-F4 + 6b"], flush=True)

# A blend weight by Product breadth (non-UK information only): re-blend the model alone and the naive baseline.
g = g.with_columns(bucket=pl.col("importing").cut([30, 80, 140], labels=["<=30", "31-80", "81-140", "141+"]))
fit, test = g.filter(pl.col("fold") <= 2), g.filter(pl.col("fold") >= 3)
check = measure(g, 0.7 * pl.col("lm") + 0.3 * pl.col("ln"))
weights = {}
for bucket in ["<=30", "31-80", "81-140", "141+"]:
    sub = fit.filter(pl.col("bucket") == bucket)
    trials = {w: measure(sub, (1 - w) * pl.col("lm") + w * pl.col("ln")) for w in np.round(np.arange(0, 1.01, 0.1), 1)}
    weights[bucket] = float(max(trials, key=trials.get))
w_expr = pl.col("bucket").cast(pl.String).replace_strict(weights, return_dtype=pl.Float64)
s = (1 - w_expr) * pl.col("lm") + w_expr * pl.col("ln")
out["blend_by_breadth"] = {
    "check_reblend_0.3_all": check, "weights": weights,
    "gain_fit": measure(fit, s) - measure(fit, "lb"), "gain_test": measure(test, s) - measure(test, "lb"),
    "gain_all": measure(g, s) - measure(g, "lb"),
    "products_by_bucket": g.group_by("bucket").agg(pl.col("hs6").n_unique()).sort("bucket").to_dicts(),
}
print("blend by breadth", out["blend_by_breadth"], flush=True)
(HERE / "results2.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
