"""Stage 4: how much each candidate would narrow the Potential Range, on 2022 log residuals.

The residual is log(1 + Actual Exports) − log(1 + blend). For each candidate, the adjustment is
log(1 + blend) + β · z(feature), with β chosen, separately within each size group, on fit rows to minimise the 10th–90th percentile
width of the residual and checked on held-out rows. Widths are reported overall and for pairs whose
blend Export Potential exceeds $10m and $100m, with the lower tail (median − 10th percentile) and
upper tail (90th − median) apart.

- UK: the five folds' held-out pairs (`build.py`), fit on folds 0–2, check on folds 3–4; also for
  UK pairs with a non-zero Flow in 2021.
- Non-UK: a sample of 2022 non-UK training rows scored by the loop's cached fold-0 model (trained
  on 2012–2021, so 2022 is out of sample for every non-UK row); fit on one hash half, check on the
  other, weighted by the rows' sampling weights. Nothing is trained: the script stops if the cached
  model is missing. The UK-like share (F4b) needs a similarity per Exporter and is not built here.

Writes range_width.json to <data>/derived/ranking_error_2022.
"""
import json
from dataclasses import replace

import numpy as np
import polars as pl

from exportspredictor.features import load_sources, load_year_flows
from exportspredictor.loop import (
    NOISE_FLOOR,
    Loop,
    Settings,
    load,
    loop_dir,
    real_data,
    training_key,
    v2_plan,
)
from exportspredictor.paths import data_dir

root = data_dir()
HERE = root / "derived" / "ranking_error_2022"
UK = "GBR"
SAMPLE = 1_500_000
BETAS = [round(b, 2) for b in np.concatenate([-np.geomspace(0.02, 4, 16)[::-1], [0.0], np.geomspace(0.02, 4, 16)])]
SIZES = {"all": 0.0, "> $10m": 1e7, "> $100m": 1e8}


def wq(values: np.ndarray, weights: np.ndarray, qs) -> list[float]:
    order = np.argsort(values, kind="stable")
    v, c = values[order], np.cumsum(weights[order])
    return [float(v[min(np.searchsorted(c, q * c[-1]), len(v) - 1)]) for q in qs]


def spread(r: np.ndarray, w: np.ndarray) -> dict:
    q10, q50, q90 = wq(r, w, (0.1, 0.5, 0.9))
    return {"rows": len(r), "width": q90 - q10, "lower_tail": q50 - q10, "upper_tail": q90 - q50}


def adjusted(df: pl.DataFrame, terms: dict) -> np.ndarray:
    r = df["resid"].to_numpy().copy()
    for name, b in terms.items():
        r -= b * df[name].to_numpy()
    return r


def width(df, terms) -> float:
    return spread(adjusted(df, terms), df["weight"].to_numpy())["width"]


def fit_terms(df, names) -> dict:
    terms = {n: 0.0 for n in names}
    for _ in range(2 if len(names) > 1 else 1):
        for n in names:
            trials = {b: width(df, {**terms, n: b}) for b in BETAS}
            terms[n] = min(trials, key=trials.get)
    return terms


def report(fit: pl.DataFrame, check: pl.DataFrame, names: list[str]) -> dict:
    """Per size group: β fitted on that group's fit rows (a range is calibrated per band of Export
    Potential, so one β for every size would trade large pairs' width for small pairs'), checked on
    its check rows."""
    out = {}
    for label, floor in SIZES.items():
        f, c = fit.filter(pl.col("blend") > floor), check.filter(pl.col("blend") > floor)
        if min(f.height, c.height) < 50:
            continue
        terms = fit_terms(f, names) if names else {}
        w = c["weight"].to_numpy()
        before, after = spread(c["resid"].to_numpy(), w), spread(adjusted(c, terms), w)
        out[label] = {"betas": terms, "at_edge": any(abs(b) == max(BETAS) for b in terms.values()),
                      "fit_rows": f.height, "before": before, "after": after,
                      "narrower_by": before["width"] - after["width"]}
    return out


def standardise(df: pl.DataFrame, exprs: dict) -> pl.DataFrame:
    df = df.with_columns(**{n: e for n, e in exprs.items()})
    return df.with_columns(((pl.col(n) - df[n].mean()) / df[n].std()).alias(n) for n in exprs)


def evaluate(fit: pl.DataFrame, check: pl.DataFrame, candidates: list[str], combos: dict) -> list[dict]:
    rows = [{"candidate": "blend as served", "check": report(fit, check, [])}]
    for name in [*candidates, *combos]:
        rows.append({"candidate": name, "check": report(fit, check, combos.get(name, [name]))})
        print(name, {k: round(v["narrower_by"], 3) for k, v in rows[-1]["check"].items()}, flush=True)
    return rows


# --- UK held-out pairs ------------------------------------------------------------------------------
UK_FEATURES = {
    "F1 suppliers": pl.col("n_suppliers").log1p(),
    "F2 concentration (HHI)": pl.col("hhi").fill_null(1.0),
    "F3 HS4 sibling imports": pl.col("sibling_imports").log1p(),
    "F4b UK-like share": pl.col("similar_share").fill_null(0.0),
    "6a own exports to Market, other Products": pl.col("uk_market_other").log1p(),
    "6b own exports to Market, same HS4": pl.col("uk_market_hs4_other").log1p(),
}
COMBOS = {
    "F1-F4": ["F1 suppliers", "F2 concentration (HHI)", "F3 HS4 sibling imports", "F4b UK-like share"],
    "F1-F4 + 6b": ["F1 suppliers", "F2 concentration (HHI)", "F3 HS4 sibling imports", "F4b UK-like share",
                   "6b own exports to Market, same HS4"],
}
uk = standardise(
    pl.read_parquet(HERE / "grid.parquet").with_columns(
        blend=pl.col("winner"), weight=pl.lit(1.0), resid=pl.col("value_usd").log1p() - pl.col("winner").log1p()
    ),
    UK_FEATURES,
)
to_iso3 = load_sources(root)[1]
traded_2021 = (
    load_year_flows(2021, to_iso3, root).filter((pl.col("exporter") == UK) & (pl.col("value_usd") > 0))
    .select(pl.col("hs6").cast(pl.String), "importer").unique().with_columns(traded_2021=pl.lit(True))
)
uk = uk.join(traded_2021, on=["hs6", "importer"], how="left").with_columns(pl.col("traded_2021").fill_null(False))
uk_fit, uk_check = uk.filter(pl.col("fold") <= 2), uk.filter(pl.col("fold") >= 3)
results = {"uk": evaluate(uk_fit, uk_check, list(UK_FEATURES), COMBOS)}
# Pairs that already traded in 2021: the served blend, and F1-F4 refitted on that subset.
traded_fit, traded_check = uk_fit.filter("traded_2021"), uk_check.filter("traded_2021")
results["uk_traded_2021"] = {
    "share_of_pairs": float(uk["traded_2021"].mean()),
    "share_of_2022_value": float(uk.filter("traded_2021")["value_usd"].sum() / uk["value_usd"].sum()),
    "blend": report(traded_fit, traded_check, []),
    "F1-F4": report(traded_fit, traded_check, COMBOS["F1-F4"]),
}
print("traded 2021", {k: {s: (round(v["before"]["width"], 3), round(v["after"]["width"], 3)) for s, v in r.items()}
                      for k, r in results["uk_traded_2021"].items() if isinstance(r, dict)}, flush=True)

# --- Non-UK 2022 rows, scored by the cached fold-0 model ------------------------------------------
saved = load(root)
winner = Settings.from_dict(saved["best"]["settings"])
assert saved[NOISE_FLOOR]  # the loop's results are present
lp = Loop(real_data(root), replace(v2_plan(root), fold=0), None, models_dir=loop_dir(root) / "models")
key = training_key(lp._resolved(winner), lp._fingerprint())
if not lp._model_file(key).exists():
    raise SystemExit(f"No cached fold-0 model {key}: this script never trains")
rows = lp.validation.filter(pl.col("exporter").cast(pl.String) != UK).sample(n=SAMPLE, seed=0)
rows = rows.with_columns(blend=lp.export_potential(winner, rows)).select(
    pl.col("exporter").cast(pl.String), pl.col("importer").cast(pl.String), pl.col("hs6").cast(pl.String),
    "value_usd", "weight", "blend",
)
print("scored non-UK rows", rows.height, flush=True)
flows = load_year_flows(2022, to_iso3, root).select(
    "exporter", "importer", pl.col("hs6").cast(pl.String), pl.col("value_usd").cast(pl.Float64)
).filter(pl.col("value_usd") > 0).with_columns(hs4=pl.col("hs6").str.slice(0, 4))
pm = flows.group_by("hs6", "importer").agg(_n=pl.len(), _s=pl.col("value_usd").sum(), _ss=(pl.col("value_usd") ** 2).sum())
h4 = flows.group_by("hs4", "importer").agg(_h4=pl.col("value_usd").sum())
own_h4 = flows.group_by("exporter", "hs4", "importer").agg(_own_h4=pl.col("value_usd").sum())
own_m = flows.group_by("exporter", "importer").agg(_own_m=pl.col("value_usd").sum())
v = pl.col("value_usd")
non_uk = (
    rows.with_columns(hs4=pl.col("hs6").str.slice(0, 4))
    .join(pm, on=["hs6", "importer"], how="left").join(h4, on=["hs4", "importer"], how="left")
    .join(own_h4, on=["exporter", "hs4", "importer"], how="left").join(own_m, on=["exporter", "importer"], how="left")
    .with_columns(pl.col("_n", "_s", "_ss", "_h4", "_own_h4", "_own_m").fill_null(0.0))
    # Every feature leaves the row's own Exporter out, as the UK's leave the UK out.
    .with_columns(
        _n_o=pl.col("_n") - (v > 0).cast(pl.Float64), _s_o=pl.col("_s") - v, _ss_o=pl.col("_ss") - v**2,
        _sib=((pl.col("_h4") - pl.col("_own_h4")) - (pl.col("_s") - v)).clip(0),
        _own_other=(pl.col("_own_m") - v).clip(0), _own_h4_other=(pl.col("_own_h4") - v).clip(0),
        resid=v.log1p() - pl.col("blend").log1p(),
    )
)
NON_UK_FEATURES = {
    "F1 suppliers": pl.col("_n_o").clip(0).log1p(),
    "F2 concentration (HHI)": pl.when(pl.col("_s_o") > 0).then(pl.col("_ss_o") / pl.col("_s_o") ** 2).otherwise(1.0),
    "F3 HS4 sibling imports": pl.col("_sib").log1p(),
    "6a own exports to Market, other Products": pl.col("_own_other").log1p(),
    "6b own exports to Market, same HS4": pl.col("_own_h4_other").log1p(),
}
non_uk = standardise(non_uk, NON_UK_FEATURES).with_columns(
    _half=(pl.concat_str("exporter", "hs6", "importer").hash(seed=7) % 2)
)
results["non_uk"] = evaluate(
    non_uk.filter(pl.col("_half") == 0), non_uk.filter(pl.col("_half") == 1), list(NON_UK_FEATURES),
    {"F1-F3": ["F1 suppliers", "F2 concentration (HHI)", "F3 HS4 sibling imports"],
     "F1-F3 + 6b": ["F1 suppliers", "F2 concentration (HHI)", "F3 HS4 sibling imports",
                    "6b own exports to Market, same HS4"]},
)
results["non_uk_sample"] = {"rows": non_uk.height, "model_key": key, "weighted": True}
(HERE / "range_width.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("saved", HERE / "range_width.json")
