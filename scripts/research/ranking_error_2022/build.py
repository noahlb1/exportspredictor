"""Stage 1: the five folds' held-out 2022 UK scores with lookups and candidate features (2022 only).

Writes grid.parquet to <data>/derived/ranking_error_2022. Reads no 2023 file.
"""
import sys

import numpy as np
import polars as pl

from exportspredictor.features import features_dir, load_sources, load_year_flows
from exportspredictor.loop import loop_dir
from exportspredictor.paths import data_dir

UK = "GBR"
YEAR = 2022
OUT = data_dir() / "derived" / "ranking_error_2022"
OUT.mkdir(parents=True, exist_ok=True)
root = data_dir()

# --- held-out scores -----------------------------------------------------------------
grid = pl.concat([
    pl.read_parquet(loop_dir(root) / "confirmation" / f"fold_{k}.parquet")
    .with_columns(pl.col("hs6").cast(pl.String), fold=pl.lit(k, pl.Int8))
    for k in range(5)
])
print("grid", grid.height, grid.columns, flush=True)

# --- 2022 flows ----------------------------------------------------------------------
flows = load_year_flows(YEAR, load_sources(root)[1], root).select(
    "exporter", "importer", pl.col("hs6").cast(pl.String), pl.col("value_usd").cast(pl.Float64)
).filter(pl.col("value_usd") > 0)
assert flows["exporter"].n_unique() > 100
others = flows.filter(pl.col("exporter") != UK)
uk = flows.filter(pl.col("exporter") == UK)
print("flows", flows.height, "uk", uk.height, flush=True)

# F1 suppliers and F2 concentration: non-UK Exporters selling the Product to the Market.
pm = others.group_by("hs6", "importer").agg(
    n_suppliers=pl.len(),
    imports_others=pl.col("value_usd").sum(),
    hhi=((pl.col("value_usd") / pl.col("value_usd").sum()) ** 2).sum(),
    top_share=pl.col("value_usd").max() / pl.col("value_usd").sum(),
)

# F3: the Market's imports (non-UK) of the same HS4's other codes.
hs4 = others.with_columns(hs4=pl.col("hs6").str.slice(0, 4))
by_hs4 = hs4.group_by("hs4", "importer").agg(hs4_imports=pl.col("value_usd").sum())
sibling = (
    pm.select("hs6", "importer", "imports_others")
    .with_columns(hs4=pl.col("hs6").str.slice(0, 4))
    .join(by_hs4, on=["hs4", "importer"], how="left")
)

# F4: imports from Exporters whose basket is like the UK's, the UK basket leaving out the Market.
products = sorted(flows["hs6"].unique().to_list())
p_index = {p: i for i, p in enumerate(products)}
exporters = sorted(others["exporter"].unique().to_list())
e_index = {e: i for i, e in enumerate(exporters)}
markets = sorted(flows["importer"].unique().to_list())
m_index = {m: i for i, m in enumerate(markets)}
E = np.zeros((len(exporters), len(products)))
eb = others.group_by("exporter", "hs6").agg(v=pl.col("value_usd").sum())
E[[e_index[e] for e in eb["exporter"]], [p_index[p] for p in eb["hs6"]]] = eb["v"].to_numpy()
U = np.zeros(len(products))
ub = uk.group_by("hs6").agg(v=pl.col("value_usd").sum())
U[[p_index[p] for p in ub["hs6"]]] = ub["v"].to_numpy()
UM = np.zeros((len(markets), len(products)))  # UK exports to each Market
UM[[m_index[m] for m in uk["importer"]], [p_index[p] for p in uk["hs6"]]] = uk["value_usd"].to_numpy()
U_loo = U[None, :] - UM  # markets x products: UK basket without that Market
# Cosine on log(1 + value), so a few huge Products (oil, gas) do not decide who looks like the UK.
El, Ul = np.log1p(E), np.log1p(np.clip(U_loo, 0, None))
dot = El @ Ul.T  # exporters x markets
sim = dot / (np.linalg.norm(El, axis=1)[:, None] * np.linalg.norm(Ul, axis=1)[None, :])
sim_long = pl.DataFrame({
    "exporter": np.repeat(exporters, len(markets)),
    "importer": np.tile(markets, len(exporters)),
    "sim": sim.ravel(),
})
weighted = (
    others.join(sim_long, on=["exporter", "importer"], how="left")
    .group_by("hs6", "importer")
    .agg(similar_imports=(pl.col("value_usd") * pl.col("sim")).sum(),
         similar_share=(pl.col("value_usd") * pl.col("sim")).sum() / pl.col("value_usd").sum())
)
print("similarity: UK-like top", [exporters[i] for i in np.argsort(-sim.mean(axis=1))[:8]], flush=True)

# Idea 6: UK exports of other Products to the same Market (breaks the leave-one-out rule as written).
uk_hs4 = uk.with_columns(hs4=pl.col("hs6").str.slice(0, 4))
uk_market = uk.group_by("importer").agg(uk_market_total=pl.col("value_usd").sum())
uk_market_hs4 = uk_hs4.group_by("importer", "hs4").agg(uk_market_hs4=pl.col("value_usd").sum())

# --- assemble ------------------------------------------------------------------------
features = pl.read_parquet(features_dir(root) / "uk_grid_2022.parquet").select(
    pl.col("hs6").cast(pl.String), pl.col("importer").cast(pl.String),
    "log_distance", "log_demand_loo", "rca_loo", "high_income_share_loo", "reexport_ratio",
    "trade_agreement", "eu_d", "common_language", "log_gdp_d", "relatedness",
)
lk = root / "derived" / "lookups"
markets_lk = pl.read_parquet(lk / "markets.parquet").select("iso3", "name", "region")
products_lk = pl.read_parquet(lk / "products.parquet").select(pl.col("hs6").cast(pl.String), "description")

out = (
    grid.with_columns(hs4=pl.col("hs6").str.slice(0, 4), sector=pl.col("hs6").str.slice(0, 2))
    .join(pm, on=["hs6", "importer"], how="left")
    .join(sibling.select("hs6", "importer", "hs4_imports"), on=["hs6", "importer"], how="left")
    .join(weighted, on=["hs6", "importer"], how="left")
    .join(uk_market, on="importer", how="left")
    .join(uk_market_hs4, on=["importer", "hs4"], how="left")
    .join(features, on=["hs6", "importer"], how="left")
    .join(markets_lk, left_on="importer", right_on="iso3", how="left")
    .join(products_lk, on="hs6", how="left")
    .with_columns(
        pl.col("n_suppliers", "imports_others", "hs4_imports", "similar_imports", "uk_market_total",
               "uk_market_hs4").fill_null(0),
        # Leave the pair's own Flow out of the HS4 and UK totals.
        sibling_imports=(pl.col("hs4_imports").fill_null(0) - pl.col("imports_others").fill_null(0)).clip(0),
        uk_market_other=(pl.col("uk_market_total").fill_null(0) - pl.col("value_usd")).clip(0),
        uk_market_hs4_other=(pl.col("uk_market_hs4").fill_null(0) - pl.col("value_usd")).clip(0),
    )
)
assert out.height == grid.height, (out.height, grid.height)
out.write_parquet(OUT / "grid.parquet")
print("wrote", OUT / "grid.parquet", out.height, flush=True)
sys.exit(0)
