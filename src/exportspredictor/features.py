"""Feature table for Flows, computed identically for training and serving.

Leakage rule built in: supply and demand aggregates leave out the row's own pair, so a
UK row's supply feature never includes the target Market (leave-one-out, ADR 0001).
"""

import polars as pl

KEYS = ["exporter", "importer", "hs6", "year"]
POLICY_FEATURES = ["trade_agreement"]
FEATURES = ["log_supply_loo", "log_demand_loo", *POLICY_FEATURES]
# +1 on every policy feature: a Trade Agreement never lowers Export Potential (ADR 0002).
MONOTONE_CONSTRAINTS = {name: 1 for name in POLICY_FEATURES}


def build_features(flows: pl.DataFrame, policy: pl.DataFrame) -> pl.DataFrame:
    """Return KEYS, FEATURES and the target `value_usd` for every Flow row."""
    supply_total = pl.col("value_usd").sum().over("exporter", "hs6", "year")
    demand_total = pl.col("value_usd").sum().over("importer", "hs6", "year")
    return (
        flows.join(policy, on=["exporter", "importer", "year"], how="left")
        .with_columns(
            log_supply_loo=(supply_total - pl.col("value_usd")).log1p(),
            log_demand_loo=(demand_total - pl.col("value_usd")).log1p(),
            trade_agreement=pl.col("trade_agreement").fill_null(0),
        )
        .select(*KEYS, *FEATURES, "value_usd")
    )
