"""Served Blocked Opportunities whose 'Export Potential with a Trade Agreement' is below what the UK already exports."""
import polars as pl
from common import latest_artefact

b = pl.read_parquet(latest_artefact() / "baseline.parquet").sort("blocked_opportunity_value_usd", descending=True)
b = b.with_columns(with_ta=pl.col("export_potential_usd") + pl.col("blocked_opportunity_value_usd"))
for n in (100, 500, 2000):
    t = b.head(n)
    print(f"top {n} Blocked Opportunities: with-agreement Export Potential below Actual Exports in",
          t.filter(pl.col("with_ta") < pl.col("actual_exports_usd")).height,
          "; Actual Exports above Export Potential in", t.filter(pl.col("actual_exports_usd") > pl.col("export_potential_usd")).height)
