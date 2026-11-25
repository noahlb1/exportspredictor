"""Reading training chunks: a table in memory, a lazy frame or a Parquet path."""

import polars as pl

from exportspredictor.training import read_rows

ROWS = pl.DataFrame({"exporter": ["GBR", "FRA", "GBR"], "value_usd": [1.0, 2.0, 3.0]})
IS_UK = pl.col("exporter") == "GBR"


def test_read_rows_keeps_the_matching_rows_of_a_table_in_memory():
    assert read_rows(ROWS, IS_UK)["value_usd"].to_list() == [1.0, 3.0]
    assert read_rows(ROWS.lazy(), IS_UK)["value_usd"].to_list() == [1.0, 3.0]


def test_read_rows_keeps_the_matching_rows_of_a_parquet_file(tmp_path):
    path = tmp_path / "training_2019.parquet"
    ROWS.write_parquet(path)

    assert read_rows(path, IS_UK)["value_usd"].to_list() == [1.0, 3.0]
    assert read_rows(str(path), IS_UK)["value_usd"].to_list() == [1.0, 3.0]
