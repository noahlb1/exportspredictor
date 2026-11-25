"""Product and Market codes read the same whatever type a table stores them as."""

import polars as pl

from exportspredictor.lookups import as_code


def test_a_code_column_is_read_as_text_whatever_its_stored_type():
    codes = pl.DataFrame({"iso3": pl.Series(["RUS", "FRA"], dtype=pl.Categorical), "hs6": [220830, 870323]})

    assert codes.select(as_code("iso3"), as_code("hs6")).rows() == [("RUS", "220830"), ("FRA", "870323")]
