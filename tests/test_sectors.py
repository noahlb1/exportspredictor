"""Sectors: the HS2 chapters Products group into."""

import polars as pl

from exportspredictor.sectors import in_sectors

PRODUCTS = pl.DataFrame({"hs6": ["220830", "870323", "220421"]})


def test_in_sectors_keeps_the_products_of_the_named_sectors():
    assert PRODUCTS.filter(in_sectors(["22"]))["hs6"].to_list() == ["220830", "220421"]


def test_no_sectors_means_every_product():
    assert PRODUCTS.filter(in_sectors(None)).height == 3
    assert PRODUCTS.filter(in_sectors([])).height == 3
