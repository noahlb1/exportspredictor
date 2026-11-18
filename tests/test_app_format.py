"""The app's own presentation helpers print exactly as the package's (the app may not import the package)."""

import pytest

from app import format as app_format
from exportspredictor import currency, wording

AMOUNTS = [0.0, 12.0, 999.0, 1_500.0, -4_500.0, 350e6, 1.2e9, -2.5e9, 61.14e9]


@pytest.mark.parametrize("value", AMOUNTS)
def test_money_prints_as_the_package_prints_it(value):
    assert app_format.format_usd(value) == currency.format_usd(value)
    assert app_format.format_change(value) == currency.format_change(value)


@pytest.mark.parametrize("years", [[], [2019], [2016, 2017, 2018], [2022, 2024]])
def test_years_print_as_the_package_prints_them(years):
    assert app_format.years_text(years) == wording.years_text(years)


@pytest.mark.parametrize(("value", "spec"), [(None, ".2f"), (0.51234, ".2f"), (5.2, ".1f")])
def test_numbers_print_as_the_package_prints_them(value, spec):
    assert app_format.number_text(value, spec) == wording.number_text(value, spec)
