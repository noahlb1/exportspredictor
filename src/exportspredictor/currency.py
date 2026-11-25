"""Money in nominal US dollars, and the Bank of England rates for the HMRC cross-check.

Every value the model, the API, the app and the notebook show is in nominal US dollars,
BACI's unit (`USD_CAPTION`). There is no £ output. The Bank of England table below is kept
only to convert HMRC's £ declarations into USD at each year's own annual average, for the
HMRC cross-check (`exportspredictor.hmrc.crosscheck`).

The table is copied from the Bank of England database
(https://www.bankofengland.co.uk/boeapps/database/, series XUAAUSS), downloaded 2026-11-17.
"""

USD_CAPTION = "Values are nominal US dollars, as reported by BACI"
USD_UNIT = "USD, nominal"  # the unit in column headers, e.g. "Export Gap (USD, nominal)"

GBP_RATE_SERIES = "XUAAUSS"
GBP_RATE_DOWNLOADED = "2026-11-17"  # when the table below was copied from the Bank of England database

# US dollars per pound sterling, annual average.
USD_PER_GBP = {
    2012: 1.5851,
    2013: 1.5644,
    2014: 1.6477,
    2015: 1.5286,
    2016: 1.3542,
    2017: 1.2888,
    2018: 1.3350,
    2019: 1.2766,
    2020: 1.2837,
    2021: 1.3757,
    2022: 1.2362,
    2023: 1.2434,
    2024: 1.2783,
    2025: 1.3185,
}


def _amount(value: float) -> str:
    for unit, size in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if value >= size:
            return f"${value / size:,.1f}{unit}"
    return f"${value:,.0f}"


def format_usd(value: float) -> str:
    """A rounded amount of US dollars, e.g. $1.2bn, $350.0m, -$4.5k."""
    return ("-" if value < 0 else "") + _amount(abs(value))


def format_change(value: float) -> str:
    """A signed, rounded amount of US dollars, e.g. +$1.2bn or -$350.0m."""
    return ("-" if value < 0 else "+") + _amount(abs(value))
