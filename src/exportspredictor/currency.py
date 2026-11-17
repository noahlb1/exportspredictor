"""The USD → GBP rate used to show values in £: the Baseline year's annual average.

Model values are in USD (BACI's unit). Advisors talk in £, so rankings also give GBP at one
stated rate: the Bank of England's annual average of the daily spot rate, US$ into
sterling (series XUAAUSS), for the Baseline year. One rate per year keeps every figure in
a ranking comparable, and an annual average matches BACI's annual Flows.

The table is copied from the Bank of England database
(https://www.bankofengland.co.uk/boeapps/database/, series XUAAUSS), downloaded 2026-11-17.
"""

GBP_RATE_SOURCE = "Bank of England, series XUAAUSS: annual average spot exchange rate, US$ into sterling"

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


def gbp_rate(year: int) -> dict:
    """The rate for `year`, as stored in artefact metadata (`gbp_rate`): usd_per_gbp, year, source."""
    if year not in USD_PER_GBP:
        raise KeyError(f"No USD/GBP annual average for {year}; add it to currency.USD_PER_GBP")
    return {"usd_per_gbp": USD_PER_GBP[year], "year": year, "source": GBP_RATE_SOURCE}
