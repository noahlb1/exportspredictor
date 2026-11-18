"""Presentation only: how the pages print amounts of money and spans of years.

The app never imports the package (it reads the model through the API), so these are the
app's own copies of `exportspredictor.currency.format_usd` / `format_change` and
`exportspredictor.wording.years_text`; `tests/test_app_format.py` keeps them identical.
"""


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


def years_text(years: list[int]) -> str:
    """A span of years, e.g. "2022–2024", or one year alone ("2019"); "?" if none."""
    if not years:
        return "?"
    first, last = min(years), max(years)
    return str(first) if first == last else f"{first}–{last}"


def number_text(value: float | None, spec: str) -> str:
    """`value` formatted with `spec`, or "n/a" where it is missing."""
    return "n/a" if value is None else format(value, spec)
