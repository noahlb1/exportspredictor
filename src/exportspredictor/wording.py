"""Plain-text pieces shared by every report the package words: year spans and numbers that may be missing.

Money has its own module (`currency.format_usd`); the Gap Period's label is `gap_period.label`.
"""

import math
from collections.abc import Iterable


def years_text(years: Iterable[int]) -> str:
    """A span of years, e.g. "2022–2024", or one year alone ("2019"); "?" if none."""
    years = list(years)
    if not years:
        return "?"
    first, last = min(years), max(years)
    return str(first) if first == last else f"{first}–{last}"


def number_text(value: float | None, spec: str) -> str:
    """`value` formatted with `spec`, or "n/a" where it is missing (None or NaN): tiny worlds leave some results undefined."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def rate_text(rate: float | None) -> str:
    """A sampling rate as a percentage, with a decimal only where it matters (58.9%, 5%)."""
    if rate is None:
        return "an unrecorded rate"
    return f"{rate:.0%}" if round(rate * 100, 1) == round(rate * 100) else f"{rate:.1%}"
