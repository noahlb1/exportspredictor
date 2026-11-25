"""Plain-text pieces shared by every report the package words: year spans, numbers that may be missing,
the caution line every app page shows under its heading and the note beside every summed figure.

Money has its own module (`currency.format_usd`); the Gap Period's label is `gap_period.label`.
"""

import math
from collections.abc import Iterable

# Under the heading of every page that shows figures (CODING_STANDARDS.md: language about results stays associational).
CAUTION = "Figures are associations the model has learned, not causes or forecasts."

# What "a Trade Agreement" means in the model (GLOSSARY.md, ADR 0011), said on the Blocked Opportunities page and
# the Scenario Builder.
TRADE_AGREEMENT_MEANING = (
    "A Trade Agreement here is one yes/no flag in the model, on for any WTO-listed free trade agreement or customs "
    "union, so adding one means an average agreement as the model has seen them, with tariffs fully removed, not a "
    "specific deal."
)

OPPORTUNITY_TYPE_HELP = (
    "Expand: the UK already exports the Product to this Market. New Market: the UK exports the Product, "
    "but not to this Market. New Product: the UK exports the Product to no Market (over the Gap Period)."
)


def totals_note(share: float | None, gap_period_label: str) -> str:
    """The note beside every sum of Export Potential, Export Gaps or Blocked Opportunity values: Export
    Potential is a median-type estimate, so its sum falls short of Actual Exports. `share` is the
    artefact's `potential_share_of_actual_exports`; None words the note without a figure."""
    reaches = (
        "falls short of their Actual Exports" if share is None
        else f"comes to about {share:.0%} of Actual Exports ({gap_period_label})"
    )
    return (
        "Export Potential is a median-type estimate: summed over every UK Product–Market pair, it "
        f"{reaches}. Read totals as relative sizes, not as dollars the UK could add."
    )


def term_definitions(gap_period_label: str) -> dict[str, str]:
    """What each term the app's tables and panels explain in a tooltip means, by row field.

    `gap_period_label` is the artefact's Gap Period label (`gap_period.label`), e.g. "2022–2024 average".
    """
    return {
        "export_potential": (
            "The value of UK exports of this Product to this Market that the model expects, given the fundamentals "
            "of the UK, the Market and the Product."
        ),
        "actual_exports": f"What the UK actually exported of this Product to this Market ({gap_period_label}).",
        "export_gap": (
            f"Export Potential minus Actual Exports ({gap_period_label}). Positive means the UK exports less than "
            "its fundamentals suggest: an Opportunity."
        ),
        "opportunity_type": OPPORTUNITY_TYPE_HELP,
    }


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


def percent_text(change: float | None) -> str:
    """A change as a signed whole percentage ("+12%", "-5%"), or "n/a" where it is missing."""
    return "n/a" if change is None else f"{change:+.0%}"
