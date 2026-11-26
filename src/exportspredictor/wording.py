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


def term_definitions(gap_period_label: str, baseline_year: int) -> dict[str, str]:
    """The one source of the key-term explanations every app page shows on hover or behind `?`, by key (mostly the
    row field). A term shown in more than one context (a table column, the detail panel) has one entry for each.

    `gap_period_label` is the artefact's Gap Period label (`gap_period.label`), e.g. "2022–2024 average";
    `baseline_year` is its Baseline year.
    """
    from exportspredictor.illustrative import ILLUSTRATIVE  # it imports this module

    year, word = baseline_year, ILLUSTRATIVE.capitalize()
    tariff = f"The tariff the Market charges on this Product from the UK in the Baseline year {year}"
    blocked_value = (
        f"The rise in {year} Export Potential the model associates with adding a UK Trade Agreement where none is in "
        "force (a known Applied Tariff goes to 0 with it); 0 where one is."
    )
    relative_gap = "the share of its Export Potential the UK does not yet realise."
    return {
        "product": "The Product (HS6, 2012 edition).",
        "market": "The Market the UK exports to.",
        "hs6": "The Product's HS6 code (2012 edition).",
        "sector": "The Product's Sector (HS2 chapter).",
        "region": "The Market's region.",
        "export_potential": (
            "The value of UK exports of this Product to this Market that the model expects, given the fundamentals "
            "of the UK, the Market and the Product."
        ),
        "baseline_export_potential": f"Baseline Export Potential: what the model expects in {year}, with no agreement.",
        "actual_exports": f"What the UK actually exported of this Product to this Market ({gap_period_label}).",
        "baseline_actual_exports": (
            f"What the UK actually exported of this Product to this Market in {year} (0 where there was no Flow)."
        ),
        "export_gap": (
            f"Export Potential minus Actual Exports ({gap_period_label}). Positive means the UK exports less than "
            "its fundamentals suggest: an Opportunity."
        ),
        "relative_export_gap": f"Export Gap divided by Export Potential ({gap_period_label}): {relative_gap}",
        "relative_export_gap_in_panel": f"Export Gap divided by Export Potential: {relative_gap}",
        "opportunity_type": OPPORTUNITY_TYPE_HELP,
        "potential_range": (
            f"Where Actual Exports usually fall for comparable pairs, low–high in $ millions ({gap_period_label})."
        ),
        "applied_tariff": f"{tariff}, in percent. Blank where it is not known.",
        "applied_tariff_in_panel": f"{tariff}.",
        "trade_agreement": f"Whether a UK Trade Agreement with the Market is in force in {year}.",
        "blocked_opportunity": "The Blocked Opportunity value clears the fixed thresholds.",
        "blocked_opportunity_value": f"{blocked_value} {word}.",
        "blocked_opportunity_value_in_panel": blocked_value,
        "barrier": (
            f"What holds this pair's Export Potential down: no UK Trade Agreement is in force, with its {year} "
            "Applied Tariff (0%, above 0, or unknown)."
        ),
        "rise_with_agreement": (
            f"The Blocked Opportunity value: the rise in {year} Export Potential the model associates with adding a "
            f"UK Trade Agreement. It is the Tariff part plus the Agreement part. {word}."
        ),
        "rise_share": f"The Rise share: the rise with agreement as a share of Baseline Export Potential ({year}). {word}.",
        "tariff_part": (
            "The Tariff part: the part of the rise the model associates with the Applied Tariff alone, a known tariff "
            "set to 0 with no Trade Agreement added. It is 0 where the tariff is 0% or unknown: the rise then rests on "
            f"the agreement as a whole. {word}."
        ),
        "scenario_effect": (
            f"Scenario Export Potential minus Baseline Export Potential, {year}: the change the model associates "
            f"with the Scenario, holding everything else at the Baseline. {word}."
        ),
        "market_reach": (
            "How widely the UK sends its Products to this Market, against how widely it does across Markets. The model "
            "associates broad UK reach into a Market with higher Export Potential there, for every Product alike, so "
            "where the UK sends fewer Products to a whole Market than expected, it shows here, not in any one "
            "Product's figures."
        ),
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
