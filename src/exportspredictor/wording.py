"""Plain-text pieces shared by every report the package words: year spans, numbers that may be missing,
the caution line every app page shows under its heading and the note beside every summed figure.

Money has its own module (`currency.format_usd`); the Gap Period's label is `gap_period.label`.
"""

import math
from collections.abc import Iterable, Sequence

from exportspredictor import gap_period

# Under the heading of every page that shows figures (CODING_STANDARDS.md: language about results stays associational).
CAUTION = "Figures are associations the model has learned, not causes or forecasts."

# What "a Trade Agreement" means in the model (GLOSSARY.md, ADR 0011), said on the Blocked Opportunities page and
# the Scenario Builder, and in the Trade Agreement explanation.
TRADE_AGREEMENT_MEANING = (
    "The model knows only whether a Trade Agreement exists (any WTO-listed free trade agreement or customs union). "
    "Adding one means an average agreement with tariffs removed, not a specific deal."
)

OPPORTUNITY_TYPE_HELP = (
    "Expand: the UK already exports the Product to this Market. New Market: the UK exports the Product, "
    "but not to this Market. New Product: the UK exports the Product to no Market (over the Gap Period)."
)

# The Selection Measure in plain words: Model Evaluation's account of how the model was chosen and the term's
# explanation both say it so.
SELECTION_MEASURE_PLAIN = (
    "how well the model ranks each Product's Markets against UK Actual Exports, on pairs it was not trained on"
)


def totals_note(share: float | None, gap_period_label: str) -> str:
    """The note beside every sum of Export Potential, Export Gaps or Blocked Opportunity values: Export
    Potential is a typical value, not an average, so its sum falls short of Actual Exports. `share` is the
    artefact's `potential_share_of_actual_exports`; None words the note without a figure."""
    reaches = (
        "it falls short of their Actual Exports" if share is None
        else f"about {share:.0%} of Actual Exports ({gap_period_label})"
    )
    return (
        "Export Potential is a typical value, not an average, so its sum runs low: over every UK Product–Market pair, "
        f"{reaches}. Read totals as relative sizes, not as dollars the UK could add."
    )


def term_definitions(gap_years: Sequence[int], baseline_year: int) -> dict[str, str]:
    """The one source of the key-term explanations every app page shows on hover or behind `?`, by key (mostly the
    row field). The Blocked Opportunities page's Export Potential and Actual Exports are Baseline-year figures, not Gap
    Period means, so they have entries of their own (`baseline_export_potential`, `baseline_actual_exports`).

    `gap_years` are the artefact's first and last Gap Period years; `baseline_year` is its Baseline year.
    """
    from exportspredictor.illustrative import ILLUSTRATIVE  # it imports this module
    from exportspredictor.scenarios import (  # it imports this module
        BLOCKED_MIN_SHARE_TEXT,
        BLOCKED_MIN_USD_TEXT,
    )

    year, word, averaged = baseline_year, ILLUSTRATIVE.capitalize(), gap_period.label(gap_years)
    product = (
        "One of about 5,000 goods categories in the Harmonized System, the international customs classification, at "
        "its six-digit level (HS6, 2012 edition)."
    )
    return {
        "pair": "One Product sold to one Market.",
        "product": product,
        "market": "The Market the UK exports to.",
        "hs6": product,
        "sector": "A broad group of Products: a two-digit chapter of the Harmonized System (HS2), such as Machinery.",
        "heading": "A four-digit Harmonized System group, between a Sector and a Product.",
        "region": "The Market's region.",
        "export_potential": (
            "The yearly value of UK exports of this Product to this Market that the model expects from their "
            "fundamentals: economy size, distance, tariffs and existing trade."
        ),
        "baseline_export_potential": f"Baseline Export Potential: what the model expects in {year}, with no agreement.",
        "actual_exports": f"What the UK actually exported of this Product to this Market ({averaged}).",
        "baseline_actual_exports": (
            f"What the UK actually exported of this Product to this Market in {year} (0 where there was no Flow)."
        ),
        "export_gap": (
            f"Export Potential minus Actual Exports ({averaged}). Positive means the UK exports less than "
            "its fundamentals suggest: an Opportunity."
        ),
        "opportunity": "A UK Product–Market pair with a positive Export Gap: the UK exports less than the model expects.",
        "relative_export_gap": (
            f"Export Gap divided by Export Potential ({averaged}): the share of its Export Potential the UK does not "
            "yet realise."
        ),
        "opportunity_type": OPPORTUNITY_TYPE_HELP,
        "potential_range": (
            f"Where Actual Exports usually fall for comparable pairs, low–high in $ millions ({averaged})."
        ),
        "uncertain_opportunity": (
            "An Opportunity whose Actual Exports lie inside its Potential Range: its Export Gap is within the variation "
            "normally seen among comparable pairs."
        ),
        "applied_tariff": (
            f"The import tax a Market charges on a Product from the UK in {year}, in percent: the lower of its "
            "preferential and standard (most-favoured-nation) rates. Missing where the tariff data hold none."
        ),
        "trade_agreement": f"Whether a UK Trade Agreement with the Market is in force in {year}. {TRADE_AGREEMENT_MEANING}",
        "trading_bloc": (
            "A named group of countries, such as the EU or CPTPP, whose members have Trade Agreements with each other."
        ),
        "customs_union": "A Trading Bloc whose members also charge the same tariffs on goods from outside.",
        "blocked_opportunity": (
            "The model associates adding a Trade Agreement with a rise in Export Potential of at least "
            f"{BLOCKED_MIN_SHARE_TEXT} and {BLOCKED_MIN_USD_TEXT}, and Actual Exports are below that level."
        ),
        "blocked_opportunity_value": (
            f"The rise in {year} Export Potential the model associates with adding a UK Trade Agreement where none is "
            f"in force (a known Applied Tariff goes to 0 with it); 0 where one is. {word}."
        ),
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
            "The part of the rise the model associates with setting the Applied Tariff alone to 0. 0 where the tariff is "
            f"0% or unknown: the rise then rests on the agreement as a whole. {word}."
        ),
        "agreement_part": (
            "The rest of the rise with agreement after the Tariff part: what the model associates with having an "
            "average Trade Agreement at all (standards, customs, legal frameworks)."
        ),
        "baseline": (
            f"Export Potential under the trade policy actually in force. The Baseline year ({year}) is the year scored."
        ),
        "gap_period": (
            f"The recent years ({years_text(gap_years)}) whose yearly Export Gaps are averaged into the Export Gap."
        ),
        "scenario": "A hypothetical change to UK trade policy: a Trade Agreement, a tariff or a Trading Bloc membership.",
        "scenario_effect": (
            f"Scenario Export Potential minus Baseline Export Potential, {year}: the change the model associates "
            f"with the Scenario, holding everything else at the Baseline. {word}."
        ),
        "illustrative": (
            "A label for Scenario figures: associations learned from past trade, higher than the changes seen after "
            "real agreements, not forecasts."
        ),
        "usd_nominal": "US dollars of the year shown, not adjusted for inflation.",
        "market_reach": (
            "How many of the Products this Market imports the UK sends there, against the UK's usual share across "
            "Markets. The model associates wider reach with higher Export Potential for every Product there alike, so "
            "a Market-wide lack of reach shows here, not in any one Product's figures."
        ),
        "naive_baseline": (
            "A simple estimate with no model: the UK's share of world exports of the Product times the Market's "
            "imports of it. The model has to beat it."
        ),
        "selection_measure": (
            f"The one score every modelling choice was made on: {SELECTION_MEASURE_PLAIN}. Tie-proof is its current "
            "form; the old measure let tied predictions share a rank."
        ),
        "model_form": (
            "How Export Potential is made: the model blended with the Naive Baseline, then scaled by Market Reach."
        ),
        "lift": (
            "How many times more often pairs with the largest Export Gaps later grew than pairs picked at random. "
            "Size-matched Lift compares them with pairs of similar Export Potential. A tendency, not a forecast."
        ),
        "event_backtest": (
            "A check of Scenario Effects against a real Trade Agreement: the model's association beside the change "
            "observed afterwards, relative to Control Markets, similar Markets without the agreement."
        ),
        "unrealised_share": "The share of a pair's Export Potential not yet exported in one year, from 0 to 1.",
        "itc": "The International Trade Centre, the UN and WTO agency that publishes the Export Potential Map.",
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
