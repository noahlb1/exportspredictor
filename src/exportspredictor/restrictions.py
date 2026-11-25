"""Sanctioned Markets and Restricted Pairs (ADR 0005): the hand-kept, dated tables behind two Caveat Tiers.

Both tiers are scored and ranked as normal, like Re-export Hubs: they change what Advisors see, not the model or its
metrics. `caveats` declares them from these tables (`SANCTIONED_MARKET`, `RESTRICTED_PAIR`).

- `SANCTIONS_REGIMES`: Markets under *broad* UK trade sanctions, each with its regime, the year its broad trade
  measures began, the last year they were in force (None while they are) and its gov.uk guidance. A Market is a
  Sanctioned Market only in Baseline years its regime covers.
- `PROHIBITIONS`: Products a Market prohibits outright, with an official source. Every pair of the Market and a
  Product under one of the HS codes is a Restricted Pair.

Every entry was checked against its sources on `VERIFIED`. When an entry changes, check it again, update `VERIFIED`,
and write an artefact version whose Baseline is marked again (`python -m exportspredictor.pipeline recaveat`).

Left out on purpose, as checked on `VERIFIED`:

- **Iran.** Both UK Iran regimes (Iran, and Iran (nuclear)) are, in gov.uk's words, "limited in scope": they restrict
  named goods (strategic, nuclear, missile, military, internal repression and monitoring goods), not trade at large.
  https://www.gov.uk/government/publications/iran-sanctions-guidance and
  https://www.gov.uk/government/publications/iran-nuclear-sanctions-guidance
- **Other UK regimes** (e.g. Libya, Myanmar, Venezuela) are targeted at persons or named goods
  (https://www.gov.uk/government/collections/uk-sanctions-regimes-under-the-sanctions-act).
- **HS 2207 (ethyl alcohol).** The Markets below allow industrial and medical alcohol under approval, so it is
  restricted, not prohibited outright: Saudi Arabia requires "special approval" for products containing alcohol and
  bans "alcohol" as a beverage (https://www.trade.gov/country-commercial-guides/saudi-arabia-prohibited-and-restricted-imports).
"""

from dataclasses import dataclass
from datetime import date

import polars as pl

from exportspredictor.exclusions import as_code

VERIFIED = date(2026, 11, 20)  # the day every entry below was last checked against its sources


@dataclass(frozen=True)
class SanctionsRegime:
    """One Market's broad UK trade sanctions: its regime, the years they cover, and where they are set out."""

    iso3: str
    regime: str  # the regime's gov.uk name, e.g. "Russia"
    start_year: int  # the first year its broad trade measures were in force
    end_year: int | None  # the last year they were in force; None while they are
    source: str  # the regime's gov.uk guidance
    evidence: str  # the instrument that sets the start (or end) year

    def covers(self, year: int) -> bool:
        """Whether the regime's broad trade measures were in force in `year`."""
        return self.start_year <= year and (self.end_year is None or year <= self.end_year)

    @property
    def reason(self) -> str:
        """What a Sanctioned Market's caveat names: the regime and its start year."""
        return f"the UK's {self.regime} sanctions regime, broad since {self.start_year}"


SANCTIONS_REGIMES: tuple[SanctionsRegime, ...] = (
    SanctionsRegime(
        "RUS",
        "Russia",
        2022,
        None,
        "https://www.gov.uk/government/publications/russia-sanctions-guidance",
        # The 2019 regime was targeted until 2022's amendments added the trade bans (luxury goods: SI 2022/452).
        "https://www.legislation.gov.uk/uksi/2022/452/contents/made",
    ),
    SanctionsRegime(
        "BLR",
        "Republic of Belarus",
        2022,
        None,
        "https://www.gov.uk/government/publications/republic-of-belarus-sanctions-guidance",
        # SI 2022/748 (in force 5 July 2022) extended the trade bans to critical-industry goods and technology.
        "https://www.legislation.gov.uk/uksi/2022/748/contents/made",
    ),
    SanctionsRegime(
        "PRK",
        "Democratic People's Republic of Korea",
        2007,
        None,
        "https://www.gov.uk/government/publications/democratic-peoples-republic-of-korea-sanctions-guidance",
        # Council Regulation (EC) No 329/2007 (luxury goods, goods and technology), in force in the UK until the UK
        # regime replaced it on 31 December 2020; gov.uk calls the UK regime "broad in scope".
        "https://www.legislation.gov.uk/eur/2007/329/contents",
    ),
    SanctionsRegime(
        "SYR",
        "Syria",
        2011,
        2024,
        "https://www.gov.uk/government/publications/syria-sanctions-guidance",
        # Broad from Regulation (EU) No 442/2011 (May 2011; replaced by No 36/2012) until SI 2025/507 removed the
        # trade, oil, investment and financial restrictions after the fall of the Assad regime in December 2024.
        "https://www.legislation.gov.uk/uksi/2025/507/note/made",
    ),
)


@dataclass(frozen=True)
class Prohibition:
    """Products a Market prohibits outright, and an official source saying so."""

    iso3: str
    market: str  # the Market's name, for the caveat
    goods: str  # what is prohibited, e.g. "alcoholic beverages"
    hs_codes: tuple[str, ...]  # HS2, HS4 or HS6 codes: every Product under one of them is covered
    source: str

    @property
    def reason(self) -> str:
        """What a Restricted Pair's caveat names."""
        return f"{self.market} prohibits imports of {self.goods}"


# Beer (2203), wine (2204), vermouth (2205), other fermented beverages such as cider (2206) and spirits (2208).
# Alcohol-free beer is 2202, so it is not covered.
ALCOHOLIC_BEVERAGES = ("2203", "2204", "2205", "2206", "2208")

PROHIBITIONS: tuple[Prohibition, ...] = (
    Prohibition(
        "SAU",
        "Saudi Arabia",
        "alcoholic beverages",
        ALCOHOLIC_BEVERAGES,
        "https://www.trade.gov/country-commercial-guides/saudi-arabia-prohibited-and-restricted-imports",
    ),
    Prohibition(
        "KWT",
        "Kuwait",
        "alcoholic beverages",
        ALCOHOLIC_BEVERAGES,
        "https://www.ttb.gov/itd/international-affairs-resources-for-kuwait",
    ),
    Prohibition(
        "LBY",
        "Libya",
        "alcoholic beverages",
        ALCOHOLIC_BEVERAGES,
        # Law No. 4 of 1994 on the prohibition of alcohol, as amended by Law No. 21 of 2016 (Libyan legislation
        # database): making, trading in or dealing in alcohol in any way is an offence.
        "https://lawsociety.ly/akn/ly/act/law/gnc/2016/21",
    ),
    Prohibition(
        "IRN",
        "Iran",
        "alcoholic beverages",
        ALCOHOLIC_BEVERAGES,
        "https://pe.usps.com/text/imm/il_004.htm",
    ),
)


def sanctioned_market_expr(year: int, regimes: tuple[SanctionsRegime, ...] = SANCTIONS_REGIMES) -> pl.Expr:
    """True where the row's Market is under a broad UK sanctions regime in force in `year`."""
    return as_code("iso3").is_in([regime.iso3 for regime in regimes if regime.covers(year)])


def sanction_reason_expr(year: int, regimes: tuple[SanctionsRegime, ...] = SANCTIONS_REGIMES) -> pl.Expr:
    """The regime of each row's Market in `year`, as its caveat names it; null where there is none."""
    reason = pl.lit(None, pl.String)
    for regime in reversed([regime for regime in regimes if regime.covers(year)]):
        reason = pl.when(as_code("iso3") == regime.iso3).then(pl.lit(regime.reason)).otherwise(reason)
    return reason


def _covers(prohibition: Prohibition) -> pl.Expr:
    product = pl.col("hs6").cast(pl.String)
    return (as_code("iso3") == prohibition.iso3) & pl.any_horizontal(
        product.str.starts_with(code) for code in prohibition.hs_codes
    )


def restricted_pair_expr(prohibitions: tuple[Prohibition, ...] = PROHIBITIONS) -> pl.Expr:
    """True where the row's Market prohibits imports of the row's Product."""
    return pl.any_horizontal(pl.lit(False), *(_covers(prohibition) for prohibition in prohibitions))


def restriction_reason_expr(prohibitions: tuple[Prohibition, ...] = PROHIBITIONS) -> pl.Expr:
    """What each Restricted Pair's Market prohibits, as its caveat names it; null elsewhere."""
    reason = pl.lit(None, pl.String)
    for prohibition in reversed(prohibitions):
        reason = pl.when(_covers(prohibition)).then(pl.lit(prohibition.reason)).otherwise(reason)
    return reason
