"""Trade Agreement, Customs Union and Trading Bloc features from the WTO RTA-IS export.

Source: the WTO RTA-IS "Export all RTAs" file (docs/research/trade-agreement-coverage.md).
Each agreement lists its parties, its goods entry-into-force date and per-party entry and
exit dates. We expand that into who was in which agreement on which dates, then into
pair-year and country-year flags.

**The 1 July rule** (as in CEPII Gravity): a membership counts in a year if it began
before 1 July of that year and had not ended by then.

Parties that are themselves blocs (e.g. "European Union" in EU–Japan) are expanded to
the bloc's member countries at each date, so Croatia joins EU agreements when it joins
the EU, and the UK leaves them when it leaves.
"""

import re
from datetime import date
from pathlib import Path

import polars as pl

from exportspredictor.paths import data_dir

RTA_EXPORT_URL = "https://rtais.wto.org/UI/ExportAllRTAList.aspx"

PAIR_FEATURES = ["trade_agreement", "customs_union"]
TRADING_BLOCS = ["eu", "cptpp", "efta"]
POLICY_FEATURES = [
    *PAIR_FEATURES,
    *(f"{bloc}_{side}" for bloc in TRADING_BLOCS for side in ("o", "d")),
]

# Records that define each Trading Bloc's membership. The EU is the EU Treaty plus every
# EC/EU enlargement record (each lists all members at the time, with UK exit dates).
_EU_RECORDS = [120, 145, 137, 132, 123, 52, 6, 869]
_BLOC_RECORDS = {
    "eu": _EU_RECORDS,
    "cptpp": [640, 1218],
    "efta": [152],
}
# Parties in the export that are blocs, and the records that define their membership.
_GROUP_PARTIES = {
    "European Union": _EU_RECORDS,
    "European Free Trade Association (EFTA)": [152],
    "ASEAN Free Trade Area (AFTA)": [1170],
    "Eurasian Economic Union (EAEU)": [909, 910, 922],
    "Gulf Cooperation Council (GCC)": [17],
    "Pacific Alliance": [889],
    "Southern African Customs Union (SACU)": [7],
    "Southern Common Market (MERCOSUR)": [130],
}
# Record-name prefixes that imply a bloc party.
_NAMED_PARTIES = {
    "EU - ": "European Union",
    "EC - ": "European Union",
    "EFTA - ": "European Free Trade Association (EFTA)",
}
# The WTO records keep the UK in the EU's agreements and customs union until the end of
# the transition period (31 Dec 2020). As a Trading Bloc member (political membership,
# as Gravity's `eu_o/_d` codes it) the UK left on 31 Jan 2020, which the records' Remarks
# state but no date column carries.
_EU_POLITICAL_EXITS = {"GBR": date(2020, 1, 31)}

_COLUMNS = {
    "RTA ID": "rta_id",
    "RTA Name": "name",
    "Type": "type",
    "Date of Entry into Force (G)": "entry_into_force",
    "Inactive Date": "inactive",
    "Original signatories": "original",
    "Current signatories": "current",
    "Specific Entry/Exit dates": "specific",
}

# WTO party names -> ISO 3166 alpha-3. Kosovo has no ISO code; XKX is the common stand-in.
COUNTRY_CODES = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND", "Angola": "AGO",
    "Anguilla": "AIA", "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM",
    "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE", "Bahamas": "BHS",
    "Bahrain, Kingdom of": "BHR", "Bangladesh": "BGD", "Barbados": "BRB", "Belarus": "BLR",
    "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN", "Bermuda": "BMU", "Bhutan": "BTN",
    "Bolivia, Plurinational State of": "BOL", "Bosnia and Herzegovina": "BIH", "Botswana": "BWA",
    "Brazil": "BRA", "British Indian Ocean Territory": "IOT",
    "British Overseas Territory of Saint Helena, Ascension and Tristan da Cunha": "SHN",
    "Brunei Darussalam": "BRN", "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cabo Verde": "CPV", "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
    "Cayman Islands": "CYM", "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL",
    "China": "CHN", "Chinese Taipei": "TWN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
    "Cook Islands": "COK", "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Côte d'Ivoire": "CIV", "Democratic Republic of the Congo": "COD",
    "Denmark": "DNK", "Djibouti": "DJI", "Dominica": "DMA", "Dominican Republic": "DOM",
    "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV", "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI", "Estonia": "EST", "Eswatini": "SWZ", "Ethiopia": "ETH",
    "Falkland Islands (Islas Malvinas)": "FLK", "Faroe Islands": "FRO", "Fiji": "FJI",
    "Finland": "FIN", "France": "FRA", "French Polynesia": "PYF",
    "French Southern Territories": "ATF", "Gabon": "GAB", "Georgia": "GEO", "Germany": "DEU",
    "Ghana": "GHA", "Greece": "GRC", "Greenland": "GRL", "Grenada": "GRD", "Guatemala": "GTM",
    "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY", "Haiti": "HTI", "Honduras": "HND",
    "Hong Kong, China": "HKG", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR",
    "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kiribati": "KIR", "Korea, Democratic People's Republic of": "PRK",
    "Korea, Republic of": "KOR", "Kosovo (under UNSC res. 1244)": "XKX",
    "Kuwait, the State of": "KWT", "Kyrgyz Republic": "KGZ",
    "Lao People's Democratic Republic": "LAO", "Latvia": "LVA", "Lebanese Republic": "LBN",
    "Lesotho": "LSO", "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Macao, China": "MAC", "Madagascar": "MDG",
    "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT",
    "Marshall Islands": "MHL", "Palau": "PLW", "South Sudan": "SSD",
    "Mauritania": "MRT", "Mauritius": "MUS", "Mayotte": "MYT", "Mexico": "MEX",
    "Micronesia, Federated States of": "FSM", "Moldova, Republic of": "MDA", "Mongolia": "MNG",
    "Montenegro": "MNE", "Montserrat": "MSR", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nauru": "NRU", "Nepal": "NPL", "Netherlands": "NLD",
    "Netherlands Antilles": "ANT", "Netherlands, Kingdom of, with respect to Aruba": "ABW",
    "New Caledonia": "NCL", "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER",
    "Nigeria": "NGA", "Niue": "NIU", "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN",
    "Pakistan": "PAK", "Palestine": "PSE", "Panama": "PAN", "Papua New Guinea": "PNG",
    "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL", "Pitcairn": "PCN", "Poland": "POL",
    "Portugal": "PRT", "Qatar": "QAT", "Romania": "ROU", "Russian Federation": "RUS",
    "Rwanda": "RWA", "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Pierre and Miquelon": "SPM", "Saint Vincent and the Grenadines": "VCT",
    "Samoa": "WSM", "San Marino": "SMR", "Sao Tomé and Principe": "STP",
    "Saudi Arabia, Kingdom of": "SAU", "Senegal": "SEN", "Serbia": "SRB", "Seychelles": "SYC",
    "Sierra Leone": "SLE", "Singapore": "SGP", "Slovak Republic": "SVK", "Slovenia": "SVN",
    "Solomon Islands": "SLB", "Somalia": "SOM", "South Africa": "ZAF",
    "South Georgia and the South Sandwich Islands": "SGS", "Spain": "ESP", "Sri Lanka": "LKA",
    "Sudan": "SDN", "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "CHE",
    "Syrian Arab Republic": "SYR", "Tajikistan": "TJK", "Tanzania": "TZA", "Thailand": "THA",
    "The Gambia": "GMB", "Togo": "TGO", "Tonga": "TON", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Turkmenistan": "TKM", "Turks and Caicos Islands": "TCA", "Tuvalu": "TUV",
    "Türkiye": "TUR", "Uganda": "UGA", "Ukraine": "UKR", "United Arab Emirates": "ARE",
    "United Kingdom": "GBR", "United States of America": "USA", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Vanuatu": "VUT", "Venezuela, Bolivarian Republic of": "VEN",
    "Viet Nam": "VNM", "Virgin Islands, British": "VGB", "Wallis and Futuna Islands": "WLF",
    "Western Sahara": "ESH", "Yemen": "YEM", "Yugoslavia, Socialist Federal Republic of": "YUG",
    "Zambia": "ZMB", "Zimbabwe": "ZWE",
}

# "Party(01-Mar-1970 - )", "Party( - 31-Dec-2020)" or "Party(01-Jan-1986 - 31-Dec-1994)".
_SPECIFIC = re.compile(r"^(?P<party>.+?)\s*\((?P<entry>[^()]*?) - (?P<exit>[^()]*?)\)$")


# --- Download -------------------------------------------------------------------------


def rta_export_path(root: Path | None = None) -> Path:
    return (root or data_dir()) / "raw" / "wto" / "AllRTAs.xlsx"


def download_rta_export(root: Path | None = None, *, force: bool = False) -> Path:
    """Download the WTO "Export all RTAs" file unless it is already there.

    Writes to `<name>.part` first and renames when complete, so no reader sees half a file.
    """
    import httpx

    target = rta_export_path(root)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with httpx.stream("GET", RTA_EXPORT_URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with part.open("wb") as out:
            for chunk in response.iter_bytes():
                out.write(chunk)
    part.replace(target)
    return target


# --- Reading and expanding memberships ------------------------------------------------


def read_rta_export(path: Path) -> pl.DataFrame:
    """Read the export (.xlsx, or a .csv extract with the same headers) into tidy columns."""
    path = Path(path)
    raw = pl.read_excel(path) if path.suffix == ".xlsx" else pl.read_csv(path, infer_schema=False)
    return read_rta_export_frame(raw)


def read_rta_export_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """Tidy an already-loaded export frame (WTO column headers) into the columns used here."""
    return raw.select(list(_COLUMNS)).rename(_COLUMNS).with_columns(
        pl.col("rta_id").cast(pl.Int64),
        pl.col("entry_into_force").cast(pl.String).str.slice(0, 10).str.to_date("%Y-%m-%d"),
        pl.col("inactive").cast(pl.String).str.slice(0, 10).str.to_date("%Y-%m-%d"),
    )


def _parse_day(text: str) -> date | None:
    from datetime import datetime

    text = text.strip()
    return datetime.strptime(text, "%d-%b-%Y").date() if text else None


def _split(names: str | None) -> list[str]:
    return [name.strip() for name in (names or "").split(";") if name.strip()]


def _party_spells(rtas: pl.DataFrame) -> pl.DataFrame:
    """One row per (agreement, party): when that party was in the agreement.

    A party starts at its specific entry date or else the goods entry-into-force date, and
    ends at the earlier of its specific exit date and the agreement's inactive date.

    Two gaps in the export are filled:
    - A record named "EU - X" or "EFTA - X" has the bloc as a party even when only its
      members at signature are listed (inactive records list no current signatories).
    - In the records that define a bloc's membership, a member listed only among current
      signatories with no specific entry date joined at an unknown date, so it is left out
      there and comes in through its accession record (e.g. Croatia's EU membership comes
      from "EU (28) Enlargement", not from the 1958 EU Treaty).
    """
    defining = {rta_id for records in _GROUP_PARTIES.values() for rta_id in records}
    rows = []
    for record in rtas.filter(pl.col("entry_into_force").is_not_null()).iter_rows(named=True):
        specific = {}
        for item in _split(record["specific"]):
            match = _SPECIFIC.match(item)
            if match is None:
                raise ValueError(f"RTA {record['rta_id']}: cannot parse entry/exit {item!r}")
            specific[match["party"]] = (_parse_day(match["entry"]), _parse_day(match["exit"]))
        original = _split(record["original"])
        implied = [group for prefix, group in _NAMED_PARTIES.items() if record["name"].startswith(prefix)]
        parties = dict.fromkeys(original + _split(record["current"]) + list(specific) + implied)
        for party in parties:
            entry, exit_ = specific.get(party, (None, None))
            if record["rta_id"] in defining and party not in original and entry is None:
                continue
            ends = [d for d in (exit_, record["inactive"]) if d is not None]
            rows.append(
                {
                    "rta_id": record["rta_id"],
                    "party": party,
                    "start": entry or record["entry_into_force"],
                    "end": min(ends) if ends else None,
                }
            )
    return pl.DataFrame(
        rows, schema={"rta_id": pl.Int64, "party": pl.String, "start": pl.Date, "end": pl.Date}
    )


def _country_spells(spells: pl.DataFrame) -> pl.DataFrame:
    """Map country parties to ISO3 and drop bloc parties (they are expanded separately)."""
    countries = spells.filter(~pl.col("party").is_in(list(_GROUP_PARTIES)))
    unknown = set(countries["party"]) - set(COUNTRY_CODES)
    if unknown:
        raise ValueError(f"WTO party names with no ISO3 code: {sorted(unknown)}")
    return countries.with_columns(iso3=pl.col("party").replace_strict(COUNTRY_CODES)).drop("party")


def memberships(rtas: pl.DataFrame) -> pl.DataFrame:
    """Every (agreement, country, start, end) spell, with bloc parties expanded to members.

    A country is in an agreement through a bloc party for the overlap of the bloc party's
    spell in the agreement and the country's spell in the bloc.

    `side` says which side of a bloc agreement a country is on: the bloc's name for
    countries in the bloc while in the agreement, "other" for the rest, and null when the
    agreement has no bloc party. EU - OCT links each territory with the EU, not the
    territories with each other; EU - United Kingdom puts the (former member) UK on the
    other side.
    """
    spells = _party_spells(rtas)
    direct = _country_spells(spells)
    bloc_spells = pl.concat(
        [
            direct.filter(pl.col("rta_id").is_in(records)).select(
                "iso3", group=pl.lit(group), member_start="start", member_end="end"
            )
            for group, records in _GROUP_PARTIES.items()
        ]
    )
    record_groups = spells.filter(pl.col("party").is_in(list(_GROUP_PARTIES))).select("rta_id", group="party").unique()
    on_bloc_side = (
        direct.join(record_groups, on="rta_id")
        .join(bloc_spells, on=["iso3", "group"])
        .filter(
            (pl.col("end").is_null() | (pl.col("member_start") <= pl.col("end")))
            & (pl.col("member_end").is_null() | (pl.col("member_end") >= pl.col("start")))
        )
        .select("rta_id", "iso3", "start", "group")
        .unique(["rta_id", "iso3", "start"])
    )
    direct = (
        direct.join(on_bloc_side, on=["rta_id", "iso3", "start"], how="left")
        .with_columns(
            side=pl.coalesce(
                "group", pl.when(pl.col("rta_id").is_in(record_groups["rta_id"].implode())).then(pl.lit("other"))
            )
        )
        .drop("group")
    )
    via_groups = []
    for group, records in _GROUP_PARTIES.items():
        members = direct.filter(pl.col("rta_id").is_in(records)).select(
            "iso3", member_start="start", member_end="end"
        )
        uses = spells.filter(pl.col("party") == group).drop("party")
        via_groups.append(
            uses.join(members, how="cross")
            .with_columns(
                start=pl.max_horizontal("start", "member_start"),
                end=pl.min_horizontal("end", "member_end"),
                side=pl.lit(group),
            )
            .filter(pl.col("end").is_null() | (pl.col("end") >= pl.col("start")))
            .select(direct.columns)
        )
    # Partial-scope agreements (type "PSA", e.g. GSTP) cut tariffs on a short list of
    # goods only. They are not Trade Agreements here, as in Gravity's `fta_wto`.
    agreements = rtas.filter(pl.col("type").is_not_null() & (pl.col("type") != "PSA")).select(
        "rta_id", customs_union=pl.col("type").str.starts_with("CU")
    )
    return pl.concat([direct, *via_groups]).join(agreements, on="rta_id")


def _in_force(spells: pl.DataFrame, years: list[int]) -> pl.DataFrame:
    """Expand spells to the years they count in under the 1 July rule."""
    cutoff = pl.date(pl.col("year"), 7, 1)
    return (
        spells.join(pl.DataFrame({"year": years}, schema={"year": pl.Int32}), how="cross")
        .filter((pl.col("start") < cutoff) & (pl.col("end").is_null() | (pl.col("end") >= cutoff)))
        .drop("start", "end")
    )


# --- Features -------------------------------------------------------------------------


def pair_agreements(rtas: pl.DataFrame, years: list[int]) -> pl.DataFrame:
    """Ordered country pairs with a Trade Agreement in force, per year.

    Columns: exporter, importer, year, trade_agreement (always 1), customs_union (0/1).
    Pairs and years not listed have no Trade Agreement.
    """
    active = _in_force(memberships(rtas), years)
    return (
        active.join(active, on=["rta_id", "year"], suffix="_d")
        .filter(
            (pl.col("iso3") != pl.col("iso3_d"))
            & (pl.col("side").is_null() | (pl.col("side") != pl.col("side_d")))
        )
        .group_by(exporter="iso3", importer="iso3_d", year="year")
        .agg(customs_union=(pl.col("customs_union") & pl.col("customs_union_d")).any())
        .select(
            "exporter",
            "importer",
            "year",
            trade_agreement=pl.lit(1, pl.Int8),
            customs_union=pl.col("customs_union").cast(pl.Int8),
        )
        .sort("exporter", "importer", "year")
    )


def bloc_membership(rtas: pl.DataFrame, years: list[int]) -> pl.DataFrame:
    """Trading Bloc membership per country per year: iso3, year and one 0/1 column per bloc.

    EU membership is political membership, so the UK leaves in 2020 even though it stays
    in the EU's Customs Union and agreements until the transition period ends.
    Countries in no tracked bloc in a year have no row.
    """
    spells = _country_spells(_party_spells(rtas))
    frames = []
    for bloc, records in _BLOC_RECORDS.items():
        bloc_spells = spells.filter(pl.col("rta_id").is_in(records))
        if bloc == "eu":
            exits = pl.DataFrame(
                {"iso3": list(_EU_POLITICAL_EXITS), "political_exit": list(_EU_POLITICAL_EXITS.values())}
            )
            bloc_spells = (
                bloc_spells.join(exits, on="iso3", how="left")
                .with_columns(end=pl.min_horizontal("end", "political_exit"))
                .drop("political_exit")
            )
        frames.append(_in_force(bloc_spells, years).select("iso3", "year", bloc=pl.lit(bloc)))
    return (
        pl.concat(frames)
        .unique()
        .with_columns(member=pl.lit(1, pl.Int8))
        .pivot(on="bloc", index=["iso3", "year"], values="member")
        .pipe(_ensure_bloc_columns)
        .sort("iso3", "year")
    )


def _ensure_bloc_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col(bloc) if bloc in frame.columns else pl.lit(None)).fill_null(0).cast(pl.Int8).alias(bloc)
        for bloc in TRADING_BLOCS
    ).select("iso3", "year", *TRADING_BLOCS)


def attach_policy(rows: pl.DataFrame, agreements: pl.DataFrame, blocs: pl.DataFrame) -> pl.DataFrame:
    """Add every POLICY_FEATURES column to rows keyed by exporter, importer and year.

    Pairs with no agreement, and countries in no tracked bloc, get 0.
    """
    year_type = rows.schema["year"]
    agreements = agreements.with_columns(pl.col("year").cast(year_type))
    blocs = blocs.with_columns(pl.col("year").cast(year_type))
    return (
        rows.join(agreements, on=["exporter", "importer", "year"], how="left")
        .join(blocs.rename({b: f"{b}_o" for b in TRADING_BLOCS}), left_on=["exporter", "year"], right_on=["iso3", "year"], how="left")
        .join(blocs.rename({b: f"{b}_d" for b in TRADING_BLOCS}), left_on=["importer", "year"], right_on=["iso3", "year"], how="left")
        .with_columns(pl.col(POLICY_FEATURES).fill_null(0).cast(pl.Int8))
        .select(*rows.columns, *POLICY_FEATURES)
    )


# --- Build entry point ----------------------------------------------------------------

DEFAULT_YEARS = list(range(2012, 2025))


def policy_paths(root: Path | None = None) -> dict[str, Path]:
    derived = (root or data_dir()) / "derived"
    return {
        "agreements": derived / "policy_agreements.parquet",
        "trading_blocs": derived / "policy_trading_blocs.parquet",
    }


def build(root: Path | None = None, years: list[int] = DEFAULT_YEARS, source: Path | None = None) -> dict[str, Path]:
    """Download the WTO export if missing, then write the policy tables. Safe to re-run.

    Writes `derived/policy_agreements.parquet` (exporter, importer, year, trade_agreement,
    customs_union) and `derived/policy_trading_blocs.parquet` (iso3, year, eu, cptpp, efta)
    under the data root, and returns their paths.
    """
    rtas = read_rta_export(source or download_rta_export(root))
    paths = policy_paths(root)
    paths["agreements"].parent.mkdir(parents=True, exist_ok=True)
    tables = {"agreements": pair_agreements(rtas, years), "trading_blocs": bloc_membership(rtas, years)}
    for name, table in tables.items():
        part = paths[name].with_name(paths[name].name + ".part")
        table.write_parquet(part)
        part.replace(paths[name])
    return paths


# --- Cross-check against CEPII Gravity ------------------------------------------------

GRAVITY_ZIP = Path("raw") / "gravity" / "Gravity_csv_V202211.zip"
GRAVITY_YEARS = list(range(2012, 2022))  # Gravity V202211's policy variables end in 2021


def read_gravity_policy(root: Path | None = None, years: list[int] = GRAVITY_YEARS) -> pl.DataFrame:
    """Gravity's policy columns for existing countries, read straight from the release zip.

    Cached as `derived/gravity_policy_check.parquet`, since the CSV is about 700 MB.
    """
    import zipfile

    root = root or data_dir()
    cache = root / "derived" / "gravity_policy_check.parquet"
    if not cache.exists():
        columns = ["year", "iso3_o", "iso3_d", "country_exists_o", "country_exists_d", "eu_o", "eu_d", "fta_wto"]
        with zipfile.ZipFile(root / GRAVITY_ZIP) as archive, archive.open("Gravity_V202211.csv") as csv:
            gravity = pl.read_csv(csv, columns=columns, schema_overrides={"year": pl.Int32})
        cache.parent.mkdir(parents=True, exist_ok=True)
        part = cache.with_name(cache.name + ".part")
        gravity.filter(
            pl.col("year").is_in(GRAVITY_YEARS)
            & (pl.col("country_exists_o") == 1)
            & (pl.col("country_exists_d") == 1)
            & (pl.col("iso3_o") != pl.col("iso3_d"))
        ).drop("country_exists_o", "country_exists_d").write_parquet(part)
        part.replace(cache)
    return pl.read_parquet(cache).filter(pl.col("year").is_in(years))


def gravity_crosscheck(
    agreements: pl.DataFrame, blocs: pl.DataFrame, gravity: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare our flags with Gravity's `fta_wto` (pairs) and `eu_o/_d` (countries).

    Compares every pair-year (and country-year) that Gravity has. Returns a summary per
    feature (pair_years, agreement_rate, both, ours_only, gravity_only, positive_overlap) and the
    disagreements grouped by pair (or by country, with a null importer, for `eu`) with
    the years they occur in.
    """
    gravity = gravity.filter(pl.col("iso3_o") != pl.col("iso3_d"))
    pairs = gravity.select(
        "year", exporter="iso3_o", importer="iso3_d", gravity=pl.col("fta_wto").fill_null(0).cast(pl.Int8)
    ).join(
        agreements.select("exporter", "importer", pl.col("year").cast(pl.Int32), ours="trade_agreement"),
        on=["exporter", "importer", "year"],
        how="left",
    )
    countries = (
        pl.concat(
            [
                gravity.select("year", exporter="iso3_o", gravity="eu_o"),
                gravity.select("year", exporter="iso3_d", gravity="eu_d"),
            ]
        )
        .unique(["exporter", "year"])
        .with_columns(pl.col("gravity").fill_null(0).cast(pl.Int8), importer=pl.lit(None, pl.String))
        .join(
            blocs.select(pl.col("iso3").alias("exporter"), pl.col("year").cast(pl.Int32), ours="eu"),
            on=["exporter", "year"],
            how="left",
        )
    )
    compared = pl.concat(
        [
            pairs.with_columns(feature=pl.lit("trade_agreement")),
            countries.select(pairs.columns).with_columns(feature=pl.lit("eu")),
        ]
    ).with_columns(pl.col("ours").fill_null(0).cast(pl.Int8))
    summary = compared.group_by("feature", maintain_order=True).agg(
        pair_years=pl.len(),
        agreement_rate=(pl.col("ours") == pl.col("gravity")).mean(),
        both=((pl.col("ours") == 1) & (pl.col("gravity") == 1)).sum(),
        ours_only=((pl.col("ours") == 1) & (pl.col("gravity") == 0)).sum(),
        gravity_only=((pl.col("ours") == 0) & (pl.col("gravity") == 1)).sum(),
    ).with_columns(
        # Most pair-years have no agreement in either source, which flatters the plain rate.
        positive_overlap=pl.col("both") / (pl.col("both") + pl.col("ours_only") + pl.col("gravity_only"))
    )
    disagreements = (
        compared.filter(pl.col("ours") != pl.col("gravity"))
        .group_by("feature", "exporter", "importer", "ours", "gravity")
        .agg(years=pl.col("year").sort(), n_years=pl.len())
        .sort(["feature", "n_years", "exporter", "importer"], descending=[False, True, False, False], nulls_last=True)
    )
    return summary, disagreements


if __name__ == "__main__":
    for name, path in build().items():
        print(f"{name}: {path}")
