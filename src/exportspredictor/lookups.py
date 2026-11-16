"""Product and Market lookups, and Product search.

The lookup tables have the same shape for the fixture and for real data:

- Products: `hs6` (zero-padded HS6 code, 2012 edition), `description`, `sector` (HS2 chapter).
- Markets: `iso3`, `name`, `region`.

Real lookups are built from BACI's code files by `exportspredictor.baci`.
"""

import re

import polars as pl

# Everyday words Advisors type that the official HS 2012 descriptions never use.
# Each maps to the Products it should find, best match first.
PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "smartphone": ("851712",),
    "mobile phone": ("851712",),
    "cell phone": ("851712",),
    "laptop": ("847130",),
    "whisky": ("220830",),
    "scotch": ("220830",),
    "gin": ("220850",),
    "car": ("870322", "870323", "870324", "870332", "870333"),
    "jet engine": ("841112", "841191"),
    "aircraft": ("880240",),
    "vaccine": ("300220",),
    "medicine": ("300490",),
    "drug": ("300490",),
}

_DEFAULT_LIMIT = 20


def search_products(products: pl.DataFrame, query: str, limit: int = _DEFAULT_LIMIT) -> pl.DataFrame:
    """Products matching `query`, best first. No match (or a blank query) gives an empty frame.

    - Digits match HS codes by prefix: "851712" finds that Product, "22" lists its Sector.
    - Words match the description when every word starts a word in it (case-insensitive),
      plus any Products named by an everyday alias such as "smartphone".
    """
    query = query.strip().lower()
    if not query:
        return products.clear()
    if query.isdigit():
        return products.filter(pl.col("hs6").str.starts_with(query)).sort("hs6").head(limit)

    # Whole words only, with an optional plural "s", so "car" does not match "carbon".
    aliased = [
        hs6
        for alias, codes in PRODUCT_ALIASES.items()
        if re.search(rf"\b{re.escape(alias)}s?\b", query)
        for hs6 in codes
    ]
    words = re.findall(r"[a-z0-9]+", query)
    if not words and not aliased:
        return products.clear()
    # Each word must start a word of the description: "whisk" finds "Whiskies".
    described = pl.all_horizontal(
        pl.col("description").str.contains(rf"(?i)\b{re.escape(word)}") for word in words
    ) if words else pl.lit(False)
    rank = {hs6: i for i, hs6 in enumerate(dict.fromkeys(aliased))}
    return (
        products.filter(pl.col("hs6").is_in(list(rank)) | described)
        .with_columns(_rank=pl.col("hs6").replace_strict(rank, default=len(rank), return_dtype=pl.Int32))
        .sort("_rank", "hs6")
        .drop("_rank")
        .head(limit)
    )


# BACI keys Taiwan as "Other Asia, nes" (`S19`); every other table uses its ISO3 and name.
_STANDARD_NAMES = {"TWN": "Taiwan"}


def market_lookup(markets: pl.DataFrame, country_map: pl.DataFrame) -> pl.DataFrame:
    """The Market lookup (`iso3`, `name`, `region`) keyed by standard ISO3, as the features are.

    `markets` is BACI's lookup (keyed by BACI's iso3, e.g. `S19`); `country_map` maps
    `baci_iso3` to `iso3`.
    """
    to_iso3 = dict(country_map.select("baci_iso3", "iso3").unique().iter_rows())
    return (
        markets.with_columns(iso3=pl.col("iso3").replace(to_iso3))
        .with_columns(name=pl.col("iso3").replace_strict(_STANDARD_NAMES, default=pl.col("name")))
        .select("iso3", "name", "region")
        .unique("iso3", keep="first", maintain_order=True)
    )
