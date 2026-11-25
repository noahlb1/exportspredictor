"""Sector names: the HS 2012 chapter titles, shortened where long, so a Sector reads as words, not an HS2 number.

A static table, not part of the artefact: a Sector is the `sector` column (the HS2 chapter, e.g. "22") of the
Product lookup, and its name is looked up here. `with_sector_names` adds a `sector_name` column to any table with a
`sector` column; `sectors` lists each Sector that has Products in a Product lookup, by code and name. Chapter 77 is
reserved in the Harmonized System and has no Products.
"""

from collections.abc import Sequence

import polars as pl

SECTOR_NAMES: dict[str, str] = {
    "01": "Live animals",
    "02": "Meat and edible offal",
    "03": "Fish and seafood",
    "04": "Dairy, eggs and honey",
    "05": "Other animal products",
    "06": "Live plants and cut flowers",
    "07": "Vegetables",
    "08": "Fruit and nuts",
    "09": "Coffee, tea and spices",
    "10": "Cereals",
    "11": "Milling products, malt and starches",
    "12": "Oil seeds and oleaginous fruits",
    "13": "Lac, gums and resins",
    "14": "Vegetable plaiting materials",
    "15": "Animal and vegetable fats and oils",
    "16": "Meat and fish preparations",
    "17": "Sugars and sugar confectionery",
    "18": "Cocoa and cocoa preparations",
    "19": "Cereal, flour and bakery preparations",
    "20": "Vegetable, fruit and nut preparations",
    "21": "Miscellaneous edible preparations",
    "22": "Beverages, spirits and vinegar",
    "23": "Food residues and animal feed",
    "24": "Tobacco",
    "25": "Salt, sulphur, stone, lime and cement",
    "26": "Ores, slag and ash",
    "27": "Mineral fuels and oils",
    "28": "Inorganic chemicals",
    "29": "Organic chemicals",
    "30": "Pharmaceutical products",
    "31": "Fertilisers",
    "32": "Tanning and dyeing extracts, paints and inks",
    "33": "Essential oils, perfumery and cosmetics",
    "34": "Soap, waxes and polishes",
    "35": "Albuminoidal substances, glues and enzymes",
    "36": "Explosives, matches and pyrotechnics",
    "37": "Photographic and cinematographic goods",
    "38": "Miscellaneous chemical products",
    "39": "Plastics",
    "40": "Rubber",
    "41": "Raw hides, skins and leather",
    "42": "Leather articles, handbags and travel goods",
    "43": "Furskins and artificial fur",
    "44": "Wood and wood charcoal",
    "45": "Cork",
    "46": "Straw, basketware and wickerwork",
    "47": "Wood pulp and recovered paper",
    "48": "Paper and paperboard",
    "49": "Printed books, newspapers and pictures",
    "50": "Silk",
    "51": "Wool and animal hair",
    "52": "Cotton",
    "53": "Other vegetable textile fibres",
    "54": "Man-made filaments",
    "55": "Man-made staple fibres",
    "56": "Wadding, felt, twine and ropes",
    "57": "Carpets",
    "58": "Special woven fabrics, lace and embroidery",
    "59": "Coated and laminated textiles",
    "60": "Knitted or crocheted fabrics",
    "61": "Clothing, knitted or crocheted",
    "62": "Clothing, not knitted or crocheted",
    "63": "Other made-up textiles and worn clothing",
    "64": "Footwear",
    "65": "Headgear",
    "66": "Umbrellas and walking sticks",
    "67": "Feathers, artificial flowers and wigs",
    "68": "Articles of stone, plaster and cement",
    "69": "Ceramic products",
    "70": "Glass and glassware",
    "71": "Pearls, precious stones and metals",
    "72": "Iron and steel",
    "73": "Articles of iron or steel",
    "74": "Copper",
    "75": "Nickel",
    "76": "Aluminium",
    "78": "Lead",
    "79": "Zinc",
    "80": "Tin",
    "81": "Other base metals and cermets",
    "82": "Tools and cutlery of base metal",
    "83": "Miscellaneous articles of base metal",
    "84": "Machinery and mechanical appliances",
    "85": "Electrical machinery and equipment",
    "86": "Railway locomotives and rolling stock",
    "87": "Vehicles other than railway",
    "88": "Aircraft and spacecraft",
    "89": "Ships and boats",
    "90": "Optical, medical and precision instruments",
    "91": "Clocks and watches",
    "92": "Musical instruments",
    "93": "Arms and ammunition",
    "94": "Furniture, bedding and lighting",
    "95": "Toys, games and sports equipment",
    "96": "Miscellaneous manufactured articles",
    "97": "Works of art and antiques",
}


def sector_of(hs6: str = "hs6") -> pl.Expr:
    """The Sector of each Product in column `hs6`: the first two digits of its HS6 code (e.g. "22" for "220830")."""
    return pl.col(hs6).cast(pl.String).str.slice(0, 2)


def with_sector_names[T: (pl.DataFrame, pl.LazyFrame)](table: T) -> T:
    """`table` with a `sector_name` column beside its `sector` column (the code where no name is known)."""
    return table.with_columns(
        sector_name=pl.col("sector").replace_strict(SECTOR_NAMES, default=pl.col("sector"), return_dtype=pl.String)
    )


def sectors(products: pl.DataFrame) -> pl.DataFrame:
    """Each Sector that has Products in a Product lookup, by code (`sector`) and name (`sector_name`), in code order."""
    return with_sector_names(products.select("sector").unique().sort("sector"))


def in_sectors(codes: Sequence[str] | None) -> pl.Expr:
    """True where column `hs6`'s Product is in one of the Sectors `codes`; True everywhere if none are named."""
    if not codes:
        return pl.lit(True)
    return sector_of("hs6").is_in(list(codes))
