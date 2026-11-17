# 27: Misleading Products evidence

Researched 2026-11-17 for issue #27 (child of map #24). This note gathers evidence only. The decision on what to exclude, flag or down-weight belongs to #34 "Product exclusions and re-export hubs".

## Bottom line

1. **Others exclude these Products.** ITC's Export Potential Map, the closest published analogue to our Export Gap, drops almost all of Sector 27 (it keeps only 2712 and 2716), drops monetary gold and precious-metal scrap, and drops arms (93) and antiques (9706). It also filters re-exports out of the supply side for chapters 73 and 84–90 ([ITC 2023](#itc)).
2. **UK official statistics treat precious metals as noise.** ONS headline trade figures "exclude non-monetary gold and other precious metals", because those flows are "large and highly volatile, distorting underlying trends" ([ONS 2022](#ons-pm)). HMRC's customs data (the basis of Comtrade and BACI) record physical movement. ONS's Balance of Payments records change of ownership. The two are expected to diverge for gold ([HMRC 2014](#hmrc-gold)).
3. **In our own data, gold is the most unstable UK Flow, and the sources disagree about it.**
   - UK 7108 exports in BACI swing between $10bn and $41bn a year (2012–2024). They fell from $33bn in 2023 to $12bn in 2024.
   - For 2023, HMRC reports £52.5bn of 7108 exports, £23.5bn of it to Switzerland. BACI reports $33.1bn, only $2.4bn of it to Switzerland. BACI's mirror reconciliation has moved most of the UK–Swiss gold elsewhere or out of the data.
4. **The suspect groups hold about 18% of the positive Export Gap, and about 32% once re-export-hub Markets are added.**
   - By value: Sector 27 holds 6.6%, Sector 71 holds 4.2%, medicaments (3004) 5.0% and blood/vaccines (3002) 1.4%.
   - By rank: these Product groups take 37 of the top-100 Opportunities and 281 of the top-1000. Adding eight hub Markets raises that to 47 and 423.
5. **Medicaments are the single largest group (5.0% of the positive gap, 9 of the top-100), led by 300490 → China ($4.3bn).** Primary sources document that pharma trade values are shaped by transfer pricing and contract manufacturing (Bernard, Jensen & Schott 2006; Irish Fiscal Advisory Council 2015). Unlike gold and oil, though, no statistical agency or potential model excludes medicaments.
6. **Oil and gas Opportunities often sit where the UK ships nothing.** Examples: crude 270900 → Ireland (a $2.7bn gap on $0.45M actual), and gas 271121 → France, China, Germany and Italy (zero actual). ONS shows UK gas exports go by pipeline to Belgium, Ireland and the Netherlands. Pipeline geography is not in the model.
7. **Confidential codes are not a material problem in BACI HS12.** BACI drops "commodities NES" and our Product list has no chapter 77/98/99 codes. The 102 Products with zero UK exports to every Market hold only $0.9bn (0.2%) of the positive gap. That set is where any HMRC full suppressions would show up.
8. **There is a gap in the evidence.** We found no primary source saying gravity models fit pipeline gas or electricity poorly. The energy-gravity papers found fit such models successfully once infrastructure is controlled for.

## The question

Which Products distort Export Gaps, and how do others treat them? The groups in scope are:

- non-monetary gold (7108) and other precious metals
- crude oil and natural gas
- re-export and entrepôt hubs
- confidential codes
- medicaments driven by transfer pricing

Quantify each group's share of the positive Export Gap from the v1 artefact.

## Evidence from the artefact

**Source.** Artefact `artefacts/20261117T022550Z`, file `baseline.parquet`, joined to `products.parquet`. This is the Baseline for 2024 and covers 1,169,550 UK Product–Market pairs.

**Totals**
- Export Potential sums to $691.8bn and Actual Exports to $407.2bn.
- 1,110,185 pairs (95%) have a positive Export Gap, summing to **$429.2bn**. Negative gaps sum to −$144.7bn.
- The top-100 Opportunities hold 15.5% of the positive gap, and the top-1000 hold 39.5%.

**How to read the table.** "Top-100/1000" counts how many of the largest positive-gap pairs (Opportunities) fall in each group. Groups are by HS4 unless stated. Hub Markets overlap with the Product groups.

| Group | Positive pairs | Positive gap $bn | % of positive gap | In top-100 | In top-1000 | % of top-1000 gap | UK actual exports $bn |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-monetary gold 7108 | 658 | 5.74 | 1.3% | 4 | 25 | 2.7% | 11.89 |
| Other precious metals 7106–7112, 7115, 7118 (excl. 7108) | 4,575 | 6.85 | 1.6% | 1 | 34 | 2.4% | 6.53 |
| Rest of Sector 71 (stones, jewellery, imitation) | 5,853 | 5.59 | 1.3% | 3 | 24 | 2.5% | 3.21 |
| Crude oil 2709 | 213 | 6.43 | 1.5% | 4 | 19 | 3.4% | 16.37 |
| Refined oil 2710 | 200 | 6.72 | 1.6% | 5 | 30 | 3.1% | 10.84 |
| Natural gas / LPG 2711 | 1,491 | 9.57 | 2.2% | 6 | 20 | 4.7% | 6.44 |
| Electricity 2716 | 223 | 1.25 | 0.3% | 0 | 8 | 0.4% | 0.97 |
| Other Sector 27 | 6,347 | 4.58 | 1.1% | 3 | 13 | 1.4% | 2.60 |
| Medicaments 3004 | 1,605 | 21.65 | 5.0% | 9 | 72 | 10.4% | 17.91 |
| Blood, vaccines 3002 | 836 | 5.87 | 1.4% | 1 | 28 | 2.3% | 7.19 |
| Arms 93 | 3,919 | 0.85 | 0.2% | 0 | 0 | 0.0% | 0.69 |
| Art and antiques 97 | 1,449 | 2.42 | 0.6% | 1 | 8 | 0.8% | 4.51 |
| Base-metal waste and scrap (7204, 7404, 7503, 7602, 3825, 4707, 8548) | 5,436 | 7.34 | 1.7% | 1 | 30 | 2.2% | 7.18 |
| **Union of the Product groups above, excluding waste** | | | **18.1%** | **37** | **281** | | |
| Re-export hub Markets (NLD, BEL, IRL, HKG, SGP, ARE, CHE, LUX) | 35,714 | 76.19 | 17.8% | 21 | 197 | 21.2% | 109.19 |
| – Netherlands | 4,471 | 24.13 | 5.6% | 6 | 68 | 6.7% | 29.50 |
| – Belgium | 4,771 | 11.29 | 2.6% | 1 | 33 | 2.5% | 13.85 |
| – Ireland | 3,809 | 12.45 | 2.9% | 5 | 22 | 3.8% | 24.19 |
| – Hong Kong | 4,536 | 6.57 | 1.5% | 2 | 18 | 2.2% | 11.28 |
| – Singapore | 4,613 | 7.49 | 1.7% | 4 | 14 | 2.1% | 9.12 |
| – UAE | 3,665 | 2.73 | 0.6% | 0 | 7 | 0.4% | 12.23 |
| – Switzerland | 4,822 | 10.21 | 2.4% | 3 | 33 | 3.3% | 8.45 |
| – Luxembourg | 5,027 | 1.33 | 0.3% | 0 | 2 | 0.2% | 0.59 |
| **Union of the Product groups or a hub Market** | | | **31.8%** | **47** | **423** | | |

**By Sector**
- Sector 27 holds $28.5bn of positive gap and 18 of the top-100.
- Sector 71 holds $18.2bn and 8 of the top-100. Its largest Products are 711319 jewellery ($4.2bn), 710813 semi-manufactured gold ($3.1bn) and 710812 unwrought gold ($2.7bn).
- Sector 30 holds $29.9bn and 10 of the top-100.

**Negative gaps.** These groups also hold large negative Export Gaps, so the model under-predicts the UK in them as often as it over-predicts:

| Group | Export Potential $bn | Actual Exports $bn | Negative gap $bn |
|---|---:|---:|---:|
| 7108 | 9.1 | 11.9 | −8.5 |
| 2709 | 12.2 | 16.4 | −10.6 |
| Sector 71 | 27.1 | 21.6 | −12.8 |

The negative gaps sit where the UK actually ships: gold to Switzerland, the UAE and Hong Kong, and crude oil to Germany and the Netherlands. The positive gaps sit elsewhere:
- 7108: Germany, Singapore, China
- 2709: Ireland, Czechia, Belgium
- 2711: France, the Netherlands, China

The model spreads these Flows across Markets by fundamentals, while the real trade is concentrated by bullion-market and pipeline geography.

**Volatility.** UK exports in BACI, $bn (from `data/derived/baci_hs12_V202601`):

| Year | 7108 | Rest of 71 | 2709 | 2711 |
|---|---:|---:|---:|---:|
| 2012 | 10.1 | 18.5 | 21.9 | 6.1 |
| 2014 | 40.7 | 12.8 | 22.6 | 5.3 |
| 2016 | 26.9 | 10.2 | 12.7 | 3.5 |
| 2020 | 18.2 | 15.3 | 16.3 | 2.3 |
| 2022 | 33.0 | 14.9 | 26.4 | 24.0 |
| 2023 | 33.1 | 10.9 | 17.7 | 12.3 |
| 2024 | 11.9 | 9.7 | 16.4 | 6.4 |

Gas quadrupled in the 2022 price shock. Gold moves by a factor of three to four from year to year.

**HMRC against BACI, 2023.** Compared using `data/derived/hmrc_ots_2023.parquet` (£) and BACI ($):

| Product | HMRC £bn | BACI $bn | Largest Markets in HMRC | Largest Markets in BACI |
|---|---:|---:|---|---|
| 7108 | 52.5 | 33.1 | CHE 23.5, CHN 14.0, HKG 5.4 | CHN 17.5, HKG 8.2, CHE 2.4 |
| 2709 | 17.0 | 17.7 | NLD 7.2, DEU 2.0, POL 1.6 | NLD 4.5, DEU 4.4, SWE 1.6 |
| 2711 | 6.9 | 12.3 | BEL 4.0, IRL 1.5, NLD 1.0 | NLD 4.3, BEL 4.0, IRL 3.5 |
| 3004 | 15.6 | 16.7 | USA 3.8, BEL 2.2, IRL 1.4 | USA 3.6, DEU 1.6, IRL 1.6 |

Total exports are similar: £425bn in HMRC and $443bn in BACI. Medicaments agree closely. Gold disagrees most, in total and by Market: the UK–Swiss gold flow is almost absent from BACI. Gas and crude disagree on Market split, which is consistent with mirror-flow reconciliation of pipeline and tanker trade.

**Notable single pairs in the top 30 Opportunities**

| Rank | Pair | Export Potential | Actual Exports | Export Gap |
|---:|---|---:|---:|---:|
| 1 | 300490 → CHN | $4.89bn | $0.57bn | $4.31bn |
| 2 | 270900 → IRL | $2.75bn | $0.45M | $2.75bn |
| 4 | 271121 → FRA | $2.07bn | 0 | $2.07bn |
| 13 | 271121 → CHN | | 0 | $1.02bn |
| 18 | 271121 → DEU | | 0 | $0.86bn |
| 23 | 284420 → KOR (enriched uranium / plutonium) | | 0 | $0.78bn |
| 24 | 271121 → ITA | | 0 | $0.76bn |
| 28 | 740400 copper scrap → DEU | | | $0.67bn |

Uranium is not in any group above but is another strategic, licensed Product.

**Confidential codes**
- The BACI HS12 Product list in `products.parquet` contains no chapter 77, 98 or 99 codes.
- 102 Products have zero UK exports to every Market. Together they hold $0.91bn of positive gap (0.2%), led by 890130 refrigerated vessels ($0.42bn). HMRC Level-1 suppressions would appear in this set, but we cannot tell a suppression from a true zero without HMRC's suppression list.

## How others treat these Products

### ITC Export Potential Map {#itc}

**Source.** ITC, *Export Potential and Diversification Assessments: A methodology to identify export opportunities* (2023 revision of Decreux & Spies 2016): https://umbraco.exportpotential.intracen.org/media/cklh2pi5/epa-methodology_230627.pdf

- **Formula.** Potential = supply × demand × ease. It is "inspired by a gravity model specified at the product level". Unrealised potential is potential minus min(potential, actual), which is conceptually our positive Export Gap.
- **Coverage.** Products "based on mineral fuels or mineral oils, or are not in line with ITC's work programme (e.g. wastes, antiques, etc.) are excluded". Annex V lists:
  - chapter 27 except 2712 and 2716
  - "Monetary gold; coins; waste and scrap of gold, silver & platinum"
  - precious metals unwrought or in powder (for the diversification indicator)
  - chapter 93 arms
  - war materiel (8710, 890610)
  - 9706 antiques
  - tobacco (24), ores (25–26), and various wastes, including waste pharmaceuticals (300692)
- **Re-exports.** "Declared exports often comprise re-exports that are not linked to any capacity of the country to produce the good". For chapters 73 and 84–90, a product is dropped for an exporter when imports are at least 2× exports and other conditions hold (for example imports at least 20× exports). Supply is also corrected because "global tariff advantages or re-exports may overestimate a country's capacity to export".
- **Raw commodities.** "Mineral primary products (HS Chapters 25 and 26) are excluded from the ease calculation as trade in raw commodities is usually not controlled by the country itself but by large foreign companies."
- **Pharma.** No exclusion of medicaments or of pharma transfer pricing.
- **Confidential data.** No explicit rule. Mirror data are used where reporting is unreliable (Annex VI).
- The 2016 original PDF was not located; this is its successor.

### CEPII BACI

**Gaulier & Zignago (2010), CEPII WP 2010-23:** https://imap.cepii.fr/PDF_PUB/wp/2010/wp2010-23.pdf

- Mirror flows are reconciled with reporter-reliability weights, and CIF import values are converted to FOB.
- **Confidentiality.** Confidential trade "will be included at the higher commodity level… (and sometimes via the use of a specific item of the product classification)".
- **Re-exports.** "Re-exportations for Hong Kong and USA are also dropped since there is no way to know the final destination of the flow." Re-exports of other hubs (Netherlands, Belgium, Singapore) stay in.
- **Unspecified codes.** "Commodities NES" is dropped ("mostly specific commodities such as military equipment"). Area NES flows are reallocated to partners (about 11.5% of flows).
- **Gold.** The working paper has **no gold-specific treatment** (full-text search), and neither do the BACI web pages (https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html). The gold disagreement we see between HMRC and BACI must therefore come from generic reconciliation, not from a gold rule.

### UN Comtrade and IMTS 2010

**IMTS 2010:** https://unstats.un.org/unsd/trade/EG-IMTS/BG-IMTS2010.pdf

- **Gold.** Non-monetary gold is "recommended for inclusion" (para 1.10). Monetary gold is excluded "since [it] is treated as a financial asset rather than a good" (para 1.45).
- **Re-exports.** Re-exports must be included and "separately identified" (para 2.17).
- **Transit.** Goods in transit are excluded, but some countries record them as imports and exports (para 1.42). This is the quasi-transit mechanism behind the Rotterdam effect.
- **Confidentiality.** Confidential data should be "reported in full detail at the next higher level of commodity aggregation" (para 10.3).

**Comtrade Plus methodology guide:** https://comtradeapi.un.org/files/v1/app/wiki/MethodologyGuideforComtradePlus.pdf
- Unmappable codes become 999999.
- "Pipelines and cables" is its own mode of transport.

**Comtrade wiki:** https://comtradewiki.azurewebsites.net/?p=444
- "Areas NES" is used for low-value or unknown partners.
- "Special Categories" (code 839) is used when a reporter "does not want the partner breakdown to be disclosed".

### ONS and HMRC {#ons-pm}

**Precious metals**
- ONS UK Trade, March 2022: "all trade figures exclude non-monetary gold and other precious metals unless otherwise stated… movements in non-monetary gold… can be large and highly volatile, distorting underlying trends". https://www.ons.gov.uk/economy/nationalaccounts/balanceofpayments/bulletins/uktrade/march2022
- ONS blog (Athow, 2020):
  - London's bullion market "can have a sizeable impact on the UK's trade figures"
  - "the gold that has been sold may not even be in the UK"
  - gold is "nearly 90%" of traded precious metals
  - https://blog.ons.gov.uk/2020/02/10/its-indestructible-but-can-we-always-believe-in-the-uk-trade-figures-with-the-disaggregated-effect-of-the-international-trade-in-non-monetary-gold/

**HMRC, *Inclusion of Non-Monetary Gold in OTS* (2014)** {#hmrc-gold} — https://www.uktradeinfo.com/media/ql5n1n3c/gold_methodology.pdf
- Non-monetary gold has been included in OTS since January 2014, backcast to 2005. The codes are 7108 11/12/13, 7109, 7112 30 and 7112 91.
- Monetary gold stays excluded.
- Gold is left out of Regional Trade Statistics, because nearly all of it would be allocated to London.
- ONS BoP records "change of ownership" while OTS records "physical movement", so "there is likely to be a divergence".

**HMRC suppressions policy** — https://www.uktradeinfo.com/trade-data/suppressions-policy/
- Level 1 is "Complete Suppression: No statistics get published for the specific commodity code". Levels 2 and 3 suppress countries and quantities.
- Suppressions are set at CN8 and roll up, with a "mask" suppression on a second code.
- The UK uses passive confidentiality (https://www.gov.uk/government/publications/uk-trade-in-goods-statistics-reasons-for-trade-in-goods-asymmetries/reasons-for-trade-in-goods-asymmetries). That page also names energy supplies such as gas and electricity as a source of asymmetries.

**Fuels.** ONS, *Trends in UK imports and exports of fuels* (2022): https://www.ons.gov.uk/economy/nationalaccounts/balanceofpayments/articles/trendsinukimportsandexportsoffuels/2022-06-29
- Crude oil went mainly to the Netherlands (£6.2bn) and China (£3.9bn).
- "Natural gas in gaseous state is imported via pipeline, primarily from Norway".
- "Almost half of the UK's gas exports in 2021 were to Ireland (47.2%)".
- The UK "re-exports imported LNG, through pipelines to Belgium, Ireland and the Netherlands".

**Rotterdam effect.** ONS, *UK Perspectives 2016: Trade with the EU and beyond*: https://www.ons.gov.uk/businessindustryandtrade/internationaltrade/articles/ukperspectives2016tradewiththeeuandbeyond/2016-05-25
- An ONS estimate found "50 per cent of all goods exports to the Netherlands were re-exported to non-EU countries", about four percentage points of UK goods exports.
- An ESCoE/ONS/HMRC project is studying the effect: https://www.escoe.ac.uk/projects/the-rotterdam-antwerp-effect-in-the-context-of-uk-trade-statistics/

### Re-export and entrepôt hubs

- **Feenstra & Hanson (2004)**, *Intermediaries in Entrepôt Trade: Hong Kong Re-Exports of Chinese Goods*, JEMS 13(1); NBER w8088 (https://nber.org/papers/w8088).
  - Hong Kong distributes "a large fraction of China's exports".
  - Re-exports are "much more expensive when they leave Hong Kong than when they enter".
- **CBS (2020):** "Nearly 60 percent of the inbound cargo [to the Netherlands] was in transit or destined for re-export". https://www.cbs.nl/en-gb/news/2020/10/60-percent-of-inbound-goods-leave-the-netherlands-again
- We found no primary source specific to Singapore, Belgium or Ireland as entrepôts. Switzerland and the UAE are hubs for gold specifically: in our data they are the largest Markets for actual UK 7108 exports.

### Medicaments and transfer pricing

- **Bernard, Jensen & Schott (2006)**, NBER w12493 (https://www.nber.org/papers/w12493). Arm's-length export prices are "substantially larger" than related-party prices, and the gap is larger for differentiated goods and for low-tax destinations.
- **Guvenen, Mataloni, Rassier & Ruhl (2022)**, AER 112(6). Profit shifting distorts measured US trade and output. https://www.aeaweb.org/doi/10.1257/aer.20190285
- **Irish Fiscal Advisory Council (2015), Box A.**
  - Under contract manufacturing, "The sale of the good is recorded as an Irish export of goods".
  - The IMF notes "the issue is most important in the pharmaceutical sector".
  - https://www.fiscalcouncil.ie/wp-content/uploads/2015/06/Box-A.pdf
- **Setser (CFR, commentary, not peer-reviewed, 2019):** pharma trade "seems motivated in large part by the ability to use transfer pricing to shift profits". https://www.cfr.org/articles/when-tax-drives-trade-data
- **Not found:** no IMF or OECD paper specific to HS 3002/3004 and profit shifting, and no potential model that excludes medicaments.

### Pipeline and grid trade in gravity models

**No source found.** No primary source was found saying gravity models fit pipeline gas or electricity poorly. The literature points the other way:

- Batalla-Bejerano, Paniagua & Trujillo-Baute (2019, *EEEP* 8(2)) find a gravity model "provides a good fit for cross-border electricity trade data", while noting that trade "is restricted by the existing cross-border transmission infrastructure". https://www.uv.es/jorpaso2/Docs/eeep8-2-Paniagua.pdf
- Farag & Zaki (2024, *The World Economy* 47(2)) estimate separate PPML gravity models for pipeline gas and LNG. https://ideas.repec.org/a/bla/worlde/v47y2024i2p806-836.html

**Where the case for exclusion rests instead:**
- ITC's chapter 27 exclusion
- BACI dropping kWh quantities
- our own evidence that 2711 Export Potential lands on Markets with no UK pipeline connection

## Open points for #34

- The groups overlap: much gold goes to CHE, ARE and HKG, and gas to NLD, BEL and IRL. Exclusion choices should be compared on the union, not group by group.
- These Products carry both large positive and large negative gaps. Excluding them from rankings and excluding them from training are different choices with different effects on the model.
- Medicaments are the largest single group, but no one excludes them. They are a candidate for an Advisor-facing caveat rather than an exclusion.
- Using BACI rather than HMRC for UK actuals moves gold and gas between Markets. Any rule keyed on Market (for example Switzerland for gold) behaves differently under each source.

## Reproduce

The scripts were throwaway. They read the artefact files and BACI/HMRC derived Parquet with Polars, for example:

```python
b = pl.read_parquet(a + "baseline.parquet").join(pl.read_parquet(a + "products.parquet"), on="hs6")
pos = b.filter(pl.col("export_gap_usd") > 0).sort("export_gap_usd", descending=True).with_row_index("rank")
```

A group's share is the sum of its `export_gap_usd` over `pos`, divided by the total. Top-k counts are rows with `rank < k`.
