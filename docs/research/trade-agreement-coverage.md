# Research 03: Does CEPII Gravity cover Trade Agreements and Trading Blocs, 2012 to the latest BACI year?

Ticket: `issues/03-gravity-policy-coverage.md`. Researched 2026-11-16.

## Bottom line

1. **No, not fully.** The current CEPII Gravity release (V202211) has policy variables only up to **2021**, and its BACI trade flows stop at **2020**. BACI HS12 now runs to **2024** (V202601), so Gravity leaves 2022–2024 uncovered.
2. Within 2012–2021 the coding is sound. The EU–Japan EPA appears from 2019, the UK's EU exit from 2020 (`eu`), and the UK–EU TCA and the UK–Japan agreement from 2021. Coverage ends before UK–Australia and UK–NZ (both 31 May 2023) and UK CPTPP accession (15 Dec 2024), so Gravity never records them.
3. The policy variables are `gatt_o/_d`, `wto_o/_d`, `eu_o/_d`, `fta_wto`, `fta_wto_raw`, `rta_coverage` and `rta_type`. All are populated through 2021. Gravity V202211 has no variable called `rta`.
4. The **WTO RTA-IS "Export all RTAs" file** is current: in the copy downloaded 2026-11-16, the latest entry-into-force date is 2026-10-01. It has exact in-force dates and per-member entry and exit dates for every agreement this project needs. DESTA 2.3 (Sept 2025) gives years only, and its in-force year is blank for UK–NZ and UK CPTPP accession.
5. **Recommendation:** build pair-year Trade Agreement and Trading Bloc features from the WTO RTA-IS export, using Gravity's 1 July rule. Use Gravity 2012–2021 only to check the build, and DESTA only for optional agreement-depth scores.

## 1. CEPII Gravity: version and time coverage

| Fact | Value | Source |
|---|---|---|
| Current release | V202211 (older releases are kept "for reproducibility") | https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8 |
| Coverage stated on the product page | "1948 to 2020" | same page |
| Coverage stated in the documentation | "Data spans from 1948 to 2019" (documentation dated 12 Oct 2023; the two pages disagree) | https://www.cepii.fr/DATA_DOWNLOAD/gravity/doc/Gravity_documentation.pdf |
| **Actual content of `Gravity_V202211.csv`** (checked by me) | Rows for 1948–**2021**. Policy variables are populated in 2021. `tradeflow_baci` is filled through **2020** and empty in 2021. There are no rows for 2022 or later. | https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip (Last-Modified 15 Apr 2024; files inside dated 2 Jan 2023) |
| Newer Gravity release? | None found. A web search turned up no 2024–2026 release, and the product page still lists V202211 as current. | product page above |
| Latest BACI | V202601 (released 30 Jan 2026); HS12 covers **2012–2024** | https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37 |

**Gap:** Gravity has no policy variables for 2022–2024, which BACI covers. It also has no data for 2025 onward, which Scenarios need.

## 2. Gravity variables for Trade Agreements and Trading Blocs

The definitions come from the documentation, §7.1–7.3 (https://www.cepii.fr/DATA_DOWNLOAD/gravity/doc/Gravity_documentation.pdf). The last year shown for each variable is what I found in the V202211 CSV.

| Variable | Level | Definition (documentation) | Source (documentation) | Last year populated |
|---|---|---|---|---|
| `gatt_o`, `gatt_d` | unilateral | 1 if the country is a GATT member | WTO list of GATT members | 2021 |
| `wto_o`, `wto_d` | unilateral | 1 if the country is a WTO member | WTO list of members | 2021 |
| `eu_o`, `eu_d` | unilateral | 1 if the country is an EU member; "set to one if countries enter between 1st January and 31st December of the given year" | European Union website | 2021 |
| `fta_wto` | bilateral | 1 if the pair is in a regional trade agreement, "source WTO supplemented by Thierry Mayer" | WTO RTA database | 2021 |
| `fta_wto_raw` | bilateral | Same, from the WTO only, with no corrections | WTO RTA database | 2021 |
| `rta_coverage` | bilateral | 0 none, 1 goods, 2 services, 3 goods and services, per the documentation. The CSV label file uses a different code order: `1=Goods, 2=Goods & Services, 3=Services`, so trust `Label_rta_coverage_V202211.csv`. | WTO RTA database | 2021 |
| `rta_type` | bilateral | PSA / FTA / CU / EIA and combinations. CSV codes: 1 CU, 2 CU & EIA, 3 EIA, 4 FTA, 5 FTA & EIA, 7 PSA & EIA (`Label_rta_type_V202211.csv`) | WTO RTA database | 2021 |

Conventions and caveats from the documentation:
- A pair counts as in an RTA in a given year if the agreement was in force **before 1 July** of that year. Pairs of a country with itself are set to 0.
- The documentation's sources table still lists `rta`, `rta_bergstrand`, `rta_type_bergstrand` and `gsp`. **None of these columns exists** in the V202211 CSV (I read the header), so treat the table as a leftover from an older version.
- `fta_wto_raw` contains errors that `fta_wto` corrects. For example, GBR–CAN has `fta_wto_raw=1` (type FTA) for 2010–2017, before CETA, while `fta_wto=0`. **Use `fta_wto`.**
- Pre-unification Germany (`DEU.1`) also produces rows with iso3 "DEU". Filter on `country_exists_o/_d`, or join on `country_id`, to avoid duplicates.

### Spot checks of UK pairs and EU–Japan in V202211 (my extraction, same CSV)

| Pair / event | What Gravity shows | Correct? |
|---|---|---|
| EU–Japan EPA (FRA–JPN, GBR–JPN) | `fta_wto` 0 through 2018, 1 from **2019**, `rta_type`=FTA & EIA | Yes (in force 1 Feb 2019, before 1 July) |
| UK leaves the EU | GBR `eu_o` = 1 through 2019, **0 from 2020** | Yes (exit 31 Jan 2020). The transition period is not reflected in `eu`. |
| UK–EU in 2020 (transition) | GBR–FRA and GBR–DEU `fta_wto`=1, `rta_type`=**CU** | Yes: the WTO record keeps the UK in the EU customs union until 31 Dec 2020 |
| UK–EU TCA | GBR–FRA and GBR–DEU 2021 `rta_type`=**FTA & EIA**, coverage goods and services | Yes |
| UK–Japan (CEPA) | GBR–JPN `fta_wto`=1 in 2021 | Yes |
| UK–Singapore and UK–Vietnam rollovers | GBR–SGP =1 from 2020 (EU–SG); GBR–VNM =1 in 2021 | Consistent with the 1 July rule |
| UK–Australia, UK–NZ | `fta_wto`=0 through 2021 (the last year in the file) | Correct for those years, but the 2023 agreements fall **outside coverage** |
| UK CPTPP accession | Not representable (Gravity ends in 2021) | **Gap** |

## 3. In-force dates (official sources)

| Agreement | In force | Official source | WTO RTA-IS record (export file) |
|---|---|---|---|
| EU–Japan EPA | **1 Feb 2019** | European Commission: "entered into force on 1 February 2019", https://policy.trade.ec.europa.eu/eu-trade-relationships-country-and-region/countries-and-regions/japan/eu-japan-agreement_en | RTA ID 836 "EU - Japan", in force 2019-02-01, with UK membership ending 31-Dec-2020 |
| UK–EU Trade and Cooperation Agreement | **Provisionally applied 1 Jan 2021; formally in force 1 May 2021** | UK Government implementation report, para 8: "It took provisional effect on 1 January 2021, and formally entered into force on 1 May 2021", https://www.gov.uk/government/publications/trade-and-cooperation-agreement-implementation-report-january-2021-to-december-2022/uk-eu-trade-and-cooperation-agreement-implementation-report-1-january-2021-to-31-december-2022 | RTA ID 1137 "EU - United Kingdom", in force 2021-01-01 (WTO uses the provisional application date) |
| UK–Australia FTA | **31 May 2023** | GOV.UK, 31 May 2023: "The UK's trade deals with Australia and New Zealand come into effect today", https://www.gov.uk/government/news/uks-first-post-brexit-trade-deals-to-go-live-at-midnight-on-wednesday | RTA ID 1152, signed 2021-12-16, in force 2023-05-31 |
| UK–New Zealand FTA | **31 May 2023** | Same GOV.UK page | RTA ID 1161, signed 2022-02-28, in force 2023-05-31 |
| UK accession to CPTPP | **15 Dec 2024** with Japan, Singapore, Chile, NZ, Vietnam, Peru, Malaysia and Brunei; **24 Dec 2024** with Australia; **22 Jun 2026** with Mexico; **1 Sep 2026** with Canada | GOV.UK CPTPP collection, https://www.gov.uk/government/collections/the-uk-and-the-comprehensive-and-progressive-agreement-for-trans-pacific-partnershipcptpp | RTA ID 1218 "CPTPP - Accession of the United Kingdom", in force 2024-12-15; Australia 24-Dec-2024, Canada 01-Sep-2026, Mexico 22-Jun-2026. ID card: https://rtais.wto.org/UI/PublicShowMemberRTAIDCard.aspx?rtaid=1218 |

The WTO RTA-IS records above were read from the "Export all RTAs" file at https://rtais.wto.org/UI/ExportAllRTAList.aspx (linked from https://rtais.wto.org/UI/PublicMaintainRTAHome.aspx), downloaded 2026-11-16. I opened only the ID card for 1218 by its URL. The other IDs come from the export, and their ID cards follow the same `?rtaid=` URL pattern.

Under Gravity's 1 July rule, these dates map to the following first in-agreement years: EU–Japan 2019, TCA 2021, UK–AUS and UK–NZ **2023**. UK–CPTPP becomes 2025 for the Dec-2024 members, 2026 for Mexico (22 June is before 1 July) and **2027** for Canada (1 Sep 2026 is after 1 July).

## 4. Filling the gap: WTO RTA-IS vs DESTA

### WTO RTA-IS (rtais.wto.org)
- The export (https://rtais.wto.org/UI/ExportAllRTAList.aspx) is an .xlsx with 662 RTAs. Columns include `RTA ID`, `RTA Name`, `Coverage`, `Type`, `Status`, `Date of Signature (G/S)`, `Date of Entry into Force (G/S)`, `Inactive Date`, `Accession?`, `Current signatories`, `Original signatories` and **`Specific Entry/Exit dates`**. The `Specific Entry/Exit dates` column holds per-member join and leave dates, for example the UK leaving EU–Japan on 31-Dec-2020 and Canada joining UK–CPTPP on 01-Sep-2026.
- It is current: the latest entry-into-force date in the export is 2026-10-01, and it already includes the 2026 Canada and Mexico ratifications.
- It is the source Gravity itself uses for `fta_wto`, `rta_type` and `rta_coverage` (Gravity documentation §7.2), so a rebuild from it stays consistent with the 2012–2021 values.
- Trading Bloc membership can also be derived from it. "EU Treaty" (ID 120) and the "EU (n) Enlargement" records (for example ID 869 "EU (28) Enlargement", UK exit 31-Dec-2020) give EU and customs-union membership. ID 640 (CPTPP) together with 1218 gives CPTPP membership by date.
- Limitations: it lists agreements as notified to the WTO. It cannot model unnotified agreements or ones that are not yet in force. Gravity's documentation also warns that the WTO data does not capture how substantial each agreement is.

### DESTA (designoftradeagreements.org)
- The current version is 2.3, "Current version: September 2025", https://www.designoftradeagreements.org/downloads/. It includes a treaty list, a treaty list in dyadic form, a dyadic withdrawals file, content coding and depth indices.
- My check of `desta_list_of_treaties_02_03.csv` and `_dyads.csv` from that page:
  - Fields are **years only** (`year`, `entryforceyear`), not dates.
  - UK–Australia has `entryforceyear` 2023.
  - **UK–NZ has a blank `entryforceyear`.**
  - **UK CPTPP accession (989+1) has a blank `entryforceyear`**, and the dyads list all 11 CPTPP partners with no per-partner dates.
  - UK–Australia shows `year`=2023, while the WTO gives signature as 2021-12-16. I could not confirm what DESTA's `year` field means without the 2.x codebook, so I cannot explain the mismatch.
  - It does record the Brexit withdrawals (dyadic withdrawal file, 2020) and the UK rollover agreements (2021).
- DESTA's strength is **agreement depth/design** (the `desta_indices_version_02_03` file), not timing.

## 5. Recommended source combination

- **Trade Agreement features** (pair-year `in_agreement`, `agreement_type`, `agreement_coverage`). Build them from the WTO RTA-IS export:
  - Expand each RTA's member list, applying `Specific Entry/Exit dates`, into ordered country pairs.
  - Apply Gravity's rule (in force before 1 July counts for that year) for 2012–2024 and onward.
  - Check the rebuild against Gravity V202211 `fta_wto` / `rta_type` / `rta_coverage` for 2012–2021.
  - Scenarios then flip the same features for a hypothetical UK–X pair.
- **Trading Bloc features** (`eu_o/_d`, `cptpp_o/_d`, optionally `customs_union` from `rta_type` = CU). Derive these from the same WTO export (EU Treaty and Enlargement records, CPTPP 640 + 1218) and cross-check `eu` against Gravity `eu_o/_d` through 2021. Note that Gravity codes the UK as non-EU in 2020 but still in the EU customs union for 2020 (transition period). Decide explicitly which flag an "EU re-join" Scenario flips.
- **Optional:** add DESTA 2.3 depth indices as an agreement-depth feature joined by agreement. Do not use DESTA for in-force timing.
- **Gravity** is still useful for the non-policy covariates (distance, language and so on), and for `gatt`/`wto` up to 2021. Membership after 2021 would need the WTO members list, which I did not check here.

## Not confirmed

- I found no newer Gravity release than V202211. If CEPII publishes one, the gap shrinks, but none appears on the product page.
- The meaning of DESTA's `year` field (signature vs other) was not checked against the 2.x codebook.
- I did not open the WTO ID cards for 836, 1137, 1152, 1161 and 640 individually. Their data comes from the official WTO export file above.
