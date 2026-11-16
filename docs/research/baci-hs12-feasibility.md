# Research 02: BACI HS12 feasibility, and HMRC OTS as a UK cross-check

Ticket: `issues/02-baci-hs12-feasibility.md`. Researched 2026-11-16. Nothing was bulk-downloaded. The zip contents and sample rows come from HTTP range reads of the zip's central directory plus the first ~4 MB of three year files (about 12 MB fetched in total).

## Bottom line

1. **Coverage:** the latest release is BACI **V202601** (released Jan 2026). HS12 covers **2012–2024**, with 2024 provisional. **2025 is not in BACI.** It is due with the next release (BACI updates every January, so ~Jan 2027).
2. **Products and countries:** the HS12 product file lists **5,202 HS6 codes**, and ~5,198 of them appear in the flows. There are **238 country codes** (some historical), with ~**226 exporters** per year. BACI itself rounds these to "200 countries / 5000 products".
3. **Size:** one 1.27 GB zip holding 13 per-year CSVs (`t,i,j,k,v,q`). They total **4.63 GB uncompressed**, at ~**9–12 M rows/year** and **~130–150 M rows** overall.
4. **32 GB RAM:** with compact dtypes (Int16/UInt16 codes, UInt32 or Categorical product, Float64 v/q) the full history takes **~3.5–4 GB** in Polars, or ~2.7 GB with Float32. Naive dtypes take ~8 GB. **Feasible.** A *dense* exporter×importer×product×year grid (~3.5 B cells) is **not** feasible; keep the data sparse or UK-centred.
5. **HMRC OTS** gives UK exports by **CN8 (aggregable to HS6) × partner × month**. It is available through a free OData API (Jan 2000–Sep 2026) and through monthly/annual bulk zips (2016+). Its numbers will differ from BACI's UK rows: OTS is in GBP and uses the current-year CN/HS edition, it suppresses detail for some codes, and it is UK-declared only. BACI is in thousand USD, recast to HS2012, and mirror-reconciled.

## 1. What BACI HS12 contains (V202601)

| Fact | Value | Source |
|---|---|---|
| Current version | 202601. The product page says "last updated January 30, 2026". The doc page, release notes and the zip's Readme all say **22 Jan 2026**. | https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37 ; https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html |
| HS12 year coverage | **2012–2024**. Other revisions also end in 2024 (HS92 1995–2024 … HS22 2022–2024). | https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html |
| Is 2024 included? | **Yes**, but provisional: "values and quantities of trade flows for the most recent year … may be significantly revised in future releases", because Comtrade is downloaded in January while the latest year is incomplete. | baci_webpage.html, FAQ |
| Is 2025 included? | **No.** "BACI is updated every year in January." No 2025 data exists as of 2026-11-16. | baci_webpage.html, FAQ |
| Stated scope | "bilateral trade flows for 200 countries at the product level (5000 products)" | baci_webpage.html |
| HS6 codes in product file | **5,202** rows in `product_codes_HS12_V202601.csv` (I read it from the zip) | https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS12_V202601.zip |
| HS6 codes actually traded | ~**5,198–5,199**, flat across years (y-axis of Fig. 35, "Number of products, revision HS12") | https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf |
| Country codes | **238** rows in `country_codes_V202601.csv`. This includes historical entities such as "Belgium-Luxembourg (...1998)". UK = `826,United Kingdom,GB,GBR`. Taiwan is code 490 "Other Asia, nes" (ISO3 field `S19`). | zip (read directly); Taiwan proxy explained in the baci_webpage.html FAQ |
| Exporters per year | ~**226** (Fig. 29, HS12) | release_notes_202601.pdf |
| Dyads per year | ~25k–31k (Fig. 23, HS12) | release_notes_202601.pdf |
| Rows ("dyad-products") per year | Fig. 11's axis for HS12 runs 9M–11M. My byte-based estimate is 9.5M (2012) to 12.4M (2023). See §2. | release_notes_202601.pdf; my estimate |
| What changed vs 202501 | "Source data (Comtrade) was updated. Fobization process was simplified, with little effects on final outcome." | release_notes_202601.pdf §1 |
| Licence | Etalab Open Licence 2.0 (any use, with attribution) | baci_webpage.html |

### File format and download size

| Fact | Value | Source |
|---|---|---|
| Download | One zip per HS revision: `https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS12_V202601.zip` | baci_webpage.html |
| Zip size | **1,267,950,839 bytes (1.27 GB)** (HTTP `Content-Length`, Last-Modified 22 Jan 2026). For comparison, HS17 is 0.79 GB and HS22 0.30 GB. | HEAD request to the URL above |
| Contents | 13 files `BACI_HS12_Y{2012..2024}_V202601.csv`, plus `product_codes_HS12_V202601.csv`, `country_codes_V202601.csv` and `Readme.txt` | zip central directory (range read) |
| Uncompressed CSV total | **4,627,105,229 bytes (4.63 GB)**. Per year: 294 MB (2012), rising to 382 MB (2023), then 361 MB (2024). | zip central directory |
| Format | CSV with comma delimiter and dot decimal. Header `t,i,j,k,v,q`. **No Parquet is offered.** | baci_webpage.html; sample rows |
| Columns | `t` year; `i` exporter; `j` importer (numeric Comtrade-style codes, "may differ from standard ISO" codes); `k` HS6 (read as **string**, because it has leading zeros); `v` value in **thousand current USD**; `q` quantity in **metric tons** | baci_webpage.html; Readme.txt in zip |
| Zeros | "only the strictly positive trade flows are recorded." A missing row means zero *or* unreported. | baci_webpage.html |
| Missing quantity | `q` is an **empty field** in some rows, e.g. `2024,4,31,902710,0.032,`. It was empty in ~3–6% of the rows I sampled. | sample rows (my observation) |
| Sample row | `2024,4,24,080810,0.176,0.1` | zip, first lines of 2024 file |

## 2. Memory footprint for 32 GB (Polars / Parquet)

**Row-count estimate.** In the sampled rows, the mean CSV line is 30.7–31.2 bytes. Dividing each year's uncompressed size by 30.9 bytes gives 9.5M–12.4M rows per year and **~150M rows in total**. The sample comes from the start of each file (exporter code 4, Afghanistan), where values are small and lines short, so this is probably an **overestimate**. The release-notes chart (axis tops out at 11M/year) points nearer **~130–140M**. I use 150M below as a conservative upper bound. Exact row counts were **not confirmed**; they need a full read.

| Representation | Bytes/row | 150M rows |
|---|---|---|
| Compact: `t` Int16, `i`/`j` UInt16, `k` UInt32 (or Categorical, 4 B), `v`/`q` Float64 (+ null bitmap for `q`) | ~26 | **~3.9 GB** |
| Compact with `v`/`q` as Float32 (fine on a log scale; loses sub-dollar precision only for large flows) | ~18 | **~2.7 GB** |
| Polars defaults: Int64 codes, `k` as String (16-byte view; 6 chars inline), Float64 | ~56 | **~8.4 GB** |
| pandas with Python-object strings for `k` | ~100+ (rough) | ~15 GB+ (rough, unverified) |

- **Verdict:** the full 2012–2024 HS12 history fits in 32 GB, with working room for joins and group-bys (budget 2–3× the frame). Better still: convert each year CSV to a Parquet partition once, then use `pl.scan_parquet(...)` lazily, so only the needed columns and years load. My estimate for zstd Parquet on disk is ~1–1.5 GB, similar to the zip. This is **not measured**.
- **Trap: dense grids.** 226 exporters × 226 importers × 5,198 products × 13 years ≈ **3.45 billion cells**, or ~13.8 GB for a *single* Float32 column. Any "fill in zeros for all combinations" step must stay sparse or be restricted. UK-centred grids are small: UK × ~225 markets × 5,198 × 13 ≈ **15M rows**. Exporter×product (or importer×product) × year panels are each ≈ 15M rows.
- The RTX 3080's 10 GB VRAM will not hold the full table. Train on aggregated or sampled feature tables, not raw flows. This is my inference, not checked against a source.

## 3. Documented caveats

**Mirror reconciliation** (owner: CEPII):
- Raw Comtrade has duplicate mirror flows (exporter-reported and importer-reported) that "virtually never" match, for two reasons: imports are CIF and exports FOB, and there are mistakes such as uncertainty about the final destination of exports or classification differences. BACI estimates CIF costs, strips them from import values, and weights each reporter by its estimated reliability. https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html (Methodology)
- The method paper explains the details. The CIF/FOB ratio is regressed on gravity variables and the product's median unit value, and reliability comes from a weighted variance decomposition of mirror ratios. Gaulier & Zignago (2010), CEPII WP 2010-23: https://www.cepii.fr/PDF_PUB/wp/2010/wp2010-23.pdf
- Consequence: **BACI's UK export value for a flow is a blend of the UK's report and the partner's import report.** It is not HMRC's number.

**Confidentiality** (the 2010 paper; the current pages are silent):
- "Due to confidentiality for instance, countries may not report some of its detailed trade. However, this trade will be included at the higher commodity level and in the total trade value (and sometimes via the use of a specific item…)" (wp2010-23.pdf, §2.1). Confidential UK HS6 detail can therefore be missing from the UK side, and partner mirror reports may partly fill it.
- The current BACI page and FAQ do **not** mention confidential trade. I could not confirm how V202601 treats it.

**Re-exports and "areas nes"** (the 2010 paper; I could not confirm these still apply in V202601):
- "Re-exportations for Hong Kong and USA are also dropped since there is no way to know the final destination of the flow." Flows within groups such as Belgium-Luxembourg are dropped to keep the geography consistent (wp2010-23.pdf, fn. 11).
- "Areas Not Elsewhere Specified" flows are reallocated to partners where mirror data allow it (wp2010-23.pdf, Appendix 6).
- The 2010 paper notes Comtrade source data "do not include flows below 1,000 dollars" (§2.1). I could not confirm this threshold for current Comtrade or BACI. The smallest `v` values in my sample (e.g. 0.032 = $32) suggest it no longer applies.
- The current BACI page does **not** use the word "re-export". Nothing documented says how UK re-exports are treated.

**Other caveats** (current docs):
- Last year provisional; flows are revised across versions; **"avoid mixing data from different versions."** Legacy versions are only for replication. https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html (FAQ, Archives)
- Version 202401 contained within-country flows and was superseded by 202401b. baci_webpage.html (Archives)
- Countries report in whichever HS revision they use, and BACI converts newer revisions to older ones. Converting to an older revision loses products, and some countries do not report in the newest revisions. For HS12 this means 2017+ data is largely converted from HS17/HS22 reports. https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/DescriptionBACI.html
- No Taiwan label: code 490 "Asia, nes" is the proxy. baci_webpage.html (FAQ)

## 4. HMRC OTS: UK HS6 exports by country

### API (owner: HMRC, uktradeinfo.com)

| Fact | Value | Source |
|---|---|---|
| Endpoint | `https://api.uktradeinfo.com/OTS` (OData). Open access, no auth header, JSON or CSV (`$format=csv`). | https://www.uktradeinfo.com/api-documentation/ |
| Limits | **40,000 rows per response** (follow `@odata.nextLink`); **60 requests/minute** | same |
| OTS fields | `MonthId, FlowTypeId, SuppressionIndex, CommodityId, CommoditySitcId, CountryId, PortId, Value, NetMass, SuppUnit` (rows are also split **by port**) | https://api.uktradeinfo.com/$metadata |
| Flow types | 1 EU Imports, 2 EU Exports, 3 Non-EU Imports, 4 Non-EU Exports. UK exports = 2 + 4. | api-documentation; `/FlowType` |
| Commodity level | `CommodityId` = CN8 or HS2/4/6 as an integer. The `Commodity` entity carries `Cn8Code` and `Hs6Code`, so HS6 is reached by aggregating CN8. | api-documentation; $metadata |
| Time coverage | Not stated in the docs. **Checked by me:** OTS rows exist for MonthId 200001 and 202609 but not 199601, so the API covers **Jan 2000 – Sep 2026**. | live API queries |
| HS6 × country works server-side | Tested: `OTS?$apply=filter(MonthId ge 202401 and MonthId le 202412 and (FlowTypeId eq 2 or FlowTypeId eq 4) and Commodity/Hs6Code eq '220421')/groupby((Country/CountryCodeAlpha),aggregate(Value with sum as v))` answered in ~5 s | live API query |
| Value unit | API docs do **not** state it. The bulk data dictionary defines STAT-VALUE as £ sterling. I assume, but have **not confirmed**, that the API's `Value` is the same (whole £). | https://www.uktradeinfo.com/trade-data/latest-bulk-data-sets/bulk-data-set-data-dictionary/ |

Pulling all of UK HS6 × country × 2012–2025 is feasible, at roughly 5,000 HS6 × ~230 countries × 12 months per year, once aggregated over ports. That is a few million rows per year, i.e. ~100 pages of 40k at 60 req/min. This is my estimate; I did not run it.

### Bulk data

| Fact | Value | Source |
|---|---|---|
| Monthly files | `bdsexpYYMM.zip` (exports). The latest is **Sep 2026** (`bdsexp2609.zip`, 2.8 MB). Each release fully replaces the previous one, revisions included. | https://www.uktradeinfo.com/trade-data/latest-bulk-data-sets/ |
| Archive | Annual export archives `bdsexp_YYarchive.zip`, **2016–2026** (2016 = 29.5 MB). "2016 through to 2023 have been replaced with up-to-date information." Nothing before 2016. | https://www.uktradeinfo.com/trade-data/latest-bulk-data-sets/bulk-data-sets-archive/ |
| Export fields | `COMCODE` (CN8), `COD-ALPHA` (country of **ultimate destination**), `STAT-VALUE` (£), `NET-MASS` (kg), `SUPPUNIT`, `ACCOUNT-CCYYMM`, port, mode of transport, suppression level, `TYPE` (1 declared; 2 below-threshold allocation; 3/4 non-response estimates) | bulk-data-set-data-dictionary (URL above) |
| Format | "large delimited text files"; EU and non-EU flows merged into one file from Jan 2022 | https://www.uktradeinfo.com/trade-data/latest-bulk-data-sets/bulk-data-sets-information-pack-includes-post-eu-exit-changes/ |

### How OTS differs from BACI's UK rows

| Dimension | HMRC OTS | BACI | Source |
|---|---|---|---|
| Value basis | UK-declared exports, FOB, **£ sterling** (converted at HMRC period rates) | Reconciled UK report + partner mirror (CIF→FOB), **thousand USD** | OTS methodology §8.2: https://www.gov.uk/government/statistics/overseas-trade-statistics-methodologies/overseas-trade-in-goods-statistics-methodology-and-quality-report--3 ; baci_webpage.html |
| Product codes | CN8 in **that year's** nomenclature. E.g. HS2022 code 8524xx has 2024 data in the API, so OTS HS6 needs an HS22/HS17 → HS12 concordance to match BACI HS12. | All years expressed in HS2012 | live API query; DescriptionBACI.html |
| Frequency and latency | Monthly, through **Sep 2026** | Annual, through **2024** | sections above |
| Trade system | **Special trade** since May 2016 (general before). Re-exports from customs warehouses/free zones and goods in transit are **not recorded**. | Whatever the UK reports to Comtrade, blended with partner data. Not documented. | OTS methodology §8.1 |
| Confidentiality | Passive suppression at CN8 (levels 1–3). Suppressed trade moves to non-trade codes, e.g. `CommodityId -990` and country 959 "Estimates" (`QV`). Suppression rolls up to CN6/CN4. | Hidden UK detail may be partly recovered from partner mirror reports | https://www.uktradeinfo.com/trade-data/suppressions-policy/ ; live API (`CommodityId -990`, `CountryId 959`) |
| Estimates | Includes below-threshold trade allocations and non-response estimates (`TYPE` 2–4) | Not applicable | OTS methodology §5.4; data dictionary |
| Data-source breaks | UK→EU exports: Intrastat until end-2020, customs declarations from 2021 (Northern Ireland stays on Intrastat under the Windsor Framework) | Inherits whatever the UK reports, smoothed by mirroring | OTS methodology §3.1–3.2 |
| vs ONS | HMRC uses the IMTS basis. ONS's BoP basis (change of ownership) differs again. | — | OTS methodology §8.4 |

**Not confirmed:** I made no numeric BACI-vs-OTS comparison for UK flows (that needs one BACI year file). I found nothing documenting how CEPII handles the UK's non-monetary gold or re-exports specifically.

## Sources

- CEPII BACI product page: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
- CEPII BACI documentation/FAQ/download page: https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html
- BACI structure page: https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/DescriptionBACI.html
- Release notes 202601: https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf
- HS12 zip (sizes and listing via HEAD/range only): https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS12_V202601.zip
- Gaulier & Zignago (2010), CEPII WP 2010-23: https://www.cepii.fr/PDF_PUB/wp/2010/wp2010-23.pdf
- uktradeinfo API docs: https://www.uktradeinfo.com/api-documentation/ and metadata https://api.uktradeinfo.com/$metadata
- uktradeinfo bulk datasets: https://www.uktradeinfo.com/trade-data/latest-bulk-data-sets/ ; archive …/bulk-data-sets-archive/ ; data dictionary …/bulk-data-set-data-dictionary/ ; info pack …/bulk-data-sets-information-pack-includes-post-eu-exit-changes/
- uktradeinfo suppressions policy: https://www.uktradeinfo.com/trade-data/suppressions-policy/
- HMRC OTS methodology and quality report: https://www.gov.uk/government/statistics/overseas-trade-statistics-methodologies/overseas-trade-in-goods-statistics-methodology-and-quality-report--3
