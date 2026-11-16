# 04: HS6 bilateral tariff data (2012 to latest)

Researched 2026-11-16 for issue `04-hs6-tariff-data`.

## Bottom line

1. No single free source gives bilateral applied (MFN plus preferential) tariffs at HS6, in bulk, for every year from 2012 to the latest BACI year. The best free one is UNCTAD TRAINS through the WITS API, and it **stops at 2021**.
2. TRAINS is free, needs no API key, and returns HS6 MFN and preferential rates. You get about 11,300 reporter-partner-year calls for 2012–2021, roughly 10 GB of JSON. Partner groups and the EU have to be expanded, and some years are missing (China has no 2012–13 data, and the UK has only 2021).
3. CEPII MAcMap-HS6 is clean and free, with ad valorem equivalents (CC BY-NC-SA), but it covers only one year in three (2013, 2016, 2019). WTO's HS6 database (ADB) bans mass downloads without permission. ITC Market Access Map is paid for UK users.
4. The 2018–19 US–China escalation is **not** in TRAINS, WTO or MAcMap annual MFN/preferential data. It is in Federal Register/USTR lists at HTS8, and it comes ready-made at HS6 in Teti et al.'s free "GTD: U.S. Trade War Extension" (2018 to Aug 2025).
5. **Recommendation: defer tariffs from v1.** Build v1 on BACI alone. In v2, add TRAINS 2012–2021 plus the GTD trade-war layer, and treat years after 2021 as "tariff unknown / carried forward".

## The question

Is bilateral applied tariff data at HS6, including preferential rates, available in bulk for about 2012 to the latest year, free and for academic use? Compare WITS/TRAINS, WTO tariff data, CEPII MAcMap-HS6 and others on coverage, gaps, licensing and download effort. Is the 2018–19 US–China tariff escalation captured at HS6?

## Comparison at a glance

| Source | Bilateral preferential? | HS6? | Years usable for 2012+ | Access / cost | Bulk vs API | Effort (1 person) |
|---|---|---|---|---|---|---|
| UNCTAD TRAINS via WITS | Yes (MFN + PREF rows per partner or partner group) | Yes | 2012–2021 (ends 2021) | Free, no key needed for API | API per reporter×partner×year; a bulk UI also exists | 3–5 days |
| WTO TTD: ADB download | Yes ("lowest/best preferential" per partner) | Yes | From 1996; latest year not confirmed | Free for non-commercial use; **mass download prohibited without permission** | Email-delivered selections; no bulk | Small for a few countries; needs permission for global |
| WTO TTD: IDB (tariff line) | Yes | Tariff line (8–10 digit) | From 1996 | **Members and authorised international organisations only** | n/a | Not available to a student |
| WTO Stats (Timeseries) API | Yes (lowest preferential, HS6 only) | Yes | Yearly | Free API key | Per-query API; rate limits not documented | Unknown; not tested (key needed) |
| WTO–IMF Tariff Tracker / Tariff actions | Yes, includes additional duties | Yes (HS 2022) | **Jan 2025 onwards** only | Free download under TTD terms | File download | Small, but wrong period for 2012–2024 |
| CEPII MAcMap-HS6 | Yes, with AVEs, all preferences | Yes (HS 2007) | 2013, 2016, 2019 only | Free, CC BY-NC-SA | Bulk zip | 1–2 days |
| ITC Market Access Map | Yes, tariff line + HS6 | Yes | Annual time series | Free in developing countries and EU; UK users likely paid | Web UI downloads | Unclear; site blocked to automated fetch |
| Teti Global Tariff Database (GTD) | Yes | HS6 microdata **not public** (aggregates are) | 1988–2021 | Aggregates free; HS6 by application only | Bulk (Dropbox) | n/a for HS6 |
| GTD: U.S. Trade War Extension | US tariffs on all partners plus partners' tariffs on US | Yes (HS 2017), plus HTS10 | Daily, Jan 2018 to 15 Aug 2025 | Free download | Bulk (Dropbox, .7z) | About 1 day |
| USTR / Federal Register (Section 301) | US on China only | HTS8 (finer than HS6) | 2018 onwards | Public domain | Notices with annexes | 2–4 days to build HS6 panel yourself |

## Findings by source

### 1. UNCTAD TRAINS via WITS (World Bank)

- **What it holds.** TRAINS holds preferential and MFN rates. The WITS API page says "Preferential and most-favored-nation (MFN) tariff rates are available in this dataset. Data are harmonized across the countries at 6-digit level of the Harmonized System." Source: [WITS REST API](https://wits.worldbank.org/witsapiintro.aspx).
- **Fields per HS6 row.** Each row has `SimpleAverage`, `TARIFFTYPE` (PREF or MFN), and counts of national lines that are preferential, MFN or non-ad-valorem (`NBR_PREF_LINES`, `NBR_MFN_LINES`, `NBR_NA_LINES`). It also has min/max rate and `EXCLUDEDFROM`. Source: [WITS API User Guide (PDF)](https://wits.worldbank.org/data/public/WITSAPI_UserGuide.pdf), section "Tariff Data Request".
- **Rate type.** It is a simple average of the ad valorem rates of the national lines inside each HS6. You choose `reported` (ad valorem only) or AVE-estimated (`aveestimated`), but not both in one request. Source: [WITS API User Guide](https://wits.worldbank.org/data/public/WITSAPI_UserGuide.pdf); [WITS REST API](https://wits.worldbank.org/witsapiintro.aspx).
- **Year coverage (live check, 2026-11-16).** I queried the data-availability endpoint (`https://wits.worldbank.org/API/V1/wits/datasource/trn/dataavailability/country/all/year/all`):
  - Reporters per year: 2012: 137, 2013: 140, 2014: 111, 2015: 139, 2016: 129, 2017: 112, 2018: 116, 2019: 121, 2020: 144, 2021: 144.
  - **No year after 2021 for any reporter.**
  - The World Bank data catalogue entry agrees: "Temporal coverage 1988 to 2021". Source: [World Bank Data Catalog: WITS UNCTAD TRAINS](https://datacatalog.worldbank.org/search/dataset/0040255/World-Integrated-Trade-Solution-UNCTAD-TRAINS).
- **Gaps seen in that availability file (2012–2021).**
  - China has no 2012 or 2013.
  - Korea has no 2019.
  - Turkey has no 2012 or 2014.
  - Mexico ends in 2018.
  - Saudi Arabia has only 2012–15, 2017 and 2020.
  - The UK (GBR) has **only 2021**, because before 2021 the UK applied the EU tariff, which TRAINS reports as reporter `EUN` (2012–2021 complete).
  - Individual EU members (for example DEU) are not reporters, so EU tariffs must be mapped from `EUN` to each member state.
- **Partners.** Each reporter-year has a partner list:
  - `000` means MFN/World.
  - 3-digit codes are single partners.
  - Alphanumeric codes are groups such as GSP beneficiaries, which you must expand with the TRAINS preference-beneficiary tables.
  - Sources: [WITS API User Guide](https://wits.worldbank.org/data/public/WITSAPI_UserGuide.pdf); [TRAINS Preference Beneficiaries help](https://wits.worldbank.org/WITS/wits/WITSHELP/Content/Utilities/D4.Using_TRAINS_Preference.htm).
  - Example: the USA 2019 list is `000;036;048;084;124;152;170;376;400;410;484;...;A41;B15;C08;C09;C10;C53;N75` (live query).
- **Request limits.**
  - "All products tariff schedules can be requested for a single reporter, partner and year."
  - A single reporter, partner and product can be requested for a range of years.
  - Source: [WITS REST API](https://wits.worldbank.org/witsapiintro.aspx).
  - Live test: `partner/all` returned HTTP 413, too large.
  - Live test: one reporter, one partner, all products for a year returned about 0.9 MB of JSON in about 1.6 s (GBR, 2021, MFN).
- **Bulk alternative.** The WITS web "Bulk Download (TRAINS)" covers MFN and preferential rates at HS6. Its "Download All Reporters and Years" option yields "a large number of files", one per year and country. It needs a WITS login. Source: [WITS Bulk Download (TRAINS) transcript](https://wits.worldbank.org/data/public/WITS_Transcripts/bulk%20download%20(trains)_en.pdf). Not tested here.
- **Licence and cost.** "Access to TRAINS database is free of charge and unlimited"; registration with WITS is required for the web tool. Source: [WITS FAQ](https://wits.worldbank.org/Faqs.html).
  - In practice the API answered without any key or login (live test).
  - Neither UNCTAD nor the World Bank accepts responsibility for accuracy. Source: [WITS FAQ](https://wits.worldbank.org/Faqs.html).
  - The data catalogue lists the licence as "Custom License", classification Public. Source: [World Bank Data Catalog](https://datacatalog.worldbank.org/search/dataset/0040255/World-Integrated-Trade-Solution-UNCTAD-TRAINS).
  - I did not read the separate UNCTAD dissemination-policy PDF that the FAQ links to.
- **Nomenclature.** Rates come in each reporter's own HS revision for that year. Each row carries a `NOMENCODE` attribute (for example H0 for China 1992). Converting to BACI's HS12 needs HS concordances. Source: live query plus the User Guide's nomenclature API.
- **Effort for one person.**
  - About 11,300 reporter-year-partner combinations exist for 2012–2021 (counted from the live availability file).
  - That is about 5 hours of sequential API calls and about 10 GB of JSON.
  - On top of that comes the logic for group expansion, EU-to-member mapping, MFN fallback, HS-revision concordance and gap-filling (carry forward or back).
  - Estimate: **3–5 working days** to get a usable HS12 bilateral panel. This is my estimate, not a sourced figure.

### 2. WTO tariff data: TTD (replaces TDF and TAO), IDB, ADB and Stats API

- **Old portals are gone.** `tdf.wto.org` returns a 301 to `ttd.wto.org`, and `tao.wto.org` also returns a 301 to `ttd.wto.org` (live check). TTD is described as a beta platform covering more than 150 economies, "including annual data from 1996 onwards for many of them". Source: [TTD User Guide (PDF)](https://ttd.wto.org/download-static/users-guide).
- **Download options.** Source: [TTD Download page](https://ttd.wto.org/en/download).
  - **ADB (Analytical Database).** HS6 applied tariffs, preferential tariffs and bilateral trade, from 1996. Gaps are filled from ITC, UN Comtrade and TDM. The ADB "provides the most favourable preferential rates available". Source: [TTD User Guide](https://ttd.wto.org/download-static/users-guide).
  - **IDB tariff line ("Tariff and import notifications").** It is "accessible only to WTO Member governments and authorized international organizations". Source: [TTD IDB download](https://ttd.wto.org/en/download/idb). **Not available to a student.**
  - **CTS (bound rates).** Also restricted to Members and authorised organisations. Bound rates are not applied rates in any case.
- **How downloads work.** You pick economy, year and product, enter an email, and get a link valid for 24 hours. Source: [TTD User Guide §7.1](https://ttd.wto.org/download-static/users-guide).
- **Licence.** Non-commercial use (research, analysis, classroom) is "granted ... without fee and without formal request". However, "**Mass downloads of information from the platform is prohibited.** The User must request the WTO Secretariat (idb@wto.org) for permission before mass-downloading". Redistribution also needs permission. Source: [TTD Terms of Use (PDF)](https://ttd.wto.org/docs/Terms_of_Use_WTO_Tariff_and_Trade_Data.pdf).
- **WTO Stats (Timeseries) API.** It is free with a sign-up key. Source: [WTO API portal](https://apiportal.wto.org/). Live call without a key returned `401 missing subscription key`.
  - MFN applied indicators exist down to HS 6-digit.
  - Preferential indicators are "available on a bilateral basis ... and only at the HS 6-digit subheading level". The value is the **lowest** preferential rate available to the partner.
  - Source: [WTO Stats Technical Notes (PDF)](https://stats.wto.org/assets/UserGuide/TechnicalNotes_en.pdf), §2.2.
  - Rate limits, paging and whether pulling the global HS6 set counts as a "mass download" were **not confirmed**.
- **Tariff actions / WTO–IMF Tariff Tracker.** This is bilateral HS6 data (HS 2022) with additional duties, and it can be downloaded. But it "documents initial tariff levels starting in January 2025". Its January 2025 baseline already includes "U.S. Section 232 duties on steel and aluminum and Section 301 duties on imports from China". So it does **not** give a 2018–19 history. Sources: [Methodology Note for the WTO-IMF Tariff Tracker (PDF)](https://ttd.wto.org/download-static/Methodology-Note-for-the-WTO-IMF-Tariff-Tracker.pdf); [TTD Tariff actions](https://ttd.wto.org/en/reports/tariff-actions).
- **Effort.** Small for a handful of reporters through the UI. For a global panel you would first need written permission from idb@wto.org, and the timeline for that is unknown.

### 3. CEPII MAcMap-HS6

- **What it holds.** A "bilateral measurement of applied tariff duties" that "takes regional agreements and trade preferences exhaustively into account". It gives an AVE for every importer-exporter-HS6 triplet. Source: [CEPII MAcMap-HS6 page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=12).
  - It handles specific duties and TRQs.
  - "Contingent protection is not included."
- **Years.** The public replication package (Fontagné, Guimbard & Orefice, JIE 2022) covers 2001, 2004, 2007, 2010, **2013, 2016**, for 152 importers and 189 exporters, in HS 2007. A separate 2019 file covers "only tariffs for the years 2001-2019 (every three years)". "Other versions ... are available on request for academic use." The page was last updated February 2024. Source: [CEPII MAcMap-HS6 page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=12).
  - So for 2012 onwards you get only **2013, 2016 and 2019**.
  - It has no 2020+ and no annual series.
  - It does not show the post-2021 UK tariff regime.
- **Licence.** Creative Commons BY-NC-SA; free download (zip links on the page). Source: same page.
- **US–China.** The page does not say whether 2019 includes Section 301 duties. "Contingent protection" usually means anti-dumping, countervailing duties and safeguards, so this is **unconfirmed**.
- **Effort.** About 1–2 days, including an HS07-to-HS12 concordance and interpolating between the three-year points.

### 4. ITC Market Access Map (macmap.org)

- **Status of this section.** `macmap.org` and the ITC account pages sit behind a Cloudflare challenge (HTTP 403 for automated fetch), so I could not read them directly. The points below come from search-engine extracts of ITC's own pages and are **not verified first-hand**.
- **What it holds.** Applied MFN and preferential tariffs at tariff line and HS6, plus TRQs, trade remedies and NTMs, with time series and raw-data download. Sources: [macmap.org](https://www.macmap.org/); [ITC news: new version of Market Access Map](https://intracen.org/news-and-events/news/new-version-of-market-access-map-provides-boost-to-transparency-in-trade).
- **Access.** The search extracts say access is free for users in developing countries, and that EU-located users have free access until 30 Nov 2028. Developed-country users otherwise get restricted access, with full access by subscription (US$900/year was quoted). Source: [ITC Market Analysis Tools: options and fees](https://mas-admintools.intracen.org/Accounts/OptionsFees.aspx) (unverified, as above).
  - **The UK is neither**, so a UK student probably does not qualify for free full access.
  - There is no documented bulk or API access for a global HS6 panel.
- **MAcMap-HS6 route.** The CEPII FAQ notes that the yearly tariff-line MAcMap is ITC's product, and that the harmonised HS6 version is CEPII's, "made available upon request for academic work". Source: [CEPII MAcMap-HS6 page, FAQ](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=12).

### 5. Teti Global Tariff Database (GTD) and its U.S. Trade War Extension

- **GTD v_beta1-2024-12.**
  - Covers bilateral MFN and preferential rates at HS6 for 200 importers, 1988–2021.
  - Only pair-level and sector-level **aggregates** can be downloaded (.dta/.csv).
  - "Access [to HS6 microdata] is restricted to a small test group of experienced researchers during this beta phase."
  - It explicitly excludes "safeguards, anti-dumping tariffs, or those related to the US trade war."
  - Source: [Feodora Teti: Data](https://feodorateti.github.io/data.html).
- **GTD: U.S. Trade War Extension (v_2025-10).**
  - "daily, product-level data on U.S. import and export tariffs vis-à-vis all trading partners from January 2018 through August 15, 2025".
  - Includes "all tariff measures imposed by the United States and applied to U.S. exports abroad during the 2018–2025 trade wars".
  - Free downloads at HTS10 and **HS6 (HS 2017)** in .dta/.csv.
  - Source: [Feodora Teti: Data](https://feodorateti.github.io/data.html), which cites Rodríguez-Clare et al. (2025), CESifo WP 12179.
  - No explicit licence is stated, only "© 2026 Feodora Teti" and a citation request. This is **unconfirmed** for redistribution.
  - It is a research dataset built from the official notices, not an official source, so spot-check it against the Federal Register (section 6).

### 6. Section 301 (US–China, 2018–19): primary sources

Federal Register notices from USTR (all via the [Federal Register API](https://www.federalregister.gov/api/v1/documents.json)):

| Action | Notice | Rate / effective date |
|---|---|---|
| List 1 (about $34bn, 818 subheadings) | [83 FR 28710, 2018-06-20](https://www.federalregister.gov/documents/2018/06/20/2018-13248/notice-of-action-and-request-for-public-comment-concerning-proposed-determination-of-action-pursuant) | +25%, from 6 July 2018 (heading 9903.88.01) |
| List 2 (about $16bn) | [83 FR 40823, 2018-08-16](https://www.federalregister.gov/documents/2018/08/16/2018-17709/notice-of-action-pursuant-to-section-301-chinas-acts-policies-and-practices-related-to-technology) | +25% |
| List 3 (about $200bn) | [83 FR 47974, 2018-09-21](https://www.federalregister.gov/documents/2018/09/21/2018-20610/notice-of-modification-of-section-301-action-chinas-acts-policies-and-practices-related-to) | +10%, then raised to 25% by [84 FR 20459, 2019-05-09](https://www.federalregister.gov/documents/2019/05/09/2019-09681/notice-of-modification-of-section-301-action-chinas-acts-policies-and-practices-related-to) (from 10 May 2019) |
| List 4A/4B (about $300bn) | [84 FR 43304, 2019-08-20](https://www.federalregister.gov/documents/2019/08/20/2019-17865/notice-of-modification-of-section-301-action-chinas-acts-policies-and-practices-related-to) | +10%, raised to 15% by [84 FR 45821](https://www.federalregister.gov/documents/2019/08/30/2019-18838/notice-of-modification-of-section-301-action-chinas-acts-policies-and-practices-related-to); 4B suspended by [84 FR 69447](https://www.federalregister.gov/documents/2019/12/18/2019-27306/notice-of-modification-of-section-301-action-chinas-acts-policies-and-practices-related-to); 4A cut to 7.5% by [85 FR 3741, 2020-01-22](https://www.federalregister.gov/documents/2020/01/22/2020-00904/notice-of-modification-of-section-301-action-chinas-acts-policies-and-practices-related-to) |

- **Granularity.** USTR identifies covered products by 8-digit (or 10-digit) HTSUS subheadings, and offers a lookup tool. Source: [USTR: How to Navigate the Section 301 Tariff Process](https://ustr.gov/issue-areas/enforcement/section-301-investigations/search). That is finer than HS6.
  - To build HS6 bilateral rates you must aggregate HTS8 to HS6. A partial-coverage HS6 is the share-of-lines problem that the WTO note also flags.
  - Product exclusions complicate it further.
- **China's retaliatory tariffs.** These are published by China's State Council Customs Tariff Commission. I did **not** fetch or confirm a primary source for them here. The Teti trade-war extension claims to include them.
- **Is it captured in the standard annual databases?**
  - **TRAINS: no.**
    - Live check: China (156) is not in the USA partner list for 2019, so US tariffs on China default to MFN (`000`).
    - `reporter/840/partner/156/product/850440/year/2019` returns `NoRecordsFound`.
    - The USA MFN simple average for 850440 is 0.12% in 2019.
    - TRAINS does capture China's own **MFN** cuts. For example, 870323 drops from 25 to 15 between 2018 and 2019 rows (live query). It does not capture China's retaliation against the US.
  - **Teti GTD base: no**, by explicit exclusion ([source](https://feodorateti.github.io/data.html)).
  - **WTO ADB annual series: not confirmed either way.** The WTO-IMF Tracker treats Section 301 as an "exceptional duty" added on top of IDB rates from its 2025 baseline ([source](https://ttd.wto.org/download-static/Methodology-Note-for-the-WTO-IMF-Tariff-Tracker.pdf)). That suggests, but does not prove, that the annual IDB/ADB MFN/preferential series leave it out.
  - **MAcMap-HS6 2019: unconfirmed.**
  - **At HS6, ready-made: yes**, in the GTD U.S. Trade War Extension.

## Implications for the project

- **Data end years.** Tariffs end at 2021 in the only free, bulk, annual, bilateral HS6 source, while BACI runs to a later year (see the BACI feasibility ticket). Any tariff feature for 2022 onwards would have to be carried forward or imputed. That is a modelling assumption, not data.
- **UK tariff history.** The UK's own tariff schedule exists in TRAINS only for 2021. For 2012–2020, "UK tariffs" means the EU's `EUN` schedule.
  - UK Export Potential depends mostly on **partners' tariffs on UK goods**, not the UK's own tariff.
  - Those rates change in 2021: the UK leaves EU FTAs, and the TCA and rollover agreements apply. TRAINS 2021 partner lists (for example China's) need checking for how they treat the UK as a partner. This was not checked.
- **Scenario needs.** Tariff-change Scenarios only need a current baseline tariff for each importer-UK-HS6 cell plus a delta. That is a smaller job than a 2012–2021 training panel. A single recent cross-section (TRAINS 2021, or a small WTO ADB/TTD pull for the UK's top markets) would do, and could come later.
- **Recommendation: defer tariffs in v1.** Revisit in v2 with this stack:
  1. TRAINS 2012–2021 via the API, as the panel.
  2. GTD U.S. Trade War Extension, overlaid for US↔partner rows from 2018.
  3. MAcMap-HS6 2013/2016/2019, as a cross-check.

## Not confirmed / open

- Whether WITS web Bulk Download (logged in) has years after 2021. The API and the data catalogue both say 2021 is the last.
- WTO Timeseries API rate limits, and whether a global HS6 pull would count as a "mass download".
- ITC Market Access Map pricing and terms for UK users (pages blocked by Cloudflare).
- Whether MAcMap-HS6 2019 or the WTO ADB include Section 301 duties.
- The UNCTAD TRAINS dissemination-policy text, and the licence for the Teti datasets.
- Primary sources for China's retaliatory tariff lists.
- Note: the ticket file `issues/04-hs6-tariff-data.md` (and the other `issues/` files) were no longer on disk when this note was written. The question above is quoted from the copy read at the start of the session.
