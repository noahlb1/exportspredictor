# Tariff data revisited (v2)

Researched 2026-11-17 for issue #28 "Tariff data revisited" (child of map #24). It updates `docs/research/hs6-tariff-data.md` (v1, 2026-11-16, issue #5), so read that first. This note records what changed and sets out the feasible paths. It does not choose one: that is ticket #35 "Feature additions".

## Bottom line

1. **The core gap has not closed.** The WITS/TRAINS API still stops at **2021** for every reporter. A live check on 2026-11-17 found the newest record dated 2025-08-11 and no year after 2021. No free, bulk, annual, bilateral HS6 tariff panel covers BACI's 2012–2024 (ADR 0003).
2. **TRAINS 2021 does carry UK-specific preferences.** In 2021, 37 reporters list the UK (`826`) as a partner, including EU (`EUN`), Japan, Korea, Switzerland, Norway, Singapore and Chile. In 2020 only New Zealand does. The Japan→UK 2021 file came back live: 2,735 HS6 rows of `PREF`, in HS2017 (`H5`). Before 2021, the UK's partners' preferences sit in EU preference groups, which must be expanded.
3. **New on the WTO side, but the licence is unclear.** The WTO's new `data.wto.org` catalogue lists the **Analytical Database (ADB)** as HS6, MFN plus preferential, "1996-present", updated monthly, under **CC BY 4.0**. The ADB download selector offers years up to 2026. But the TTD Terms of Use still say "Mass downloads ... is prohibited" without permission from idb@wto.org, and the download needs an emailed login. **This is the most plausible source for 2022–2024, if WTO gives permission.**
4. **The WTO–IMF Tariff Tracker covers 2025 onwards only.** It is HS6, bilateral, and includes the 2025–26 additional duties, for 15 reporters including the UK, the EU, the US, China and India (page last updated 30 Oct 2026). It is no use for training years, but it is a candidate **current Baseline** for tariff-change Scenarios.
5. **Trade-war layers.** Teti's *GTD: U.S. Trade War Extension* is unchanged (v_2025-10, HS6 in HS2017, Jan 2018 to 15 Aug 2025, free download, no licence stated). It still covers the 2018–19 US–China escalation that TRAINS misses. A new CEPII dataset (Guimbard 2026, CC BY-NC 4.0) gives US import tariffs for 2025–26 at HS6 (HS2022), on a MAcMap-HS6 **v2022** baseline.
6. **Unchanged.** Teti's GTD HS6 microdata are still not public, and the public GTD still ends at 2021. CEPII's public MAcMap-HS6 still ends at 2019; a v2022 exists but is "on request for academic use". ITC Market Access Map still has no free full access for UK users. Global Trade Alert's tariff-rate fields are paid.
7. **UK sources cover the UK's own import tariff, not partners' tariffs on UK goods.** These are the DBT Data API (OGL v3, no key) and the HMRC Trade Tariff API (OAuth key, rate limited from autumn 2026). They matter for UK Export Potential only indirectly.

## What changed since v1

| Source | v1 (2026-11-16) | Now (2026-11-17) | Change |
|---|---|---|---|
| WITS / UNCTAD TRAINS API | 2012–2021, no later year | Same; newest `lastupdateddate` 2025/08/11 | None |
| WTO ADB (HS6, MFN + best preferential) | TTD download; mass download prohibited | Also listed on `data.wto.org` as **CC BY 4.0**, monthly, "1996-present"; selector offers years to 2026; login by email; ToU unchanged | Licence signal is new but contradicts the ToU |
| WTO–IMF Tariff Tracker | From Jan 2025 | From Jan 2025, 15 reporters incl. UK; updated 30 Oct 2026; HS6 download by email link | Wider reporter set, still 2025+ |
| WTO Timeseries API | Key needed; limits unknown | Same (`401 missing subscription key`); limits still undocumented | None |
| CEPII MAcMap-HS6 | Public 2001–2016 (+2019 file) | Same public files; **v2022 exists** (cited as baseline in CEPII WP 2026-07), on request | v2022 known to exist |
| CEPII US import tariffs 2025–2026 | — | New: HS2022 HS6, by origin, CC BY-NC 4.0 | New |
| Teti GTD | 1988–2021; HS6 microdata restricted | Same | None |
| GTD U.S. Trade War Extension | Jan 2018–15 Aug 2025 | Same (v_2025-10) | None |
| Global Trade Alert | Not covered | Interventions database CC BY-NC 4.0; tariff-rate fields and US tariff estimates are paid | New to this note |
| ITC Market Access Map | Paid for UK users | Same; free for EU users to 30 Nov 2028 | None |
| UK tariff (DBT, HMRC) | Not covered | DBT Data API OGL v3; HMRC Trade Tariff API with keys | New to this note |

## Findings by source

### 1. UNCTAD TRAINS via WITS

- **Years, re-checked live.** `https://wits.worldbank.org/API/V1/wits/datasource/trn/dataavailability/country/all/year/all` (2026-11-17) still lists, per year, 2012: 137, 2013: 140, 2014: 111, 2015: 139, 2016: 129, 2017: 112, 2018: 116, 2019: 121, 2020: 144 and 2021: 144 reporters. It lists **nothing after 2021**, and the latest `lastupdateddate` in the file is 2025/08/11. The World Bank catalogue still says "1988 to 2021". Source: [World Bank Data Catalog](https://datacatalog.worldbank.org/search/dataset/0040255/World-Integrated-Trade-Solution-UNCTAD-TRAINS).
- **UNCTAD has newer data internally.** Its *Key statistics and trends in trade policy* uses TRAINS for later years. Source: [UNCTAD, Key statistics and trends in trade policy 2024](https://unctad.org/publication/key-statistics-and-trends-trade-policy-2024). Those years have not reached the public WITS API. Asking `tab@unctad.org` is an open option, but the outcome is not known.
- **The UK as a partner (live, from the same availability file).**
  - 2021: 37 reporters list `826` directly: BLZ, BRB, BWA, CHE, CHL, COL, CRI, DOM, ECU, EUN, GEO, GTM, GUY, HND, ISL, JAM, JPN, KOR, LIE, LSO, MAR, MDA, MKD, MUS, NAM, NIC, NOR, NZL, PER, SGP, SLV, SUR, SWZ, SYC, UKR, VCT and ZAF. These are mostly the UK's rollover agreements plus the TCA.
  - 2020: only NZL. Before 2021, preferences to the UK come through EU preference group codes. For example, KOR 2015 lists `A18;A36;A47;A48;M03;N52;N56;N96`, and JPN 2020 lists `C61;C62`. These must be expanded with the TRAINS preference-beneficiary tables (see v1).
  - Not every FTA shows up as a preference. For example, CHL 2016 lists only `000` (MFN), even though Chile has an FTA with the EU. A reporter-year with no preference rows therefore means MFN in the data, not necessarily MFN in reality.
- **Live sample.** `API/V1/SDMX/V21/datasource/TRN/reporter/392/partner/826/product/all/year/2021/datatype/reported` returned 2,735 HS6 series, all `TARIFFTYPE="PREF"` and `NOMENCODE="H5"`, at about 0.9 MB.
- **HS revision mix (reporter-years, from the availability file).**

  | Year | H2 (HS02) | H3 (HS07) | H4 (HS12) | H5 (HS17) |
  |---|---|---|---|---|
  | 2012 | 8 | 60 | 69 | 0 |
  | 2015 | 1 | 13 | 125 | 0 |
  | 2017 | 1 | 9 | 25 | 77 |
  | 2019 | 1 | 6 | 9 | 105 |
  | 2021 | 1 | 4 | 5 | 134 |

  So most reporter-years from 2017 onwards need an HS17→HS12 conversion, and the early years need HS07→HS12.
- **Effective applied rate convention.** WITS defines the effectively applied rate as "the lowest rate between MFN and Preferential tariffs". It takes the lowest preferential rate among national lines, and when preferential data are missing, "the MFN tariff is used in place of the preferential tariff". Source: [WITS bilateral tariff technical note](https://wits.worldbank.org/Bilateral-Tariff-Technical-Note.html).
- **Access and licence** are unchanged from v1. The API is free with no key, and the catalogue licence is "Custom License".

### 2. WTO Tariff & Trade Data (TTD), ADB and the WTO–IMF Tariff Tracker

- **ADB catalogue entry (new).** `data.wto.org` describes the Analytical Database as MFN applied and preferential tariffs plus trade at "the Harmonized System sub-heading level". Its tables are Tariff-Trade, Preferential-Detail (each rate with regime code and partner), Inventory, and Indicator/Flag. Preferences are the "Best-available preference", the lowest duty found. It also gives:
  - Temporal coverage "1996-present" and update frequency "Monthly".
  - Licence: `https://creativecommons.org/licenses/by/4.0/` (in the JSON-LD).
  - Source: [data.wto.org: Analytical database](https://data.wto.org/dataset/analyticaldb).
  - The parent TTD entry gives ODbL 1.0 instead. Source: [data.wto.org: WTO TTD](https://data.wto.org/en/dataset/wto_ttd). The two catalogue entries do not agree with each other.
- **ADB download page.**
  - Filters: Reporters, Years ("Latest available year", then 2026 back to 1996), Data type and HS code list.
  - An "authorized email" login link is required.
  - The data types shown in the selector are MFN applied duty, Final bound duty, Imports by partner and Exports by destination. How the preferential table is selected was **not confirmed** from the page text. The page also shows no "all economies" option.
  - Source: [TTD ADB six-digit download](https://ttd.wto.org/en/download/six-digit).
- **Terms of Use are unchanged** (PDF dated 2025-02-28, read 2026-11-17). Non-commercial research use is "granted ... without fee and without formal request", but "Mass downloads of information from the platform is prohibited. The User must request the WTO Secretariat (idb@wto.org) for permission before mass-downloading". Re-dissemination needs permission and the citation "WTO Tariff and Trade Data". Source: [TTD Terms of Use (PDF)](https://ttd.wto.org/docs/Terms_of_Use_WTO_Tariff_and_Trade_Data.pdf).
  - **Reading:** the catalogue's CC BY 4.0 tag and the ToU's mass-download ban conflict. The safe route is to email idb@wto.org with the scope: about 150 reporters × 2022–2024 × HS6, for a course project, not redistributed.
- **Automated access.** Plain `curl` to `ttd.wto.org` PDFs returned HTTP 403 (Azure gateway / Cloudflare), so scripted pulls are also not straightforward.
- **WTO Timeseries API.** It is unchanged. `https://api.wto.org/timeseries/v1/...` returns 401 without a subscription key, and the rate limits are still undocumented. Source: [WTO API portal](https://apiportal.wto.org/). Its preferential indicators are HS6 bilateral "lowest" preferential rates (see v1 §2).
- **WTO–IMF Tariff Tracker ("Tariff actions").**
  - It records "initial tariff levels (applied and preferential rates)" and "subsequent changes in effective duties", by trading partner, HS6 and implementation date. It starts in January 2025.
  - Reporters (15): Argentina, Australia, Brazil, Cambodia, Canada, China, Colombia, Ecuador, European Union, India, Mexico, Switzerland, United Kingdom, United States and Zimbabwe.
  - Download by email link, under the general ToU. Page "Last updated: 30 October 2026".
  - Sources: [Tariff actions](https://ttd.wto.org/en/reports/tariff-actions); [HS6 download](https://ttd.wto.org/en/download/tariff-actions/six-digit); [Methodology note](https://ttd.wto.org/download-static/Methodology-Note-for-the-WTO-IMF-Tariff-Tracker.pdf).
  - The HS revision is not stated on the pages; v1 read it as HS 2022 from the methodology note.

### 3. CEPII

- **MAcMap-HS6, public files are unchanged.** HS6 for 2001–2016 every three years, plus a separate 2019 file, in HS2007, licensed CC BY-NC-SA. "Other versions ... are available on request for academic use" (macmaphs6@cepii.fr). Source: [CEPII MAcMap-HS6](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=12).
- **MAcMap-HS6 v2022 exists.** CEPII's new US-tariff dataset states that its baseline is "MAcMap-HS6 version 2022 applied protection" (ITC-CEPII). Source: [CEPII: U.S. import tariffs 2025–2026](https://cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=44). The v2022 file is **not** on the public MAcMap page, so getting it means an academic request.
  - It would give one more cross-section (2022) inside BACI's range.
  - It is the only post-2021 all-pairs bilateral HS6 source found that is not WTO.
- **U.S. import tariffs 2025–2026 (new).**
  - Ad valorem equivalents of US import tariffs by origin and entry-into-force date, at HS6 (HS2022). The baseline is 1 Jan 2025.
  - Licence CC BY-NC 4.0. Zip with CSV and Parquet.
  - It falls outside 2012–2024, so it is relevant only for Scenarios or a later BACI year.
  - Source: same page; [CEPII WP 2026-07](https://www.cepii.fr/PDF_PUB/wp/2026/wp2026-07.pdf).

### 4. Teti Global Tariff Database and the U.S. Trade War Extension

- These are unchanged since v1. The GTD is v_beta1-2024-12, 1988–2021, and "The detailed microdata at the HS6-level are currently not publicly available."
- The trade-war extension covers "from January 2018 through August 15, 2025", at HS6 (HS2017) and HTS10. No licence is stated, only "© 2026 Feodora Teti".
- Source: [Feodora Teti: Data](https://feodorateti.github.io/data.html).
- The extension remains the only ready-made HS6 source for the 2018–19 US–China duties in training years. For UK Export Potential it matters indirectly: as a competitor shock in US and China imports, not as a tariff on UK goods. The exception is US Section 232 on UK steel and aluminium; whether the extension includes it was **not confirmed**.

### 5. Global Trade Alert

- **Licence.** Website content is under "Creative Commons Attribution-NonCommercial 4.0 International". Exports are capped, and "a full research download is available for approved accounts". The API, Data Files and "Premium Datasets" are licensed. The "size of the change (tariff rates ...)" is an Enhanced Data Field, sold through the API, Dashboard and Data Files. Source: [GTA data access](https://globaltradealert.org/data-access).
- **US tariff endpoints** reject `policy_date` before 2025-01-20. Source: [GTA API quickstart: US tariffs](https://globaltradealert.org/api-access/quickstart/us-tariffs).
- **Fit.** GTA is an event log of interventions, not a rate panel. Free access gives which measures hit which products and partners, but not the rates, so it could only make a "tariff-hike event" flag. HS6 product coding in the free export was **not confirmed**.

### 6. ITC Market Access Map

- This is unchanged. It is free for developing-country users and for EU users until 30 Nov 2028. Other developed-country users, including the UK, get restricted access, with full access by subscription. US$900/year is quoted on a fee page that says its prices are valid "as of 01.01.08". Sources: [ITC options and fees](https://mas-admintools.intracen.org/Accounts/OptionsFees.aspx); [EC: free access for EU users](https://international-partnerships.ec.europa.eu/news-and-events/news/international-trade-centres-market-analysis-tools-now-free-all-eu-users-2021-07-22_es).
- `macmap.org` still returns 403 to automated fetches, so none of this was verified first-hand.

### 7. UK Global Tariff and UK preferential schedules

- **DBT Data API.**
  - Dataset `uk-tariff-2021-01-01` ("Tariffs to trade with the UK from 1 January 2021"), with tables `commodities`, `measures-as-defined` and `measures-on-declarable-commodities`.
  - Formats: SQLite, JSON, ODS, CSV and Parquet. Versions are immutable.
  - No authentication and no stated rate limit. "All content is available under the Open Government Licence v3.0".
  - Source: [DBT Data API](https://data.api.trade.gov.uk/).
  - It holds the UK's MFN (UKGT) and the UK's **import** preferences by origin, at 10-digit commodity codes. An HS6 view has to be aggregated.
- **HMRC Trade Tariff API.**
  - The live UK tariff, as JSON (for example `/api/v2/geographical_areas/JP` returned a JSON:API object on a live check).
  - OAuth client-credentials keys. Rate limits are being introduced in autumn 2026, and the sources disagree on the numbers: the docs and a service note say 500/min from 1 Oct 2026, and a docs PR says 750/min per key.
  - Sources: [Trade Tariff developer docs](https://docs.trade-tariff.service.gov.uk/); [docs PR #292](https://github.com/trade-tariff/trade-tariff-api-docs/pull/292).
  - Historical "as of" queries were **not confirmed**.
- **Fit.** UK Export Potential depends on **Markets' tariffs on UK goods**. The UK's own import tariff is a Barrier on the other direction of trade. These UK sources therefore help only with:
  1. A sanity check that a Trade Agreement is in force.
  2. The UK's side of a Scenario, if one is ever built.

  For tariffs that Markets apply to UK goods, the sources are TRAINS 2021 (37 reporters), the WTO ADB, the WTO–IMF tracker (2025+) and ITC MAcMap (paid).
- **DBT preference utilisation statistics** report how much UK exports use preferences, not the rates. Source: [Preference utilisation of UK goods in 2024](https://www.gov.uk/government/statistics/preference-utilisation-of-uk-trade-in-goods-2024/preference-utilisation-of-uk-goods-in-2024).

### 8. HS concordance to HS12

- UNSD publishes official conversion and correlation tables: [HS2017→HS2012](https://unstats.un.org/unsd/classifications/Econ/tables/HS2017toHS2012ConversionAndCorrelationTables.xlsx), [HS2022→HS2012](https://unstats.un.org/unsd/classifications/Econ/tables/HS2022toHS2012ConversionAndCorrelationTables.xlsx) and [HS2012↔HS2007](https://unstats.un.org/unsd/classifications/Econ/tables/HS%202012%20to%20HS%202007%20Correlation%20and%20conversion%20tables.xls). Source: [UNSD classifications](https://unstats.un.org/unsd/classifications/Econ).
- For tariffs, a one-to-many split (old code → several HS12 codes) copies the rate to each child. A many-to-one merge needs a rule, such as the simple mean or the min, which is consistent with WITS's "lowest" convention. That rule is a modelling choice to record, not a sourced fact.

## Ways to fill 2022–2024 (options, not a decision)

| Option | Gives | Cost / risk |
|---|---|---|
| A. Carry TRAINS 2021 forward to 2022–2024 (v1 plan) | Full coverage; no new data | Misses FTA phase-downs, new agreements (UK–Australia/NZ 2023, CPTPP accession Dec 2024) and the HS2022 switch. The evaluation years would hold imputed features, which biases a time-split test |
| B. WTO ADB for 2022–2024 (all reporters, HS6, best preferential) | Real post-2021 rates, including UK-partner preferences | Needs idb@wto.org permission (ToU) despite the CC BY 4.0 tag; login per email; unknown turnaround; no API |
| C. MAcMap-HS6 v2022 on academic request | One clean all-pairs 2022 cross-section with AVEs | Request to CEPII; CC BY-NC-SA; HS revision to confirm; still needs carry-forward to 2023–24 |
| D. WTO Timeseries API (HS6 preferential indicators) | Possibly 2022+ | Key needed; limits unknown; the same "mass download" question as B |
| E. Ask UNCTAD (tab@unctad.org) for TRAINS 2022+ | Same schema as the 2012–2021 panel | Unknown availability and turnaround |
| F. Overlay the Teti trade-war extension for US↔partner rows 2018–2024 | Captures the 2018–19 escalation | No licence stated; HS17→HS12; mostly a competitor shock for UK flows |
| G. Treat post-2021 as missing (NaN plus an "imputed" flag) | Honest about the gap | XGBoost routes NaN by learned default, and the monotone constraint does not apply to the missing branch |

One combination fits the v1 plan's spirit at the least cost: **A + F now, B or C requested in parallel and swapped in if granted.** That is an observation for #35, not a decision.

## How tariffs would sit under ADR 0002 (monotone, clipped)

- **Feature form.** Use a single numeric feature per (Market, Product, year) with the UK, or with the exporter in the global panel: the effectively applied ad valorem rate, for example `ln(1 + t)`. It gets an XGBoost `monotone_constraints` entry of **−1**, so a higher tariff never raises Export Potential.
  - A companion "preference margin" (MFN − applied) would also need a constraint (+1). It moves together with the Trade Agreement flag, so the two monotone features share credit. That is a modelling note, not a sourced fact.
- **Clipping.** Clip Scenario inputs to the `[min, max]` of the tariff feature over **training years only**, and compute that range on the fit years (time-split guardrail).
  - Without the trade-war layer, the training range for most manufactures tops out at the MFN peak.
  - A Scenario such as "Market raises tariff by 25 pp" could then hit the clip, and its Scenario Effect would flatten beyond the range seen. That is the intended ADR 0002 behaviour, and the UI should show it.
  - With layer F, US–China 2018–19 duties of 25% are in range.
- **Scenarios change only policy features.** A tariff-change Scenario edits this feature (and the margin, if used) on the Baseline row and re-predicts with the out-of-fold model (ADR 0001).
  - The **Baseline tariff** for a current Scenario could come from the last training year (2024, possibly carried forward) or from the WTO–IMF tracker's 2025 levels.
  - Tracker values would have to be clipped too, and that baseline would no longer match the training feature's provenance.
- **Leave-one-out.** Tariff rates are not built from UK trade data, so the leave-one-out supply guardrail does not apply. **Trade-weighted** HS6 averages would make it apply, if the weights include UK flows to the target Market. Simple averages, as TRAINS `SimpleAverage` gives, avoid this.
- **Language.** Any reported tariff Scenario Effect stays associational: "the model associates a 10 pp tariff cut with +X%".

## Not confirmed / open

- Whether WTO will allow a 2022–2024 global ADB pull, and how to reconcile the CC BY 4.0 catalogue tag with the ToU.
- How the ADB download selects preferential rates (the selector shows MFN, bound and trade only).
- The HS revision of MAcMap-HS6 v2022, and whether CEPII will share it.
- Whether the WITS logged-in bulk download has any year after 2021. The API and the catalogue say no.
- Whether the Teti extension includes Section 232 on UK goods. Its licence also remains open.
- Whether the HMRC Trade Tariff API offers historical dates.
- Whether the GTA free export carries HS6 codes.
