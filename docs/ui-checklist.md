# Manual UI checklist

Run this before a release, against `podman compose up` with real artefacts, in a browser window at least 1280 px wide (the app is designed for 1280 px and wider; phones are unsupported). Tick each item.

**Every page**
- [ ] The menu lists five pages with no group headings: Leaderboard, Blocked Opportunities, Scenario Builder, Model Evaluation and How the Model Works. The app opens on the Leaderboard.
- [ ] The logo tile (a white container ship on a blue square) sits alone at the top of the sidebar, shows in a collapsed sidebar, and is the browser-tab icon.
- [ ] The app follows the computer's light or dark mode: near-black text on a light page with a sidebar that stands apart, or light text on a warm near-black page. Check captions, charts and the info, warning and error boxes in both.
- [ ] Click something slow (e.g. a filter on the real Leaderboard): after a short delay the logo's ship rides the waves in the middle of the page until the app is done.
- [ ] Under its introduction, every page shows the grey line "Figures are associations the model has learned, not causes or forecasts." After a divider at its foot, a grey footer states the Baseline year (provisional data), that Export Gaps are 2022–2024 averages and "Values are nominal US dollars, as reported by BACI". No page shows a blue banner, a pound sign or a currency switch.
- [ ] Stop the API (`podman compose stop api`) and reload: within a few seconds every page shows a clear "API is unavailable" message, not an error or a spinner. Start it again (`podman compose start api`) and wait for <http://localhost:8000/health> before reloading.
- [ ] Narrow the window to about 700 px and open each page: none shows an error, though columns may stack or cut long names.

**Leaderboard: ranking and filters**
- [ ] The page opens without searching, listing UK Product–Market pairs by Export Gap, largest first, with no Excluded Product (e.g. crude oil or gold).
- [ ] On the real artefact (about 1.16 M pairs), the page loads, and answers each filter change, within about a second.
- [ ] The sidebar holds, in order: Product search, Market, Sector (by name, e.g. "Beverages, spirits and vinegar"), Market region, Opportunity Type, UK Trade Agreement in force, minimum Export Gap, Hide Flagged Products, Hide Re-export Hubs, Hide Sanctioned Markets, Hide Restricted Pairs, Hide Offset Pairs, Rank by, then **Reset filters**.
- [ ] One line above the table names the filters in force (or says there are none), "N pairs shown of M" and the total positive Export Gap ("2022–2024 average"), and changes with the filters. There is no row of summary metrics. A caption under it says Export Potential is a median-type estimate that sums to about N% of Actual Exports (N from the served artefact), so totals are relative sizes, not dollars the UK could add.
- [ ] Each row shows its Opportunity Type (Expand, New Market or New Product), Relative Export Gap, Potential Range as "low–high", whether a UK Trade Agreement is in force (whisky: Japan Yes, India No), and any caveat badge.
- [ ] Export Potential, Actual Exports and Export Gap sit side by side, each headed "2022–2024 average", with "(USD, nominal)" in the column headers.
- [ ] Switch "Rank by" to Relative Export Gap: a minimum Export Potential selector appears ($0, $100.0k, $1.0m, $10.0m, $100.0m; default $1.0m), pairs below it disappear, and pairs tied at 100% are ordered by Export Gap.
- [ ] Search an HS6 code (`220830`, whisky) and words (`smartphone` finds `851712`): the Product is chosen and its Markets are ranked by Export Gap. A prefix matching several Products (`8703`) offers each.
- [ ] Search text that matches nothing (`zzz`): a clear "No Product matches" warning.
- [ ] Search an Excluded Product (`gold`): it is marked "(Excluded Product)", and choosing it shows why it is not ranked and points to the Excluded Products section of Model Evaluation.
- [ ] Choose a Market (e.g. Japan): the Markets table keeps only Japan, with its Products ranked by Export Gap on the right; the region filter narrows the Market list. Above the Products, one **Market Reach** sentence says how many Products that Market imported from other Exporters in 2023, how many of them the UK sent too, and about how many were expected. Its tooltip says it applies to every Product of the Market alike.
- [ ] The Sector, region, Market, Opportunity Type, Trade Agreement, minimum-Gap and hide filters each narrow the rows; a filter matching nothing says so.
- [ ] Each filter you set appears in the URL (Product as its HS6 code, Market as its ISO3 code; defaults left out). The URL in a fresh tab shows the same filters and rows. `?product=220830&market=JPN` leaves that one pair.
- [ ] Set some filters, open another page and come back: the filters are still set.
- [ ] **Reset filters** restores the full ranking and clears the filters from the URL; the view stays.
- [ ] Export the CSV, unfiltered and with one Product or one Market chosen: its columns and rows match the table on screen.
- [ ] The first visit shows three steps (Read the ranking, Open a pair, Narrow it down) above the tables, the second saying the panel is below the tables. **Got it** closes them, and **What is this?** beside the heading brings them back.

**Leaderboard: caveat badges**

Every hide filter is off by default, and ticking one removes exactly its rows. No ranking page shows an "Uncertain Opportunity" badge or filter.

- [ ] Aircraft (`880230`): every row carries "Flagged Product", and the panel's warning says "volatile: UK exports swing widely from year to year". Banknotes and stamps (`490700`) say "one-off: few, large contracts".
- [ ] Jewellery (`711319`): every row carries "Flagged Product", and the Netherlands, Hong Kong and other Re-export Hubs carry "Re-export Hub".
- [ ] Whisky (`220830`): Saudi Arabia, Kuwait, Libya and Iran carry "Restricted Pair"; Russia, Belarus, North Korea and Syria carry "Sanctioned Market". The Blocked Opportunities page has the same two hide filters.
- [ ] The top two rows, turbine parts (`841191`) to France and turbo-jets (`841112`) to Germany, carry "Offset Pair". "Hide Offset Pairs" removes exactly the Offset Pairs (21 of the top 100 on the real artefact), keeps every other row's order, and the line above the table says "Offset Pairs hidden". The Blocked Opportunities page has the same filter.

**Leaderboard: views**
- [ ] A switch above the tables offers **By Market** (the default), **By Product** and **Wide table**. In each, the chosen pair's detail panel sits under the tables at full page width.
- [ ] **By Market**: Markets on the left, each with its number of pairs and total positive Export Gap, largest first (the totals add up to the line above), with the top Market chosen. Its Products are on the right (Export Gap, Export Potential, Actual Exports, Applied Tariff %, Relative Export Gap, UK Trade Agreement, Blocked Opportunity, Caveats), under a line naming the Market and its number of Products. The panel already explains the top pair. Click any value of another Market: its row is tinted and its Products replace the first. Click any value of a Product: its row is tinted, and the panel explains that pair from its top.
- [ ] **By Product**: the same with Products on the left and the chosen Product's Markets on the right.
- [ ] A chosen Market or Product lists all its pairs, though the page shows only the top 500 of the whole ranking: check the Market USA in By Market and whisky (`220830`) in By Product, with no other filter.
- [ ] Search a Product (`220830`) and switch to **By Product**: the Products table keeps only that Product. Choose a Market and switch back to **By Market**: the Markets table keeps only that Market.
- [ ] **Wide table**: every shown pair (the top 500, or all with **Show all**) in one table, with the wide columns (Applied Tariff % among them) and the others under Show/hide columns. The same pair stays selected and tinted across views; in **By Product** it stays only when the chosen Product is its Product, otherwise that Product's top pair is chosen. By Market and By Product each keep their own chosen Market or Product.
- [ ] The view is in the URL: `?view=product`, `?view=wide`, or nothing for By Market. A URL with `view=product` in a fresh tab opens By Product. The view survives a visit to another page, and **Reset filters**.
- [ ] Open `?product=902214&market=JPN` in **Wide table**: Actual Exports of $18,882 show as `0.0` (under $50,000 rounds to 0.0 in $ millions), and the column still sorts as numbers.
- [ ] Choose the Market USA (5,138 pairs) and turn on **Show all**: a row click answers in about a second.

**Leaderboard: the detail panel**
- [ ] The panel opens with the pair, its HS6, Sector and region, then a warning for each caveat:
  - a Flagged Product, naming its reason (Sector 71, volatile or one-off);
  - a Re-export Hub (e.g. `711319` to Hong Kong);
  - a Sanctioned Market (any Product to Russia), naming the regime and the year it became broad;
  - a Restricted Pair (whisky to Saudi Arabia), naming what the Market prohibits;
  - an Offset Pair (turbine parts `841191` to France), giving how far the UK sells above Export Potential in the Heading's other Products, the four-digit Heading (8411) and the Product carrying most of it (turbo-jets), and saying the Export Gap may be trade recorded under a neighbouring Product;
  - an Applied Tariff carried forward (e.g. `300490` to India).
- [ ] Export Potential, Actual Exports and Export Gap show as three figures ("2022–2024 average"), each with a tooltip in the API's wording.
- [ ] The facts run: Applied Tariff, UK Trade Agreement in force, the Blocked Opportunity value (illustrative), the Opportunity Type with its meaning, the Relative Export Gap and the Potential Range. The Potential Range's tooltip describes it in plain words (comparable pairs, never a confidence interval). There is no Potential Range bar and no Uncertain Opportunity warning.
- [ ] Under the facts, the panel shows the pair's Market as one **Market Reach** sentence, the same one the Market filter shows.
- [ ] A pair with no UK Trade Agreement (whisky to India) shows **The Barrier**: its Applied Tariff and the amount the model associates with adding an agreement, marked Illustrative, as a warning when the pair is a Blocked Opportunity. A pair with an agreement (whisky to Japan) shows none.
- [ ] The factors are in plain words (no raw feature names), phrased as associations, as a bar chart (raised in blue, lowered in red, readable in light and dark mode) and a table. **All factors** lists every one.
- [ ] The year-by-year chart sets Actual Exports against Export Potential. **Figures by year** shows the last six years in $ millions with each year's Applied Tariff mark, and the three 2022–2024 yearly gaps average to the Export Gap.
- [ ] Open `?product=220830&market=JPN` in a fresh tab: the panel explains that pair.

**Blocked Opportunities**
- [ ] The page lists only Blocked Opportunities (2,128 on the served artefact), with no "Rank by" toggle; the Leaderboard does not offer that order.
- [ ] The line above the tables gives the filters in force, the number of Blocked Opportunities and their total value, labelled illustrative. One caption gives the unit ($ millions), the Baseline year, the fixed thresholds (20% and $1M), then the Leaderboard's median-type note with its N%.
- [ ] The same three views as the Leaderboard. **By Market** (the default): the Markets table (Market, Blocked Opportunities, "Total value, illustrative", its tooltip saying illustrative too) on the left, largest total first, with the top Market and its top Blocked Opportunity chosen. On the right, a line names the Market and its number of Products, and a second gives its total value, labelled illustrative. Click any value of another Market: its row is tinted, and the lines and the Products table follow.
- [ ] Under the Markets table, a note says why a Market's total differs from a Scenario, with a link "Scenario Builder: a Trade Agreement with <Market>" for the chosen Market.
- [ ] The right table (Product, Barrier, Actual Exports, Export Potential, With a Trade Agreement, Rise with agreement, Rise %, Tariff part, Above Export Potential) lists every Blocked Opportunity of the chosen Market, in the sidebar's order. **By Product** swaps the tables, with Market in place of Product. **Wide table** lists every Blocked Opportunity with both.
- [ ] Under the tables at full width, the chosen pair's panel opens with a box naming the pair and its Barrier, then the before-and-after chart (Actual Exports, Export Potential, Export Potential with a Trade Agreement), the sentence splitting the rise into its Tariff part and Agreement part, the caveats, then the facts, factors and years. A pair chosen in one view stays chosen in another that shows it.
- [ ] Choose By Product: the URL gains `?view=product`, and that link opens By Product. The Leaderboard's view is kept apart.
- [ ] The Export Gap appears nowhere on the page except the shared footer.
- [ ] The Sector, region, "Show only pairs worth at least…" and hide filters, and a Product search, each narrow the tables, the line and the CSV. Reset filters restores them. They are kept apart from the Leaderboard's filters. There is no Market, Opportunity Type or Trade Agreement filter.
- [ ] Under the tables, an expander ("Illustrative: what these figures are read against") holds the PPML gravity benchmark, the model's association and the Event Backtest, or says one was not run for this version.
- [ ] Every caption describes the value as what the model associates with adding a Trade Agreement, not as a cause or a forecast.
- [ ] Export the CSV: every filtered Blocked Opportunity, across all Markets, in full dollars.

**Scenario Builder**
- [ ] The menu item reads "Scenario Builder" and opens `/Scenario_Builder`; the subtitle starts with "Build a Scenario". On Blocked Opportunities, choose India and click "Scenario Builder: a Trade Agreement with India": the builder opens with Custom Trade Agreements adding India, not yet run.
- [ ] The builder sits in a box at the top, and the page shows no result until **Run Scenario** is pressed. Change a control after a run: a line says the controls differ from the result shown. Run a Scenario, open another page from the menu and come back: the controls and the result are as left, with no such line.
- [ ] Run a Trading Bloc preset (e.g. EU re-join): it returns within a few seconds. The summary sentence frames the result as an association and states no figure. The metrics beside it state the Scenario Effect and its share of Baseline once each, then come the change per Market and the top Products gaining. No caption repeats the association caveat.
- [ ] Run the CPTPP preset: a non-zero Scenario Effect (the 2024 Baseline has no UK CPTPP membership).
- [ ] Add a Trade Agreement with the USA: the Blocked Opportunities metric and section match the Blocked Opportunities page's count and figures for the USA, in that page's Products table with a Market column, under the fixed rule (20% and $1M) the caption states. The caption says Sanctioned Markets, Restricted Pairs and Offset Pairs are hidden, links to the Blocked Opportunities page with them shown, and explains why the Scenario Effect (+$18.2bn) differs from the Market's Blocked Opportunity total ($13.1bn). Clicking a pair opens that page's before/after panel. The builder has no threshold inputs.
- [ ] Run a tariff change or EU re-join (neither adds a Trade Agreement where none is in force): no Blocked Opportunities metric or section, only a line saying the Scenario adds none, stating the fixed rule (20% of Baseline Export Potential and $1M) and linking to the Blocked Opportunities page.
- [ ] Click any value of a Market row in Change per Market: the row is tinted, and that Market's most affected pairs are listed under it, largest effect first. The top Market is selected on open.
- [ ] Click any value of a pair in the drill-down (none is tinted before): the row is tinted, and a box under the table shows its caveats, its Baseline and Scenario Export Potential, the Scenario Effect and both Applied Tariffs. Its link opens the Leaderboard on that Product and Market.
- [ ] Add a Trade Agreement with the USA and type `840890` in the Product search above the drill-down: the table gives way to that pair's box, though the pair is outside the top 50. Clear the search: the table returns. Search a Product the Scenario leaves unchanged in that Market: a line says so.
- [ ] The tables are in $ millions, the unit stated once above them. The caption above says a Market or a pair can be clicked (not "any value": Top Products gaining cannot) and ends with the median-type note, with its N%.
- [ ] The drill-down has a Caveats column: add a Trade Agreement with Saudi Arabia, and its alcoholic beverages read "Restricted Pair"; an Offset Pair's row reads "Offset Pair".
- [ ] A Scenario that changes nothing (an agreement with Japan, which already exists) says so in a neutral info box. An invalid one (Custom Trade Agreements with nothing chosen) is a warning. The preset list leaves out a preset that changes nothing in the Baseline year (Leave, in 2024).
- [ ] In tariff mode, set a tariff above the training range for one Market (5000% for the USA; the range ends at 3,000% on the real artefact): the Scenario Effect is never a rise, and a neutral info box says the effect flattens beyond the training range. A line under the result names the rate the model reads, the edge of the range, and counts against it.
- [ ] Set the USA's tariff to 10%: a line under the result says how many pairs it sets to 10%, how many it raises and how many it lowers (vehicles at 25% are lowered).
- [ ] Remove a Trade Agreement (e.g. with Japan): the pair tables show the restored 2021 MFN tariff.
- [ ] After a run, the Illustrative note sits closed under the metrics, in the same expander as on the Blocked Opportunities page; opened, it states the EU–Japan Event Backtest result (predicted against observed).

**Model Evaluation**
- [ ] Four figures open the page: the rank correlation for UK pairs with the Naive Baseline's beside it, the share of Actual Exports inside the Potential Range against its 80% target, size-matched Lift, and ranking stability. Under them, "Artefact ID" names the served artefact folder. Every section below is folded except "How accurate is the model?" and "Known limitations".
- [ ] The plain-English accuracy headline renders, green if the model beats the Naive Baseline and amber with a plain warning if not, with the test year and the non-UK training sample stated below it.
- [ ] Under the headline, one table sets every Market against the 30 largest and 60 largest importers, model against Naive Baseline.
- [ ] "How the model was chosen" shows the "Alternatives considered on 2022" table, then the confirmation on all five folds with columns "Chosen Model Form" and "The model alone". The page shows no version names, no table of earlier versions and nothing "traded away".
- [ ] "What Market Reach changes" states, with it and without it, the tie-proof Selection Measure, the 30 largest and 60 largest Markets, Lift and size-matched Lift, and the share of UK pairs inside their Potential Range, and says Scenario Effects are unchanged by it.
- [ ] The metric tables (model against Naive Baseline) render for UK pairs and for all Exporters.
- [ ] "Where the model is weakest" lists the weakest Sectors and regions; switching "Rows evaluated" changes the tables.
- [ ] "Observed against carried-forward tariffs" splits the test rows by Applied Tariff (observed, carried forward from 2021) for both scopes, with a sentence saying how they split (on 2023 every row is carried forward).
- [ ] The stability sentence renders, followed by the Potential Range's held-out coverage against its 80% target, the share of Opportunities whose Actual Exports lie inside their Potential Range (92% on the served version), a coverage table by size of Export Potential (under $2m, $2m–10m, $10m–100m, over $100m), the share of UK pairs whose Actual Exports lie inside their Potential Range, and a coverage table per Exporter.
- [ ] The Lift section states how many times more often the largest Export Gaps grew, with a 95% interval, worded as a tendency. It shows size-matched Lift next to raw Lift, explains that raw Lift mostly reflects pair size, and shows the growth share for each tenth of Export Gaps.
- [ ] The Event Backtest shows, for EU exports to Japan and Japan's exports to the EU, a table of predicted against observed change per year. It names the control Markets, says the Applied Tariff is 0 with the agreement on and the MFN rate with it off, and states the COVID caveat. The UK–Australia/New Zealand secondary check follows.
- [ ] "Excluded Products", just above "Known limitations", says how many Products are left out, lists each reason with its number of Products, and holds the full list (HS6, Product, Sector by name, reason) in an expander. No ranking page shows an Excluded Products list.
- [ ] "Known limitations" renders with no "no tariffs" entry. It covers the result limitations (such as "Does not beat a naive ranking for UK pairs" when it does not), goods only, tariffs carried forward from 2021, Excluded Products, the top-100 Exporter scope, nominal US dollars, what the Potential Range assumes, associations not causes, provisional 2024, the 2012 edition, reporting quality and sanctions.

**How the Model Works**
- [ ] The page is last in the menu. Three headings (How the model is built, How the model is checked, What it produces and how it is served) each list their ideas in bold, with a one-sentence summary.
- [ ] "The test year, read once" is one sentence: every choice was made on 2022, and the model was tested once on 2023. Under every other idea, three folded layers open: What it is, Why this project uses it, How it is done here. The boosting, Selection Measure, blend, Potential Range, Market Reach and gravity formulas render as maths. The page names no course notes and no ADR file.
- [ ] The figures match the served model: "The data" states the 100 largest economies and the sampling rates (5% of Zero Cells, then about 59% of other Exporters' rows); "The blend with the Naive Baseline" the weights 0.7 and 0.3; "Market Reach" the strength 4; "The Potential Range" 80% and 20 bands.
