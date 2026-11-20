# Decision log: spec #103 (App usability pass)

Integration branch: `spec/103-app-usability` (based on main 4e36c1d).

## orchestrator: How to order #107 against #105?
- Options: run #107 in parallel with #105; run #107 after #105 merges.
- Chose: run #107 after #105 merges, in parallel with #106.
- Why: #107's Excluded Products list shows Sector names, which #105 owns (static HS 2012 chapter table), so waiting avoids two versions of the names.

## orchestrator: Who owns Baseline-year Actual Exports (shared by #109 and #110)?
- Options: each ticket writes its own; one owner.
- Chose: #110 owns the package function that reads Baseline-year Actual Exports from the yearly history (0 where no Flow); #109 calls it.
- Why: two versions would drift apart on data the fixture does not cover.

## orchestrator: Draft PR or local integration branch?
- Options: open a draft PR; close tickets with comments and report the branch.
- Chose: no PR; close each ticket with a comment and report the integration branch `spec/103-app-usability`.
- Why: this repo's issue tracker closes work with `gh issue close`, earlier specs merged without PRs, and nobody asked for one.

## #104: What to do with Product → Markets and Market → Products until they are deleted?
- Options: keep them in the menu; drop them from `st.navigation` now; declare them hidden (`visibility="hidden"`).
- Chose: hidden, at their old URL paths, like Why?.
- Why: the menu shows exactly the four pages, the old links and their tests keep working, and the deleting tickets only remove two `st.Page` lines and the files.

## #104: Where does the theme file live?
- Options: `.streamlit/config.toml` at the repo root; `app/.streamlit/config.toml` beside the entry point.
- Chose: `app/.streamlit/config.toml`.
- Why: Streamlit 1.67 reads config beside the entry script, and the Containerfile copies only `app/`, so the container gets the theme with no build change.

## #104: How are the page files named and where do they serve?
- Options: keep the numbered `pages/N_*.py` files; rename them by page.
- Chose: rename to `pages/leaderboard.py`, `blocked_opportunities.py`, `scenarios.py`, `model_evaluation.py` (and the three hidden pages), with `url_path` set to the old paths (`Blocked_Opportunities`, `Scenarios`, `Model_Evaluation`, `Why`, ...).
- Why: numbers only ordered the old `pages/` menu, which `st.navigation` ignores; keeping the URL paths keeps existing links and `why_link` working.

## #104: Where do the caution line and the page frame live?
- Options: the caution line in the app; in a new package module; in `exportspredictor.wording`. The frame in `app/ui.py`; in a new `app/frame.py`.
- Chose: `wording.CAUTION`, served as `/labels` `caution`; the frame in `app/frame.py`.
- Why: the spec says the package words it; `wording` already holds shared plain-text pieces; a separate frame module keeps `ui.py` to cached reads and tables.

## #105: How are the Leaderboard's filters kept in the URL?
- Options: Streamlit's `bind="query-params"` on each widget; read and write `st.query_params` by hand.
- Chose: by hand, in `app/filters.py`, with `persist_state="session"` on each widget for the session.
- Why: a bound selectbox writes its displayed label to the URL (a Market's name), but the spec wants the ISO3 and HS6 codes.
- Detail: a URL naming any filter that the page did not write itself sets every filter (named ones to its values, the rest to defaults); a URL naming none (a return through the menu) keeps the session's filters and writes them back.

## #105: What are the query-parameter names and values?
- Options: short names (`q`, `type`, `agreement`); the existing widget keys.
- Chose: the widget keys: `product` (HS6), `market` (ISO3), `sector` (HS2 code), `region`, `opportunity_type`, `trade_agreement` (yes/no), `min_gap_usd`, `hide_flagged`/`hide_hubs` (true), `rank_by`, `min_potential_usd`.
- Why: they match the API's query parameters (except `product`, which is also the Leaderboard read's name), so a link reads like the read it makes.

## #105: What do the counts in "N pairs shown of M" mean?
- Options: M is every UK pair; M is every pair passing the filters.
- Chose: N is the rows shown, M the pairs passing the filters (the summary's count).
- Why: it matches #106's "top 500 of N pairs" and says how many rows are held back.

## #105: Where does the Sectors read take its Sectors from, and is the Excluded Product's Sector in it?
- Options: Sectors of ranked Products only; Sectors of every Product in the artefact's lookup.
- Chose: every Product in the lookup, Excluded Products included.
- Why: the spec says "each Sector that has Products in the artefact", and #107's Excluded Products list shows their Sector names.

## #105: What happens to the Blocked Opportunities page's filters and the Excluded Products expander now?
- Options: rework them here; leave them to #109 and #107.
- Chose: leave both; the Blocked page's filters only get their own widget keys (`blocked_` prefix) and Sector names.
- Why: #109 rebuilds that page's filters and #107 removes the expander from every ranking page; separate keys keep the two pages' state apart now.

## #105: Does the Product search still narrow its matches to the chosen Sector?
- Options: keep narrowing; search all Products.
- Chose: search all Products; the Product and Sector filters combine in the API read.
- Why: a Product kept in the URL must not vanish from the search because of another filter, and a mismatch shows as zero rows with both filters named.

## #107: Can the Excluded Products section be an expander when it holds the full list in an expander?
- Options: nest the full list's expander inside the section's expander; make the section a plain heading; show the full list behind a toggle.
- Chose: nest the expanders.
- Why: the spec asks for both, and Streamlit 1.67 allows nesting (its docs only advise against it); the inner one is the only nested expander on the page.

## #107: What does each of the four figures show beside its value?
- Options: a bare value; the value with a short grey comparison under it (`st.metric` delta, colour and arrow off).
- Chose: the value with a grey comparison: the Naive Baseline's rank correlation; the Potential Range's target; what above 1 means for size-matched Lift; the share of the top Opportunities the same in both years. Each has a help tooltip; a figure the artefact lacks shows "n/a".
- Why: the spec says each figure is read "against" something, and a green or red arrow would read a comparison as good or bad news.

## #107: How are the reasons ordered and where is the Excluded Products list built?
- Options: count the reasons in the API; count them in the page from `/products/excluded`. Order by count; by the list's order.
- Chose: the page counts the rows of `/products/excluded` (one read), most Products first, ties in the list's order; `app.ui.excluded_products_list` is removed with the ranking pages' expander.
- Why: grouping rows already served is presentation, needs no API change, and keeps one read.

## #106: Where is the Leaderboard's new table built, and does the Blocked Opportunities page switch to it?
- Options: rework `ui.show_ranking` for both pages; build the Leaderboard's table in a new module and leave `show_ranking` to the Blocked page.
- Chose: a new `app/tables.py` (money, heading and tariff helpers plus `leaderboard_frame` / `leaderboard_columns` / `leaderboard_csv`); the Blocked page keeps `show_ranking` (long headings, full dollars) until #109 rebuilds it.
- Why: #109 replaces that page's table anyway, and keeping it untouched avoids colliding with #107 and #109.

## #106: What is the Show all control and where does it sit?
- Options: a checkbox in the sidebar; a toggle above the table; a button.
- Chose: a toggle "Show all N pairs" (key `show_all`) under the active-filters line, drawn by `filters.show_all(pairs)`.
- Why: it belongs to the table, not to the filters, and a toggle states on/off; the URL and Reset handling stay in `app/filters.py` with the other parameters.

## #106: What headings does the CSV use?
- Options: the table's short headings; the short headings for text columns and full headings with period and unit for money.
- Chose: short headings for text columns; money as "Export Gap, 2022–2024 average (USD, nominal)" and "Blocked Opportunity value, 2024, illustrative (USD, nominal)"; the Potential Range as two numeric low/high columns in dollars.
- Why: a CSV has no line above it to state the unit and period, and a spreadsheet needs the range as numbers.

## #106: Does `/labels` now need loaded artefacts?
- Options: keep it artefact-free and leave the Gap Period out of the terms; make it depend on the loaded artefact.
- Chose: it depends on the loaded artefact (503 until one loads), so the terms carry the artefact's Gap Period label.
- Why: the spec takes that label from the artefact; the router runs no page until the API is ready, so the app never reads `/labels` before then.

## #106: What does the hidden Sector column show?
- Options: the HS2 code; the Sector name.
- Chose: the Sector name (`sector_name`); its tooltip says it is the HS2 chapter.
- Why: the spec lists Sectors by name everywhere; the CSV carries the same name.

## Merge #106: conflict in `app/pages/leaderboard.py` with #107
- Options: keep #107's `show_leaderboard` call; keep #106's table and drop the call to `excluded_products_list`.
- Chose: #106's table, without the `excluded_products_list` call and import (#107 removed the function).
- Why: each ticket keeps what it set out to do: #106 owns the table and #107 owns where Excluded Products are listed.

## #110: Where does Baseline-year Actual Exports live, and what is it called?
- Options: a column built in the API; a package function next to `gap_period.baseline_year_scores`.
- Chose: `exportspredictor.gap_period.with_baseline_actual_exports(pairs, history, baseline_year)`, adding `baseline_actual_exports_usd` (0 where the history has 0 or no row); Scenario pairs carry that field.
- Why: #109 calls the same function, and the name says which year it is, apart from the Gap Period mean `actual_exports_usd`.

## #110: Does a pair's box show before any pair is clicked?
- Options: select the top pair on open (as Markets are); show no box until a pair is clicked.
- Chose: no box until clicked (`selection_mode="single-row"`); the Markets table opens on the top Market.
- Why: the spec opens only the Market on the top row; two boxes open by default would push Top Products and Blocked Opportunities down.
- Reversed in part by the final review (#103): the tables use `pick_row`'s click-any-value tint, not `selection_mode="single-row"`; a pair's box still waits for a click (`tables.pick_row_if_chosen`).

## #110: How does the page remember the selected Market and pairs?
- Options: read the tables' selection state; remember the choice in session state through an `on_select` callback.
- Chose: callbacks write `scenario_market` (ISO3) and `scenario_drill_pair` / `scenario_blocked_pair` ({hs6, iso3}); table keys carry the run (and the drill-down's Market), and Run Scenario clears the three.
- Why: AppTest cannot click a row, so tests set the remembered choice (spec, Testing Decisions); keys tied to the run stop an old row position from selecting a different pair in a new result.
- Amended by the final review (#103): `pick_row`'s callback now writes the same three keys.

## #110: Link or button from the pair box to the Leaderboard?
- Options: `st.page_link` with `query_params`; a button calling `st.switch_page`.
- Chose: `st.page_link("pages/leaderboard.py", query_params={"product", "market"})`.
- Why: the ticket asks for a link, it stays in the same session like `switch_page`, and AppTest exposes its page and query string for the test.

## #110: Which columns do the Scenario tables show?
- Options: keep HS6 and Market in every table; name each row by Product (and Market where it varies) only.
- Chose: Change per Market: Market, then the effect columns and Pairs changed; drill-down: Product, effects, both tariffs (its heading names the Market); Top Products: Product, effects, Pairs changed; Blocked Opportunities: Product, Market, effects, Actual Exports, both tariffs. The HS6 code is in the pair box.
- Why: fewer columns keep the tables narrow, and the spec's short headings list names no HS6 column.

## #110: Tariff Sectors in the builder
- Options: keep bare HS2 codes (`app.ui.SECTORS`); offer `/sectors` by name.
- Chose: `/sectors`, shown as "name (code)"; the request still sends codes.
- Why: the spec lists Sectors by name everywhere; `app.ui.SECTORS` now has no user and can go.

## #108: How is the facts table drawn, given the Potential Range row needs its own tooltip?
- Options: an `st.dataframe` of name/value rows; one `st.markdown` line per fact with `help`.
- Chose: one markdown line per fact in a bordered container, each with a tooltip where it needs one (the Potential Range's is the API's sentence).
- Why: a dataframe has no per-row tooltip, and the spec asks for one on the Potential Range row.

## #108: How is a row chosen by clicking any value, and how is the whole row tinted?
- Options: single-row selection (a tick-box column); single-cell selection with a callback that remembers the clicked row's pair, and a pandas Styler tint.
- Chose: single-cell selection, an `on_select` callback storing (HS6, ISO3) in session state, and a Styler tint on that row, in a reusable `app.tables.pick_row`.
- Why: the spec wants no tick-box column and the whole row tinted; a callback updates the pair only on a real click, so a stale cell selection never re-chooses a row after the filters change. Styler on 6,000 × 16 cells takes about 0.4 s.

## #108: What data does the narrow table carry?
- Options: the full frame with three columns in `column_order`; only Product, Market and Export Gap.
- Chose: only the three columns; the wide view carries every column and the Show/hide menu.
- Why: the spec's narrow table is three columns; a full frame would offer 13 more in its column menu and triple the Styler cost.

## #108: Where does the Opportunity Type's meaning in the facts come from?
- Options: the full term definition; the type's own sentence picked out of the API's definition.
- Chose: the type's own sentence from the API's `opportunity_type` definition (the full definition as the tooltip), falling back to the whole text.
- Why: the spec asks for the type "with its meaning", and the API owns the wording.

## #108: Is "Why?" reworded outside the app (notebook, model card, ADR 0007)?
- Options: reword them now; leave them.
- Chose: leave them, noted in NOTES-108.md.
- Why: the spec leaves the notebook and model card untouched, and ADRs record past decisions.

## #109: How are Blocked Opportunities tested when the fixture has none?
- Options: lower the thresholds in tests; change the fixture world; a scaled copy of the fixture artefacts.
- Chose: `conftest.blocked_artefacts` (served by `blocked_api_url`): every `_usd` column of the Baseline and history × 1,000, giving four Blocked Opportunities (two in India, two in Hong Kong), one of them (cars, Hong Kong) above Export Potential with a Trade Agreement.
- Why: the thresholds are fixed on this read, and changing the fixture world would move every other test's figures.

## #109: Does the Blocked Opportunities read keep a Market filter?
- Options: keep `market`; drop it (422).
- Chose: drop it, with `opportunity_type`, `trade_agreement`, `min_gap_usd` and `rank_by`; the read has its own `BlockedQuery` and adds `min_value_usd`.
- Why: the spec lists the read's filters without a Market, and the page filters by Market itself from the one read.

## #109: How does the shared panel leave out the Export Gap?
- Options: reuse it unchanged; a `with_gap=False` option on `panel.from_facts`.
- Chose: `with_gap=False` drops the Relative Export Gap and Potential Range facts and the yearly table's Export Gap column and caption sentence.
- Why: #102 takes the Export Gap out of this page's panel, and the Potential Range's sentence speaks of the Export Gap.
- Amended by the final review (#103): the Potential Range fact is back (it is not an Export Gap); `with_gap=False` drops only the Relative Export Gap and the yearly Export Gap, and the Potential Range's Uncertain Opportunity sentence no longer names the Export Gap.

## #109: Does the shared footer, which names the Export Gaps' period, stay on the Blocked Opportunities page?
- Options: drop that sentence on this page; keep the footer as on every page.
- Chose: keep it; the test's "no Export Gap" check skips the footer.
- Why: the spec fixes the footer's wording on every page, and it states the app's periods, not a figure of this page.

## #109: Where do the Blocked Opportunities filters live?
- Options: keep `ui.blocked_filters`; generalise `app/filters.py`.
- Chose: `filters.blocked_filters()` beside `leaderboard_filters()`, sharing the URL code with a key prefix (`blocked_`); URL names are the API's (`product`, `sector`, `region`, `min_value_usd`, `hide_flagged`, `hide_hubs`).
- Why: one place reads and writes filters in the URL, and the prefix keeps the two pages' state apart.

## #109: Is Export Potential with a Trade Agreement a new glossary term?
- Options: a new entry; a sentence in the Blocked Opportunity value entry.
- Chose: a sentence in the Blocked Opportunity value entry.
- Why: #102 adds no new term; the figure is Baseline Export Potential plus the value, named where the value is defined.

## #111: How is the research note brought onto the branch?
- Options: `git merge research/streamlit-usability`; check out only the note.
- Chose: the merge.
- Why: the branch carries only the note (one commit adding `docs/research/streamlit-native-usability.md`).

## #111: How are table columns kept in view at 1280 px, given the spec forbids pinning?
- Options: pin Product; fixed pixel widths; leave autosizing.
- Chose: a fixed 200 px Product column on every table (`app.tables.product_column`), a "medium" Factor column, fixed Market and count widths in the Blocked Markets table, and that table a third of the row.
- Why: autosizing let a long Product name push Market and Export Gap out of the narrow Leaderboard table; pinning is ruled out (faint text); widths are native (research note) and change no decided layout.

## #111: How are the panel's figures made to fit?
- Options: theme `metricValueFontSize`; CSS on the panel; a wider panel.
- Chose: `metricValueFontSize = "1.5rem"` under `[theme]`.
- Why: a native option that keeps the app following light and dark mode (checked in the browser); a wider panel would change #108's split. Labels still cut: reported.

## #111: Is a slower-than-a-second Show-all click fixed?
- Options: redesign the tint or selection; report it.
- Chose: report it (1.4–1.7 s on 5,138 rows).
- Why: the spec says to report it on the ticket and leave the design.

## #111: Layout fixes without a test?
- Options: assert column widths in AppTest; verify in the browser only.
- Chose: browser only, with screenshots.
- Why: the spec's Testing Decisions rule out asserting widths and layout; wording fixes found in the browser got tests.

## #103: Where does the app read the ranking row cap?
- Options: a new limits read; a field on `/labels`.
- Chose: `/labels` `max_rows` (`rankings.MAX_ROWS`), read by `app.ui.read_limit()`.
- Why: `/labels` already serves the app's options and is cached, so no page makes another read.

## #103: How does the Potential Range tooltip avoid naming the Export Gap on Blocked Opportunities?
- Options: an app-side variant of the API's sentence; reword the package's Uncertain Opportunity sentence.
- Chose: reword it in `conformal.potential_range_text` ("lie inside it, within the variation normally seen among comparable pairs").
- Why: one wording for both panels, same meaning as the glossary's Uncertain Opportunity, and the Blocked page shows no Export Gap.

## #103: How are Excluded Product reasons counted and served?
- Options: change `/products/excluded` into an object with counts; a new read.
- Chose: `exclusions.reason_counts` served at `/products/excluded/reasons` (reason, products; most first).
- Why: the list's contract stays as it is, and counting lives in the package.

## #103: Scenario pair tables: top row chosen by default or none?
- Options: `pick_row` as is (top row tinted, two boxes open); a variant with no row chosen until a click.
- Chose: `tables.pick_row_if_chosen` for the drill-down and Blocked Opportunities; `pick_row` for Change per Market.
- Why: keeps #110's no-box-until-clicked decision while matching the other pages' click and tint.
