# Streamlit's native options for tables, filters, navigation and logo

Researched 2026-11-19 for issue [#94](https://github.com/noahlb1/exportspredictor/issues/94), a child of the map [#93](https://github.com/noahlb1/exportspredictor/issues/93) (App usability pass).

Scope: what Streamlit **1.67.0** (the version in `uv.lock`, released 2026-11-05) does natively, with no custom components. Every API below was read from the docstrings of the installed package (`.venv/Lib/site-packages/streamlit`, 1.67.0), which are the text of the official API reference. "Since" versions come from the official release notes. Anything not confirmed against one of those is marked **Unconfirmed**. Nothing here was run in a browser; the prototypes should check the items marked unconfirmed.

## Bottom line

1. **Almost everything the usability pass wants is native in 1.67.** Pinned columns, a user-facing column chooser, pixel column widths, row selection, a side-drawer dialog, sidebar filters, URL-bound filters, grouped navigation, a hidden Why? page with an unchanged URL, a logo, and lazy loading or paging of long rankings all exist.
2. **Not possible natively:**
   - **Wrapped or two-line column headings.** No parameter of `st.dataframe` or `st.column_config` wraps a heading. The fix is short headings plus `help=` tooltips and the unit stated once above the table.
   - **A number format that divides.** No `format` shows $ millions from full-dollar data. The table needs its own columns already divided by 1,000,000; the CSV is built from the full-dollar frame, as it is today.
   - **Pinning to the right, or a pinned column that the user cannot unpin or hide.** Pinning is left-side only, and every column stays in the user's column menu.
   - **Truly hiding a column.** A hidden column can always be shown again by the user from the toolbar.
   - **Lazy loading together with row selection, search, CSV download or `pandas.Styler`.** Lazy mode turns those off.
   - **A nested URL** such as `/pair/why`. `url_path` cannot contain a slash.
3. **A change of behaviour to plan for:** the Why? link is a `LinkColumn` URL today. Opening an internal page by URL **starts a new session**, so the Leaderboard's filters are lost on the way back. Three native replacements avoid that: row selection driving a detail panel or `st.dialog`, a `ButtonColumn` whose callback calls `st.switch_page(..., query_params=...)`, or filters bound to the URL with `bind="query-params"` (which survive a new session because they live in the URL).
4. **500 rows is far below any limit.** Up to 150,000 rows are sent to the browser with every feature on. Above that, or with `lazy=True`, rows load on demand. The practical ceiling for this app is the API's JSON payload, not the grid.

## 1. Wide tables in `st.dataframe`

| Need | API in 1.67.0 | Since | Limits |
|---|---|---|---|
| Pinned (frozen) columns | `pinned=True` on any `st.column_config.*Column(...)` | 1.41 (config); users can pin and unpin from the column menu since 1.42 | Left side only. Index columns are pinned by default. The user can unpin from the column menu. |
| Hide columns by default | `column_order=[...]` (columns left out are hidden), or `column_config={"col": None}` | Long-standing | Both are a default, not a lock: "Hidden columns can still be shown by the user via the table toolbar." Sensitive data must be dropped from the frame. |
| User-facing column chooser | Built in: the toolbar's **Show/hide columns** menu, and **Hide column** in each column's header menu | Users can hide columns since 1.43; hide or unhide all since 1.49; the visibility menu is always shown since 1.56 | No API to pre-set or read the user's choice, and none to remove the chooser. Whether the choice survives a rerun that changes the data is **Unconfirmed**. |
| Column widths | `width="small"` (75px), `"medium"` (200px), `"large"` (400px) or an integer in pixels, on any column config | Pixel widths since 1.47 | If the columns total less than the table width, the spare width is shared out among all columns. Users can drag borders or pick **Autosize** (1.43). |
| Table width and height | `st.dataframe(width="stretch" \| "content" \| int, height="auto" \| "content" \| "stretch" \| int)` | `width="content"` 1.52; flex sizes 1.49 | `height="auto"` shows at most ten rows; `"content"` is capped at 10,000px. `use_container_width` is deprecated. |
| Row height | `row_height=<px>` | 1.43 | One height for all rows. It does not wrap headings. |
| Heading text and tooltip | `label` (first argument) and `help=` (Markdown tooltip) on a column config | Long-standing | One line. See below. |
| Cell alignment | `alignment="left" \| "center" \| "right"` | 1.56 | Cells only. |
| Header colours | `theme.dataframeHeaderBackgroundColor` (1.47), `theme.dataframeHeaderTextColor` (1.65), `theme.dataframeBorderColor` (1.46) in `config.toml` | as listed | App-wide, not per table. |
| Accessible name | `alt="Top Opportunities by Export Gap"` | 1.65 | A short name, not a description. |

### Wrapped or two-line headings: not available

No parameter in the 1.67.0 signatures of `st.dataframe`, `Column`, `NumberColumn` or any other column type wraps a heading or sets a header height, and the dataframes concept page lists no such feature. How an over-long heading is drawn (cut off at the column width, presumably) is **Unconfirmed**.

Workarounds, all native:

- Short headings ("Export Gap") with the qualifier in `help=` and the unit and Gap Period stated once in a caption above the table. This matches the decision on the map to state the unit once.
- A pandas `MultiIndex` on the columns gives a two-level header ("Dataframes support multi-index headers", 1.39), for example a top level naming the Gap Period over three columns. Limits: it is incompatible with `lazy=True` (the docstring names it as a case that raises), and how `column_config` keys address such columns is **Unconfirmed**.
- `st.table` is static: no sorting, pinning, selection or column chooser.
- **Unconfirmed:** whether a newline in a label renders as a second line. Nothing in the docs says it does; assume not.

### Number formats, and $ millions from full-dollar data

`st.column_config.NumberColumn(format=...)` accepts, in 1.67.0: `None`, `"plain"`, `"localized"`, `"percent"`, `"dollar"`, `"euro"`, `"yen"`, `"accounting"`, `"compact"`, `"scientific"`, `"engineering"`, `"bytes"`, `"duration"`, or a printf string with a prefix and suffix (`"£ %.2f"`; `"%,d"` for thousands separators; sprintf-js syntax). Presets arrived in 1.43 (`"yen"` 1.46, `"bytes"` 1.47). With a preset, `step` overrides the display precision (1.47); with a printf string it does not.

**No format scales the value.** A format only decides how the number already in the cell is written. So:

- **To show $ millions to one decimal**, build display columns divided by 1,000,000 and format them `"%.1f"`. Sorting still works, because the cell holds a number. The app already builds the CSV from its own frame (`table.to_csv`), so the CSV keeps full dollars as long as it is built from the undivided frame. The toolbar's own **Download as CSV** would export the divided values; the `client.disableDataExport` config option removes it (and clipboard copy) app-wide (1.60).
- **A very small value** under `"%.1f"`: anything below $50,000 shows as `0.0`, and a small negative Export Gap as `-0.0`. This is what printf rounding gives; it is **Unconfirmed** in the browser. The format cannot show "<0.1". If that matters, the choices are two decimals, or a text column built in Python (which then sorts as text, so the order would have to stay the API's).
- With **no** format on a float column, Streamlit shows the first significant figure of a small value instead of 0 ("small float values display their first significant figure instead of displaying as 0", 1.42), but the number of decimals then varies down the column.
- `"compact"` on full-dollar data gives an adaptive form ("1.2K", "1.2M"). That is the form the map kept for summary metrics, not the fixed $ millions it settled for table columns.
- `"dollar"` writes "$1,234.57": full dollars with cents.
- Users can change a numeric column's format themselves from the header menu's **Format** item (1.43). There is no API to remove that item.
- `pandas.Styler` can show custom cell text over the true values, but `column_config` formatting takes precedence over it, lazy loading does not support it, and whether sorting then uses the true values is **Unconfirmed**. Pre-dividing is simpler.

## 2. Row selection

API: `event = st.dataframe(df, key="ranking", on_select="rerun", selection_mode="single-row")`.

- `on_select`: `"ignore"` (default), `"rerun"`, or a callable (run as a callback before the script). With anything but `"ignore"` the call returns a `DataframeState`; the selected rows are `event.selection.rows`, a list of integer positions. Since 1.35.
- `selection_mode`: `"multi-row"` (default), `"single-row"`, `"single-row-required"` (exactly one row always selected, the first by default; 1.56), `"single-column"`, `"multi-column"`, `"single-cell"`, `"multi-cell"` (cells since 1.49), or a list of these. Column selection turns column sorting off; row selection does not.
- Row positions refer to the **original frame**, even after the user sorts in the browser, so `rows[event.selection.rows[0]]` is the right pair. Selections survive a user sort since 1.60.
- With a `key`, the selection is readable and writable at `st.session_state[key]` (programmatic selection since 1.56); `selection_default=` sets the first-render selection. Multi-cell selections cannot be set from code.
- The index column is hidden by default when row selection is on (1.51); the checkboxes are always visible (1.46).
- **Not available with `lazy=True`.**

### Can a selected row drive a detail panel or dialog without losing the filters? Yes

Selecting a row reruns the script in the same session. Every filter widget that is rendered again on that rerun keeps its value, so the filters are untouched. Two native shapes:

- **Detail panel in place**: after the table, `if event.selection.rows: ...` renders the pair's detail (the Why? content) below or beside the table. Simplest, and nothing to dismiss.
- **Dialog or side drawer**: a function decorated with `@st.dialog(title, width="small" | "medium" | "large" | <px>, position="center" | "left" | "right", dismissible=True, icon=None, on_dismiss="ignore" | "rerun" | callable)` and called when a row is selected. `position="left"` or `"right"` opens a full-height, user-resizable drawer (1.65); pixel width since 1.67; `on_dismiss` and `dismissible` since 1.48. `st.dialog` has been generally available since 1.37.

Limits of the dialog route:

- Only one dialog can be open per script run, and a dialog is modal even as a drawer: the rest of the app is inert while it is open.
- A dialog behaves like a fragment: widgets inside it rerun only the dialog function.
- The selection outlives the dialog. Unless it is cleared, the next full rerun (any filter change) sees the same selected row and calls the dialog function again. Clear it in an `on_dismiss` callback by writing an empty selection to `st.session_state[key]` (a 1.65 fix made repeated clearing uncheck the row in the grid). The exact interplay is **Unconfirmed** and is the first thing the row-detail prototype should check.
- `st.sidebar` cannot be used inside a dialog.

### The other native row action: `ButtonColumn` (1.59)

`st.column_config.ButtonColumn(label, type="secondary" | "primary" | "tertiary", on_click=callback, key="why_click", pinned=..., width=...)` draws a button in every row; the cell value is the button's label (Material icons allowed). On a click, `st.session_state[key]` holds `row` (int) and `label` for that one rerun. `key` is required for clicks to do anything. Since 1.63, `st.switch_page()` takes effect inside a widget callback, so the callback can call `st.switch_page(why_page, query_params={"hs6": ..., "iso3": ...})` (query parameters on `st.switch_page` and `st.page_link` since 1.52). That reaches the Why? page **in the same session**, unlike today's `LinkColumn`. Whether `row` is the original position after a user sort is **Unconfirmed**.

## 3. Filters

- **Sidebar placement**: `st.sidebar.selectbox(...)` or `with st.sidebar:`. Any widget can go there. With `st.navigation(position="sidebar")` the page menu sits at the top of the sidebar and the filters below it; `expanded=` controls how many menu items show when the sidebar has other content. `st.set_page_config(initial_sidebar_state="auto" | "expanded" | "collapsed" | "locked" | <int>)`: `"locked"` stops users toggling the sidebar (1.59), and an integer sets its default width (1.53; what the integer does to the open or closed state is **Unconfirmed**).
- **Resetting widgets that have a `key`**: assign the default to `st.session_state[key]` **before** the widget is created in that run. In practice, a "Reset filters" button with `on_click=reset`, where `reset` sets each filter key to its default. A callback "runs as a prefix to the script run", so it is always early enough. Assigning after the widget has been created in the same run raises a `StreamlitAPIException` ("... cannot be modified after the widget with key ... is instantiated").
- **Keeping filter state across pages**: by default it is lost ("By default, widgets are not stateful between pages"; a widget that stops being rendered resets). In 1.67 there are two native fixes, both needing a `key`:
  - `persist_state="session"` on the widget (1.59) keeps the value for the whole session, including across page switches. `"page"` keeps it only while the user stays on the page.
  - `bind="query-params"` (below) keeps it in the URL, which also survives page switches and takes precedence over `persist_state`.
- **Filters in the URL**: `bind="query-params"` on the widget (1.55), with a `key`; the key is the query parameter's name. In 1.67.0 it is on `selectbox`, `multiselect`, `radio`, `checkbox`, `number_input`, the text inputs, the sliders, the date and time inputs, `pills` and `segmented_control`, `st.pagination`, and on `st.tabs` and `st.expander` (1.65). Behaviour:
  - Changing the widget updates the URL; opening the URL sets the widget. A shared link therefore reproduces the filtered view.
  - A value equal to the default is removed from the URL, so an unfiltered view has a clean URL.
  - An invalid value in the URL is ignored and removed.
  - A bound parameter cannot be set or deleted through `st.query_params`; change it through `st.session_state[key]`, and the URL follows (1.58). So the reset callback above also clears the URL.
  - Browser back and forward restore bound widgets (1.65).
  - The app's widget keys (`sector`, `region`, `market`, `min_gap_usd`, `opportunity_type`, `trade_agreement`, `hide_flagged`, `hide_hubs`, `rank_by`, `min_potential_usd`) would become public parameter names, so renaming a key later breaks shared links.
  - How option values are written to the URL (the option itself, or its `format_func` label) is **Unconfirmed**.
- **`st.query_params` by hand** (1.30): a dict-like object with `[]`, `.get_all()`, `.update()`, `.from_dict()` (1.34), `.to_dict()` and `.clear()`. Values are strings. It is what the Why? page needs for `hs6` and `iso3`, and it remains the tool for parameters that are not widgets. `st.switch_page` clears non-embed query parameters unless `query_params=` is passed.

## 4. Navigation: `st.navigation` / `st.Page` against `pages/`

The app uses the `pages/` folder today. `st.navigation` (1.36) replaces it: "As soon as any session of your app executes the `st.navigation` command, your app will ignore the `pages/` directory". The entrypoint (`app/Home.py`) becomes a router that runs on every rerun, holds the shared frame (`st.set_page_config`, `st.logo`, the banner) and ends with `st.navigation(...).run()`. The Leaderboard therefore has to move out of `Home.py` into its own page file or a callable, passed with `default=True` so that it stays at the root URL. The other page files can stay where they are in `app/pages/`.

- **Grouping**: pass a dict, `st.navigation({"Opportunities": [leaderboard, blocked], "Scenarios": [scenarios]})`. Each key is a section label (Markdown allowed). Sidebar sections are collapsible (1.47). `position="top"` puts the menu in the header with each section as a dropdown (1.46); an empty-string key lists its pages first, outside any dropdown (1.50). `position="hidden"` shows no menu, for a hand-built one from `st.page_link`. One level of grouping only. `expanded=` (1.39; integer form 1.56) sets how many items show before "View more".
- **Hiding a page from the menu**: `st.Page("pages/5_Why.py", visibility="hidden")` (1.55). The page stays routable: "Hidden pages defined by Python files or callables remain accessible by `st.page_link` and `st.switch_page`", and by direct URL provided it is passed to `st.navigation` during the session's first run. This fits the Why? page exactly.
- **Renaming without changing URLs**: `title=` sets the menu label and browser-tab title; `url_path=` sets the URL. They are independent. If `url_path` is omitted it is inferred from the filename with the leading number stripped, so `st.Page("pages/5_Why.py")` is still served at `/Why` and `3_Blocked_Opportunities.py` at `/Blocked_Opportunities`. To rename a file and keep its URL, set `url_path` explicitly. Limits: every `url_path` must be unique across all pages, hidden ones included; it cannot contain `/`; the default page is always served at the root.
- **Removed pages** (Product → Markets and Market → Products): once they are not passed to `st.navigation` their URLs stop resolving. Streamlit has no redirect; what an old link then shows is **Unconfirmed**.
- **Other `st.Page` parameters**: `icon=` (emoji or `":material/name:"`), `help=` (menu tooltip, also the default `st.page_link` tooltip; 1.67), `default=`. A page can also be an external URL (1.56), which then needs a `title`.
- **Linking and switching**: `st.page_link` and `st.switch_page(page, query_params=)` accept a path or a `Page` object. "Navigating to an internal page by URL starts a new session"; `st.switch_page` and `st.page_link` do not.

## 5. Logo and identity

- **`st.logo(image, *, size="medium", link=None, icon_image=None)`** (`size` since 1.39). Draws the logo in the upper-left corner of the app and its sidebar. `image` is anything `st.image` accepts (not a list) or, since 1.54, a single emoji or `":material/name:"`. Limits:
  - Maximum height 20px (`"small"`), 24px (`"medium"`) or 32px (`"large"`), and no wider than the sidebar. A logo is a small wordmark, not a banner.
  - `icon_image` is the variant shown when the sidebar is closed; pass a wide image as `image` and a square one as `icon_image`.
  - One image serves both light and dark mode. Either design a mark that reads on both, or fix the theme and set `client.toolbarMode="minimal"`, as the docstring advises.
  - Clicking it goes to the app's home page (1.53); `link=` accepts only an external `http(s)` URL.
  - The last call in a run wins, so call it once in the entrypoint. SVGs with a `viewBox` are supported (1.45).
- **Page icon (favicon)**: `st.set_page_config(page_icon=...)`: an image as for `st.image`, an emoji or shortcode, or `":material/name:"` (a Material icon favicon is always black). With `st.navigation`, `st.Page(icon=...)` is shown beside the page's menu label and its title in the browser tab. `page_title` sets the tab title. `st.set_page_config` calls are additive, so the entrypoint can set values for every page.
- **Theme in `.streamlit/config.toml`**, under `[theme]`. The app has no config file yet. Since 1.46 Streamlit also looks for configuration relative to the entrypoint file. Options present in 1.67.0:
  - Colours: `base` (`"light"` or `"dark"`), `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, `linkColor`, `linkUnderline`, `borderColor`, `codeTextColor`, `codeBackgroundColor`, and a palette (`redColor`, `orangeColor`, `yellowColor`, `blueColor`, `greenColor`, `violetColor`, `grayColor`, each with a `...BackgroundColor` and a `...TextColor` variant).
  - Type: `font`, `headingFont`, `codeFont`, `fontFaces` (self-hosted fonts), `baseFontSize`, `baseFontWeight`, `headingFontSizes`, `headingFontWeights`, `codeFontSize`, `codeFontWeight`, `metricValueFontSize`, `metricValueFontWeight`.
  - Shape and spacing: `baseRadius`, `buttonRadius`, `showWidgetBorder`, `showSidebarBorder`, `paddingTop`, `paddingBottom` (1.66; these trim the blank space above and below the page, which bears on the "smaller banner" candidate).
  - Tables and charts: `dataframeBorderColor`, `dataframeHeaderBackgroundColor`, `dataframeHeaderTextColor`, `chartCategoricalColors`, `chartSequentialColors`, `chartDivergingColors`.
  - Sections: `[theme.sidebar]` styles the sidebar separately, and `[theme.light]` / `[theme.dark]` define both modes (1.51).
  - 1.67.0 shipped "a refreshed default theme", so the default look is new in this version.
  - Not available: styling beyond these options. Custom CSS is not a supported theming route.

## 6. Paging, lazy loading and table size

Three native routes to a ranking longer than 500 rows:

1. **Send more rows eagerly.** Nothing in Streamlit caps the table at 500; `ROWS_SHOWN` is the app's own constant. The grid is "designed to theoretically handle tables with millions of rows". All rows travel to the browser in one message, limited by `server.maxMessageSize` (default 200 MB). Up to 150,000 rows every feature works. Above 150,000 rows sent eagerly, column sorting and CSV download are disabled.
2. **Lazy loading: `st.dataframe(..., lazy=True | False | None)`** (1.61). The server keeps the frame and sends only the rows in view, fetching more on scroll.
   - `None` (default): automatic for in-memory pandas, Polars and PyArrow tables over 150,000 rows, and for unevaluated objects such as a Polars `LazyFrame` over 10,000 rows.
   - `True`: always, except that 1,000 rows or fewer are still sent eagerly.
   - Sorting is done on the server and still works.
   - **Not supported when lazy: search, CSV download, selections (`on_select`) and `pandas.Styler`.** Multi-level column headers raise an error.
   - Whether `ButtonColumn` and `LinkColumn` work in lazy mode is **Unconfirmed**.
   - This is lazy between the Streamlit server and the browser only. The app would still fetch the whole ranking from the API into the Streamlit process first.
3. **Server-side paging: `st.pagination(num_pages, *, default=1, max_visible_pages=7, width="content", key=None, on_change="rerun", disabled=False, bind=None, persist_state=None)`** (1.58). It returns the selected page, 1-indexed. It is only the page control: the app passes the page to the API as an offset and shows that slice in an ordinary `st.dataframe`. `bind="query-params"` puts the page in the URL. Limits: a header-click sort orders only the rows on the current page, so the ranking order must come from the API's `rank_by`; the page should be set back to 1 when a filter changes (write `st.session_state[key] = 1` in a filter's `on_change` callback). This needs an offset on the API read.

How large stays responsive: the documented thresholds are the 150,000-row eager limit and the 200 MB message limit. The docs give no timing figures, and warn that a slow connection and server memory are the practical limits. For this app each rerun also pays for the API's JSON response and for building a pandas frame, and `st.cache_data` keeps a copy per filter combination. Actual timings are **Unconfirmed**; measure in a prototype if the row cap is raised.

## What this means for the eight candidate improvements

| Candidate (from the map) | Native? | With |
|---|---|---|
| A narrower table | Yes | `column_order` to hide by default, the built-in column chooser, pinned Product and Market, pixel widths, short labels with `help` |
| Row detail in place | Yes | `on_select="rerun"`, `selection_mode="single-row"`, then a panel or an `st.dialog` drawer |
| Filters in the sidebar with a reset | Yes | `st.sidebar`, a reset button with an `on_click` callback writing to `st.session_state` |
| Sector names | Yes | A text column or `format_func`; no Streamlit limit |
| A smaller banner | Yes | `theme.paddingTop`, `st.logo`, `help=` tooltips; the rest is ordinary layout |
| First-visit guidance | Partly | `st.dialog` or an expander shown once per session via `st.session_state`. Nothing native remembers a visitor across sessions except the URL. |
| Regrouped navigation | Yes | `st.navigation` with a dict; Why? as `visibility="hidden"` |
| One home for the Excluded Products list | Yes | Its own `st.Page`, or a dialog |

## Sources

- Installed package, Streamlit 1.67.0, docstrings and signatures (the source of the API reference pages): `streamlit/elements/arrow.py` (`st.dataframe`, selection state), `streamlit/elements/lib/column_types.py` (`st.column_config`), `streamlit/commands/navigation.py`, `streamlit/navigation/page.py`, `streamlit/source_util.py` (URL inferred from a filename), `streamlit/commands/logo.py`, `streamlit/commands/page_config.py`, `streamlit/commands/execution_control.py` (`st.switch_page`), `streamlit/elements/dialog_decorator.py`, `streamlit/elements/widgets/*.py` (`bind`, `persist_state`, `st.pagination`), `streamlit/config.py` (theme and server options), `streamlit/errors.py`, and the menu labels in the bundled front end (`static/static/js/DataFrame.*.js`: Autosize, Pin column, Hide column, Show/hide columns, Download as CSV, Search).
- Release notes: [2026](https://docs.streamlit.io/develop/quick-reference/release-notes/2026) (1.53 to 1.67), [2025](https://docs.streamlit.io/develop/quick-reference/release-notes/2025) (1.42 to 1.52), [2024](https://docs.streamlit.io/develop/quick-reference/release-notes/2024) (1.30 to 1.41).
- Concept pages: [Dataframes](https://docs.streamlit.io/develop/concepts/design/dataframes) (large data, lazy loading, the 150,000-row threshold, the header menu), [Widget behavior](https://docs.streamlit.io/develop/concepts/architecture/widget-behavior) (state across pages, `persist_state`, query-parameter binding, callback order).
- API reference, generated from the docstrings above and not fetched page by page: [st.dataframe](https://docs.streamlit.io/develop/api-reference/data/st.dataframe), [st.column_config](https://docs.streamlit.io/develop/api-reference/data/st.column_config), [st.navigation](https://docs.streamlit.io/develop/api-reference/navigation/st.navigation), [st.Page](https://docs.streamlit.io/develop/api-reference/navigation/st.page), [st.logo](https://docs.streamlit.io/develop/api-reference/media/st.logo), [st.dialog](https://docs.streamlit.io/develop/api-reference/execution-flow/st.dialog), [st.query_params](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.query_params), [config.toml](https://docs.streamlit.io/develop/api-reference/configuration/config.toml).

Caveats on the "since" versions: the release notes were read through a summarising fetch, not line by line, so a version may be off by one release; none of them is later than 1.67, because every API was confirmed in the installed package. The version `st.logo` first arrived in was not confirmed (it predates 1.39, where `size` was added). The 1.56 notes announce programmatic selection as "the selection parameter"; in 1.67.0 the parameter is `selection_default`.
