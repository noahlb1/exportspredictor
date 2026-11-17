"""PROTOTYPE, throwaway (wayfinder ticket "Top Opportunities landing view", #31).

Question: how should the app's landing page show the best Opportunities without the Advisor
having to search? Four structurally different variants on the Home route, switched with
`?variant=A|B|C|D` or the floating bar (or the ← → keys).

Prototype shortcuts, deliberately not production:
- reads the newest artefact's Parquet directly instead of going through the API (a winner
  needs a new API endpoint, e.g. `/opportunities/top`);
- the Export Gap is the artefact's 2024 Baseline gap, not yet the Gap Period mean;
- values are nominal USD (the Currency decision), no £.
"""

import altair as alt
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

from exportspredictor.paths import artefacts_dir
from exportspredictor.scenarios import is_blocked

VARIANTS = {
    "A": "Leaderboard",
    "B": "Potential vs realised quadrant",
    "C": "Market briefs: Opportunities | Blocked",
    "D": "Sector × region heatmap",
}

# Preview of the "Product exclusions and re-export hubs" ticket: Products the first run showed
# dominating the Export Gap. Not a decision; a toggle so the landing view can be judged both ways.
DISTORTING_PREFIXES = ("7106", "7108", "7109", "7110", "7111", "7112", "2709", "2711")

STATUS = {
    "expand": "Expand (UK already sells here)",
    "new_market": "New Market (UK sells this Product elsewhere)",
    "new_product": "New Product (UK sells it nowhere)",
}


@st.cache_resource(show_spinner="Loading the artefact…")
def pairs() -> pl.DataFrame:
    folder = sorted(p for p in artefacts_dir().iterdir() if p.is_dir())[-1]
    read = lambda name: pl.read_parquet(folder / f"{name}.parquet")  # noqa: E731
    baseline = read("baseline")
    ta = read("baseline_features").select("hs6", "iso3", trade_agreement=pl.col("trade_agreement") == 1)
    sold_anywhere = baseline.group_by("hs6").agg(uk_total=pl.col("actual_exports_usd").sum())
    return (
        baseline.join(ta, on=["hs6", "iso3"], how="left")
        .join(read("products"), on="hs6", how="left")
        .join(read("markets").rename({"name": "market"}), on="iso3", how="left")
        .join(sold_anywhere, on="hs6", how="left")
        .with_columns(
            relative_gap=pl.col("export_gap_usd") / pl.col("export_potential_usd"),
            realised=(pl.col("actual_exports_usd") / pl.col("export_potential_usd")).clip(0, 2),
            blocked_opportunity=is_blocked("blocked_effect_usd", "export_potential_usd"),
            status=pl.when(pl.col("actual_exports_usd") > 0)
            .then(pl.lit(STATUS["expand"]))
            .when(pl.col("uk_total") > 0)
            .then(pl.lit(STATUS["new_market"]))
            .otherwise(pl.lit(STATUS["new_product"])),
            distorting=pl.col("hs6").str.slice(0, 4).is_in(DISTORTING_PREFIXES),
            label=pl.col("hs6") + " " + pl.col("description").str.slice(0, 40) + " → " + pl.col("market"),
        )
    )


def filters(data: pl.DataFrame) -> pl.DataFrame:
    """The shared sidebar filters (every variant gets the same ones, so layouts can be compared)."""
    sb = st.sidebar
    sb.header("Filters")
    sectors = sb.multiselect("Sector (HS2)", sorted(data["sector"].unique().to_list()))
    regions = sb.multiselect("Market region", sorted(data["region"].unique().to_list()))
    markets = sb.multiselect("Market", sorted(data["market"].unique().to_list()))
    statuses = sb.multiselect("UK already exports?", list(STATUS.values()), help="Per pair: here / elsewhere / nowhere")
    min_gap = sb.number_input("Minimum Export Gap (USD)", value=1_000_000, step=1_000_000)
    ta = sb.radio("UK Trade Agreement in force", ["Any", "Yes", "No"], horizontal=True)
    hide = sb.checkbox("Hide precious metals, crude oil, natural gas (preview)", value=False)
    out = data.filter(pl.col("export_gap_usd") >= min_gap)
    if sectors:
        out = out.filter(pl.col("sector").is_in(sectors))
    if regions:
        out = out.filter(pl.col("region").is_in(regions))
    if markets:
        out = out.filter(pl.col("market").is_in(markets))
    if statuses:
        out = out.filter(pl.col("status").is_in(statuses))
    if ta != "Any":
        out = out.filter(pl.col("trade_agreement") == (ta == "Yes"))
    if hide:
        out = out.filter(~pl.col("distorting"))
    sb.caption(f"{out.height:,} of {data.height:,} Product–Market pairs pass.")
    return out


MONEY = st.column_config.NumberColumn(format="dollar")
TABLE_COLUMNS = {
    "hs6": "HS6",
    "description": "Product",
    "market": "Market",
    "export_potential_usd": "Export Potential",
    "actual_exports_usd": "Actual Exports",
    "export_gap_usd": "Export Gap",
    "relative_gap": "Relative gap",
    "blocked_effect_usd": "Blocked Opportunity value",
    "blocked_opportunity": "Blocked?",
    "status": "UK already exports?",
}


def table(rows: pl.DataFrame, height: int = 500) -> None:
    st.dataframe(
        rows.select(list(TABLE_COLUMNS)).rename(TABLE_COLUMNS).to_pandas(),
        hide_index=True,
        height=height,
        column_config={
            **{h: MONEY for h in ["Export Potential", "Actual Exports", "Export Gap", "Blocked Opportunity value"]},
            "Relative gap": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )


# ---------------------------------------------------------------- A: one global leaderboard
def variant_a(data: pl.DataFrame) -> None:
    st.subheader("Top Opportunities")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pairs", f"{data.height:,}")
    c2.metric("Total positive Export Gap", f"${data['export_gap_usd'].sum() / 1e9:,.1f}bn")
    c3.metric("Blocked Opportunities", f"{data['blocked_opportunity'].sum():,}")
    order = st.radio(
        "Rank by",
        ["Absolute Export Gap", "Relative gap (gap ÷ Export Potential)", "Blocked Opportunity value"],
        horizontal=True,
    )
    if order.startswith("Relative"):
        floor = st.select_slider(
            "…among pairs with Export Potential of at least (USD)",
            [0, 1e5, 1e6, 1e7, 1e8],
            value=1e6,
            format_func=lambda v: f"${v:,.0f}",
        )
        st.caption(
            "Relative gap = 1 − Actual ÷ Potential, so every pair with no UK exports ties at 100%. "
            "Ties are broken by absolute Export Gap."
        )
        data = data.filter(pl.col("export_potential_usd") >= floor).sort(
            ["relative_gap", "export_gap_usd"], descending=True
        )
    elif order.startswith("Blocked"):
        data = data.sort("blocked_effect_usd", descending=True)
    else:
        data = data.sort("export_gap_usd", descending=True)
    table(data.head(200), height=650)


# ---------------------------------------------------------------- B: 2-D quadrant instead of one sort key
def variant_b(data: pl.DataFrame) -> None:
    st.subheader("Where is the UK furthest below its potential?")
    st.caption(
        "Top 2,000 pairs by Export Gap. Right = large Export Potential; bottom = little of it realised. "
        "The bottom-right corner holds the large, unrealised Opportunities. Hollow points are Blocked "
        "Opportunities. Drag a box to list pairs below."
    )
    top = data.sort("export_gap_usd", descending=True).head(2000).to_pandas()
    brush = alt.selection_interval(name="box")
    chart = (
        alt.Chart(top)
        .mark_point(filled=True, opacity=0.7)
        .encode(
            x=alt.X("export_potential_usd:Q", scale=alt.Scale(type="log"), title="Export Potential (USD, log)"),
            y=alt.Y("realised:Q", title="Share of Export Potential realised (Actual ÷ Potential)"),
            size=alt.Size("export_gap_usd:Q", title="Export Gap", legend=None),
            color=alt.Color("status:N", title="UK already exports?", legend=alt.Legend(orient="bottom")),
            shape=alt.Shape("blocked_opportunity:N", title="Blocked?", scale=alt.Scale(range=["circle", "diamond"])),
            tooltip=["label", alt.Tooltip("export_gap_usd:Q", format="$,.0f"), "status", "blocked_opportunity"],
        )
        .add_params(brush)
        .properties(height=520)
    )
    rule = alt.Chart().mark_rule(strokeDash=[4, 4], color="gray").encode(y=alt.datum(1))
    event = st.altair_chart(chart + rule, width="stretch", on_select="rerun", selection_mode="box")
    box = event.get("selection", {}).get("box") if event else None
    if box:
        x0, x1 = box["export_potential_usd"]
        y0, y1 = box["realised"]
        picked = data.filter(
            pl.col("export_potential_usd").is_between(x0, x1) & pl.col("realised").is_between(y0, y1)
        ).sort("export_gap_usd", descending=True)
        st.write(f"**{picked.height:,} pairs in the box**")
        table(picked.head(200))
    else:
        st.write("**Largest Export Gaps** (drag a box on the chart to narrow)")
        table(data.sort("export_gap_usd", descending=True).head(25), height=400)


# ---------------------------------------------------------------- C: aggregated by Market, two columns
def variant_c(data: pl.DataFrame) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Opportunities")
        st.caption("Markets ranked by summed Export Gap; their top Products inside.")
        by_market = data.group_by("market").agg(
            gap=pl.col("export_gap_usd").sum(), n=pl.len(), ta=pl.col("trade_agreement").first()
        ).sort("gap", descending=True)
        for row in by_market.head(12).iter_rows(named=True):
            badge = " · Trade Agreement" if row["ta"] else ""
            with st.expander(f"**{row['market']}**: ${row['gap'] / 1e9:,.2f}bn across {row['n']:,} Products{badge}"):
                top = data.filter(pl.col("market") == row["market"]).sort("export_gap_usd", descending=True).head(8)
                for p in top.iter_rows(named=True):
                    st.markdown(
                        f"- `{p['hs6']}` {p['description'][:60]}: **${p['export_gap_usd'] / 1e6:,.1f}M** "
                        f"gap · {p['status'].split(' (')[0]}"
                    )
    with right:
        st.subheader("Blocked Opportunities")
        st.caption("Markets with no UK Trade Agreement, ranked by summed Blocked Opportunity value.")
        blocked = data.filter(pl.col("blocked_opportunity"))
        by_market = blocked.group_by("market").agg(v=pl.col("blocked_effect_usd").sum(), n=pl.len()).sort(
            "v", descending=True
        )
        if by_market.is_empty():
            st.info("No Blocked Opportunities pass these filters.")
        for row in by_market.head(12).iter_rows(named=True):
            with st.expander(f"**{row['market']}**: ${row['v'] / 1e9:,.2f}bn across {row['n']:,} Products"):
                top = blocked.filter(pl.col("market") == row["market"]).sort("blocked_effect_usd", descending=True)
                for p in top.head(8).iter_rows(named=True):
                    st.markdown(
                        f"- `{p['hs6']}` {p['description'][:60]}: **+${p['blocked_effect_usd'] / 1e6:,.1f}M** "
                        f"with a Trade Agreement"
                    )


# ---------------------------------------------------------------- D: Sector × region heatmap, drill down
def variant_d(data: pl.DataFrame) -> None:
    metric = st.radio("Colour by", ["Export Gap", "Blocked Opportunity value"], horizontal=True)
    column = "export_gap_usd" if metric == "Export Gap" else "blocked_effect_usd"
    cells = data.group_by("sector", "region").agg(v=pl.col(column).sum(), n=pl.len())
    top_sectors = cells.group_by("sector").agg(pl.col("v").sum()).sort("v", descending=True).head(30)["sector"]
    cells = cells.filter(pl.col("sector").is_in(top_sectors)).with_columns(bn=pl.col("v") / 1e9).to_pandas()
    st.caption(f"The 30 Sectors with the largest summed {metric}. Click a cell to list its pairs.")
    pick = alt.selection_point(name="cell", fields=["sector", "region"])
    chart = (
        alt.Chart(cells)
        .mark_rect()
        .encode(
            x=alt.X("region:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("sector:N", sort=list(top_sectors), title="Sector (HS2)"),
            color=alt.Color("bn:Q", title=f"{metric} ($bn)", scale=alt.Scale(scheme="blues")),
            tooltip=["sector", "region", alt.Tooltip("bn:Q", format=",.2f", title="$bn"), "n"],
        )
        .add_params(pick)
        .properties(height=700)
    )
    event = st.altair_chart(chart, width="stretch", on_select="rerun")
    chosen = (event.get("selection", {}) or {}).get("cell") if event else None
    if chosen:
        s, r = chosen[0]["sector"], chosen[0]["region"]
        st.write(f"**Sector {s} in {r}**")
        sub = data.filter((pl.col("sector") == s) & (pl.col("region") == r))
        table(sub.sort(column, descending=True).head(200))


# ---------------------------------------------------------------- switcher
def switcher(current: str) -> None:
    keys = list(VARIANTS)
    i = keys.index(current)
    prev, nxt = keys[i - 1], keys[(i + 1) % len(keys)]
    st.html(
        f"""
        <div style="position:fixed;bottom:18px;left:50%;transform:translateX(-50%);z-index:999999;
             background:#111;color:#fff;border-radius:999px;padding:8px 18px;font:14px sans-serif;
             box-shadow:0 4px 16px rgba(0,0,0,.35);display:flex;gap:16px;align-items:center">
          <a href="?variant={prev}" target="_self" style="color:#fff;text-decoration:none">◀</a>
          <span>PROTOTYPE · {current} ({VARIANTS[current]})</span>
          <a href="?variant={nxt}" target="_self" style="color:#fff;text-decoration:none">▶</a>
        </div>"""
    )
    components.html(
        f"""<script>
        const d = window.parent.document;
        if (!d.__protoKeys) {{
          d.__protoKeys = true;
          d.addEventListener('keydown', e => {{
            const t = e.target;
            if (t.closest && t.closest('input,textarea,[contenteditable]')) return;
            const v = new URL(window.parent.location).searchParams.get('variant') || 'A';
            const keys = {keys!r}; const i = keys.indexOf(v);
            let n = null;
            if (e.key === 'ArrowRight') n = keys[(i + 1) % keys.length];
            if (e.key === 'ArrowLeft') n = keys[(i - 1 + keys.length) % keys.length];
            if (n) window.parent.location.search = '?variant=' + n;
          }});
        }}
        </script>""",
        height=0,
    )


def render() -> None:
    variant = st.query_params.get("variant", "A")
    variant = variant if variant in VARIANTS else "A"
    data = filters(pairs())
    {"A": variant_a, "B": variant_b, "C": variant_c, "D": variant_d}[variant](data)
    switcher(variant)
