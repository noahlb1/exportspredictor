"""The Leaderboard, the app's default page: every UK Product–Market pair, ranked from the precomputed Baseline
table, so Advisors see the top Opportunities without searching. Its filters sit in the sidebar and in the URL
(`app.filters`); a Product search or a Market filter ranks one Product's Markets or one Market's Products.

One cached read of up to `READ_LIMIT` rows per filter set: the page shows the top `ROWS_SHOWN` of them (all of them
with Show all), and the CSV holds them all, in full dollars.
"""

import streamlit as st

from app import client
from app.filters import leaderboard_filters, show_all
from app.format import format_usd
from app.frame import page
from app.tables import (
    LEADERBOARD_SHOWN,
    WHY,
    leaderboard_columns,
    leaderboard_csv,
    leaderboard_frame,
    leaderboard_unit,
)
from app.ui import (
    READ_LIMIT,
    ROWS_SHOWN,
    caveats,
    excluded_products_list,
    fetch,
    rank_orders,
)

with page(
    "Leaderboard",
    "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's fundamentals "
    "suggest.",
):
    try:
        filters = leaderboard_filters()
        body = fetch("/opportunities/top", client.api_url(), **filters.params, limit=READ_LIMIT)
    except client.NotFound as error:
        # An Excluded Product chosen in the search (or named in a link): its reason, and no ranking.
        st.warning(str(error))
        st.markdown(
            "Excluded Products are never ranked. The **Excluded Products** section of **Model Evaluation** lists "
            "each one with its reason."
        )
        st.page_link("pages/model_evaluation.py", label="Open Model Evaluation")
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        excluded_products_list()
        summary, read = body["summary"], body["opportunities"]
        pairs = summary["pairs"]
        line = st.empty()  # the active-filters line, written once Show all is known
        rows = read if show_all(pairs) else read[:ROWS_SHOWN]
        in_force = "; ".join(filters.named) if filters.named else "No filters"
        if pairs > READ_LIMIT:
            shown = f"Top {len(rows):,} of {pairs:,} pairs: narrow them down with the filters to show them all."
        else:
            shown = f"{len(rows):,} pairs shown of {pairs:,}."
        line.markdown(
            f"**{in_force}.** {shown} "
            f"Total positive Export Gap, {body['gap_period_label']}: {format_usd(summary['positive_export_gap_usd'])}."
        )
        order = rank_orders()[body["rank_by"]]
        st.caption(f"Ranked by {order['title']}, Baseline year {body['baseline_year']}. {order['description']}")
        if not rows:
            st.info("No Opportunities match these filters.")
        else:
            st.caption(leaderboard_unit(body))
            st.caption("Open **Why?** on a row to see which factors raised or lowered its Export Potential.")
            st.dataframe(
                leaderboard_frame(rows),
                hide_index=True,
                column_order=[*LEADERBOARD_SHOWN, WHY],
                column_config=leaderboard_columns(body),
            )
            flagged = [caveat for field, (_, caveat) in caveats().items() if any(row[field] for row in read)]
            if flagged:
                st.caption(" ".join(flagged) + " Use the hide filters to leave them out.")
            if pairs > len(read):
                st.caption(f"The CSV holds the top {len(read):,} of {pairs:,} pairs.")
            st.download_button(
                "Download CSV",
                leaderboard_csv(body, read).to_csv(index=False),
                file_name="leaderboard.csv",
                mime="text/csv",
                key="csv",
            )
