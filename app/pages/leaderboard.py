"""The Leaderboard, the app's default page: every UK Product–Market pair, ranked from the precomputed Baseline
table, so Advisors see the top Opportunities without searching. Its filters sit in the sidebar and in the URL
(`app.filters`); a Product search or a Market filter ranks one Product's Markets or one Market's Products."""

import streamlit as st

from app import client
from app.filters import leaderboard_filters
from app.format import format_usd
from app.frame import page
from app.ui import ROWS_SHOWN, fetch, show_leaderboard

with page(
    "Leaderboard",
    "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's fundamentals "
    "suggest.",
):
    try:
        filters = leaderboard_filters()
        body = fetch("/opportunities/top", client.api_url(), **filters.params, limit=ROWS_SHOWN)
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
        summary, shown = body["summary"], len(body["opportunities"])
        in_force = "; ".join(filters.named) if filters.named else "No filters"
        st.markdown(
            f"**{in_force}.** {shown:,} pairs shown of {summary['pairs']:,}. "
            f"Total positive Export Gap, {body['gap_period_label']}: {format_usd(summary['positive_export_gap_usd'])}."
        )
        show_leaderboard(body, file_name="leaderboard")
