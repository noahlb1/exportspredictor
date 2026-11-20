"""The Model Evaluation page, run with AppTest against a live local API on fixture artefacts."""

from dataclasses import replace

import pytest
from conftest import run_page
from streamlit.testing.v1 import AppTest

from exportspredictor import artefacts as store
from exportspredictor.evaluation import LIMITATIONS

PAGE = "pages/model_evaluation.py"


def _run(monkeypatch, url: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", url)
    page = run_page(PAGE)
    assert not page.exception
    return page


def _text(page: AppTest) -> str:
    elements = [*page.markdown, *page.caption, *page.success, *page.warning, *page.subheader, *page.info]
    return " ".join(element.value for element in elements)


@pytest.fixture
def page(monkeypatch, live_api, fixture_artefacts_dir):
    with live_api(fixture_artefacts_dir) as url:
        yield _run(monkeypatch, url)


def test_page_opens_with_four_figures_from_the_evaluation(page, fixture_artefacts):
    report = fixture_artefacts.evaluation
    headline, coverage, stability = report["headline"], report["potential_range"], report["stability"]
    figures = {metric.label: metric for metric in page.metric}

    assert list(figures) == [
        "Rank correlation, UK pairs", "Inside the Potential Range", "Size-matched Lift", "Ranking stability",
    ]
    rank = figures["Rank correlation, UK pairs"]
    assert rank.value == f"{headline['model']:.2f}"
    assert f"{headline['naive']:.2f}" in rank.delta and "Naive Baseline" in rank.delta
    inside = figures["Inside the Potential Range"]
    assert inside.value == f"{coverage['coverage']:.0%}" and f"{coverage['target']:.0%}" in inside.delta
    assert figures["Size-matched Lift"].value == f"{report['lift']['size_matched']['lift']:.2f}"
    steady = figures["Ranking stability"]
    assert steady.value == f"{stability['spearman_potential']:.2f}"
    assert f"{stability['top_overlap']:.0%}" in steady.delta


def test_every_section_is_folded_but_accuracy_and_the_known_limitations(page):
    sections = {e.label: e.proto.expanded for e in page.expander if not e.label.startswith("All ")}
    labels = list(sections)

    assert labels[0] == "How accurate is the model?" and labels[-2:] == ["Excluded Products", "Known limitations"]
    assert "Stability of the rankings" in labels and "How the model was chosen" in labels
    assert {label for label, open_ in sections.items() if open_} == {"How accurate is the model?", "Known limitations"}


def test_page_shows_a_plain_english_headline_against_the_naive_baseline(page, fixture_artefacts):
    headline = fixture_artefacts.evaluation["headline"]
    shown = page.success if headline["beats_naive"] else page.warning

    assert any(headline["text"] in element.value for element in shown)


def test_page_shows_metrics_and_the_weakest_sectors_and_regions(page, fixture_artefacts):
    tables = [element.value for element in page.dataframe]
    sector_table = next(t for t in tables if "Sector" in t.columns)
    region_table = next(t for t in tables if "Region" in t.columns)
    weakest = fixture_artefacts.evaluation["accuracy"]["uk_out_of_fold"]["by_sector"][0]["group"]

    assert any("Naive baseline" in t.columns for t in tables)
    assert sector_table["Sector"].iloc[0] == weakest
    assert set(region_table["Region"]) == {"East Asia & Pacific", "North America", "South Asia"}
    assert "Weakest Sectors" in _text(page) and "Weakest regions" in _text(page)


def test_page_lists_every_known_limitation(page):
    text = _text(page)

    for limitation in LIMITATIONS:
        assert limitation["title"] in text
    # The fixture's bands are log offsets (ADR 0007, amended): no fixed dollar width to report
    assert "The Potential Range is a fixed width in dollars" not in text


def test_page_shows_the_imputed_observed_tariff_split(page, fixture_artefacts):
    report = fixture_artefacts.evaluation
    tables = [element.value for element in page.dataframe]
    split = next(t for t in tables if "Applied Tariff" in t.columns)

    assert list(split["Applied Tariff"]) == ["Observed", "Carried forward from 2021"]
    imputed = report["accuracy"]["uk_out_of_fold"]["by_tariff"][1]["rows"]
    assert list(split["Rows"]) == [0, imputed]
    assert report["tariff_split_text"] in _text(page)


def test_page_shows_the_potential_range_coverage(page, fixture_artefacts):
    coverage = fixture_artefacts.evaluation["potential_range"]
    tables = [element.value for element in page.dataframe]

    assert "Potential Range" in _text(page)
    assert f"{coverage['coverage']:.0%}" in _text(page) and f"{coverage['target']:.0%}" in _text(page)
    assert any("Coverage" in t.columns and "Exporter" in t.columns for t in tables)


def test_page_shows_the_potential_range_coverage_by_size(page, fixture_artefacts):
    by_size = fixture_artefacts.evaluation["potential_range"]["by_size"]
    tables = [element.value for element in page.dataframe]

    table = next(t for t in tables if "Size of Export Potential" in t.columns)
    assert list(table["Size of Export Potential"]) == [row["size"] for row in by_size]
    assert list(table["Rows"]) == [row["rows"] for row in by_size]


def test_page_drops_no_tariffs_and_lists_v2s_limitations(page):
    text = _text(page)

    assert "No tariffs" not in text
    for title in ["Tariffs carried forward from 2021", "Excluded Products", "Trained on the top 100 Exporters",
                  "Values are nominal US dollars", "What the Potential Range assumes"]:
        assert title in text, title


def test_page_says_how_the_event_backtest_sets_the_tariff(page, fixture_artefacts):
    assert fixture_artefacts.evaluation["event_backtest"]["events"][0]["states"] in _text(page)


def test_page_states_the_test_year(page, fixture_artefacts):
    assert str(fixture_artefacts.evaluation["test_year"]) in _text(page)


def test_page_states_the_non_uk_training_sample(monkeypatch, live_api, fixture_artefacts, tmp_path):
    sampled = {**fixture_artefacts.evaluation, "non_uk_sampling_rate": 0.3}
    store.save(replace(fixture_artefacts, evaluation=sampled), tmp_path)

    with live_api(tmp_path) as url:
        page = _run(monkeypatch, url)

    assert "30%" in _text(page)


def _largest_markets_table(page: AppTest):
    return next(element.value for element in page.dataframe if "Markets measured" in element.value.columns)


def test_page_shows_the_measure_on_the_largest_importing_markets_beside_the_headline(page, fixture_artefacts):
    report = fixture_artefacts.evaluation
    headline, (top30, top60) = report["headline"], report["large_markets"]["rows"]
    table = _largest_markets_table(page)

    assert list(table["Markets measured"]) == ["Every Market (the headline)", "30 largest importers",
                                                "60 largest importers"]
    assert list(table["Model"]) == [f"{headline['model']:.2f}", f"{top30['model']:.2f}", f"{top60['model']:.2f}"]
    assert list(table["Naive Baseline"]) == [f"{headline['naive']:.2f}", f"{top30['naive']:.2f}",
                                             f"{top60['naive']:.2f}"]
    assert report["large_markets"]["text"] in _text(page)


def test_page_shows_n_a_for_an_artefact_without_the_largest_markets_measure(
    monkeypatch, live_api, fixture_artefacts, tmp_path
):
    older = {key: value for key, value in fixture_artefacts.evaluation.items() if key != "large_markets"}
    store.save(replace(fixture_artefacts, evaluation=older), tmp_path)

    with live_api(tmp_path) as url:
        table = _largest_markets_table(_run(monkeypatch, url))

    assert list(table["Model"])[1:] == ["n/a", "n/a"] and list(table["Naive Baseline"])[1:] == ["n/a", "n/a"]


def test_page_says_so_when_the_artefact_has_no_evaluation(monkeypatch, live_api, fixture_artefacts, tmp_path):
    store.save(replace(fixture_artefacts, evaluation=None), tmp_path)

    with live_api(tmp_path) as url:
        page = _run(monkeypatch, url)

    assert any("no evaluation" in element.value.lower() for element in page.warning)


def test_page_shows_lift_with_its_interval_as_a_tendency(page, fixture_artefacts):
    lift = fixture_artefacts.evaluation["lift"]

    assert lift["text"] in _text(page)
    assert "Lift" in _text(page)


def test_page_shows_size_matched_lift_next_to_raw_lift_and_explains_the_difference(page, fixture_artefacts):
    lift = fixture_artefacts.evaluation["lift"]
    tables = [element.value for element in page.dataframe]
    figures = next(t for t in tables if "Lift" in t.columns and "95% interval" in t.columns)
    text = _text(page)

    assert list(figures["Measure"]) == ["Raw Lift", "Size-matched Lift"]
    assert list(figures["Lift"]) == [pytest.approx(lift["lift"]), pytest.approx(lift["size_matched"]["lift"])]
    assert f"${lift['min_growth_usd']:,.0f}" in text  # the growth threshold, from the API
    assert "tendency" in text
    first, last = lift["gap_years"][0], lift["gap_years"][-1]
    assert f"{first}–{last}" in text  # the Gap Period the Export Gaps are averaged over


def test_page_shows_the_event_backtest_predicted_against_observed_with_its_control_group(page, fixture_artefacts):
    backtest = fixture_artefacts.evaluation["event_backtest"]
    tables = [element.value for element in page.dataframe]
    effects = next(t for t in tables if "Observed (difference-in-differences)" in t.columns)

    assert "Predicted by the model" in effects.columns
    assert list(effects["Year"]) == ["2022", "2023", "2022–2023"]
    assert "India" in _text(page)  # the control Market, by name
    assert backtest["caveat"] in _text(page)


def test_page_states_the_share_of_opportunities_inside_their_potential_range(page, fixture_artefacts):
    # Read from the evaluation report through the API (spec #66): the share replaces a badge on almost every row
    coverage = fixture_artefacts.evaluation["potential_range"]
    share = f"{coverage['uncertain_opportunities']:,} of the {coverage['opportunities']:,} Opportunities"

    assert 0 < coverage["uncertain_opportunities"] <= coverage["opportunities"]
    stated = [element.value for element in page.markdown if share in element.value]
    assert stated, "the share is not on the page"
    assert any(
        f"({coverage['uncertain_opportunity_share']:.0%})" in text and "inside their Potential Range" in text.split(share)[1]
        for text in stated
    )
