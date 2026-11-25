"""The Model Evaluation page's ITC section, rendered from the API's `itc_comparison` (#175)."""

from dataclasses import replace

from conftest import run_page

from exportspredictor import artefacts as store
from exportspredictor import realisation

PAGE = "pages/model_evaluation.py"
SECTION = "How the model compares with ITC's Export Potential Map"


def _page(monkeypatch, live_api, root):
    with live_api(root) as url:
        monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", url)
        page = run_page(PAGE)
    assert not page.exception
    return page


def test_the_section_follows_lift_folded_with_the_api_s_text_its_table_and_caveat(
    monkeypatch, live_api, fixture_artefacts, fixture_artefacts_dir
):
    page = _page(monkeypatch, live_api, fixture_artefacts_dir)
    served = realisation.served(fixture_artefacts.evaluation[realisation.REPORT_KEY])

    labels = [e.label for e in page.expander]
    assert labels.index(SECTION) == labels.index("Do large Export Gaps point at growth? (Lift)") + 1
    [section] = [e for e in page.expander if e.label == SECTION]
    assert not section.proto.expanded
    assert served["text"] in [m.value for m in section.markdown]
    assert served["caveat"] in " ".join(c.value for c in section.caption)
    [table] = section.dataframe
    frame = table.value
    # The HS2 rows only, the level compared with ITC, with the model and the Naive Baseline beside ITC's figures.
    assert list(frame.columns) == ["Test", "Level", "Term", "ITC (76 sub-sectors)", "Model", "Naive Baseline"]
    assert set(frame["Level"]) == {"HS2"}
    assert list(frame["ITC (76 sub-sectors)"]) == ["0.615", "-0.386", "0.023", "0.243"]
    captions = " ".join(c.value for c in section.caption)
    assert "Naive Baseline" in captions and "HS2" in captions
    assert "dropped" in " ".join(c.value for c in section.caption)


def test_nothing_renders_when_the_field_is_absent(monkeypatch, live_api, fixture_artefacts, tmp_path):
    evaluation = {k: v for k, v in fixture_artefacts.evaluation.items() if k != realisation.REPORT_KEY}
    store.save(replace(fixture_artefacts, evaluation=evaluation), tmp_path)

    page = _page(monkeypatch, live_api, tmp_path)

    assert SECTION not in [e.label for e in page.expander]
