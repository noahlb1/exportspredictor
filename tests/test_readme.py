"""The README and the manual UI checklist (docs/ui-checklist.md) keep up with the app and the rebuild script."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
CHECKLIST = REPO / "docs" / "ui-checklist.md"


def _checklist() -> str:
    return CHECKLIST.read_text(encoding="utf-8")


def test_the_readme_links_to_the_checklist():
    assert "(docs/ui-checklist.md)" in README.read_text(encoding="utf-8")


def _page_titles() -> list[str]:
    """The title of every page the router (app/Home.py) declares."""
    router = (REPO / "app" / "Home.py").read_text(encoding="utf-8")
    return re.findall(r'st\.Page\([^)]*title="([^"]+)"', router)


def test_checklist_names_every_app_page():
    checklist = _checklist()
    missing = [title for title in _page_titles() if title not in checklist]
    assert not missing, f"Add these pages to docs/ui-checklist.md: {missing}"


def test_checklist_covers_a_scenario_run_and_a_csv_export():
    checklist = _checklist()
    assert "Scenario" in checklist
    assert "CSV" in checklist


def test_readme_documents_the_one_command_run_and_the_rebuild_script():
    text = README.read_text(encoding="utf-8")
    assert "podman compose up" in text
    assert "scripts/rebuild.py" in text
    assert "uv sync" in text


def test_checklist_checks_values_are_in_usd_with_no_pounds():
    checklist = _checklist()
    assert "USD" in checklist
    assert "£" not in checklist


def _rebuild_section() -> str:
    text = README.read_text(encoding="utf-8")
    return text.split("## Rebuild the data and artefacts", 1)[1].split("\n## ", 1)[0]


def test_rebuild_table_names_every_step_of_the_rebuild_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rebuild", REPO / "scripts" / "rebuild.py")
    rebuild = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rebuild)
    section = _rebuild_section()

    missing = [step.name for step in rebuild.STEPS if step.name not in section]
    assert not missing, f"Add these rebuild steps to the README's rebuild table: {missing}"


def test_rebuild_section_gives_the_pipeline_command():
    assert "python -m exportspredictor.pipeline" in _rebuild_section()


def _sections() -> list[str]:
    """The README's `## ` sections in order, each starting at its heading text."""
    return README.read_text(encoding="utf-8").split("\n## ")[1:]


def _section(heading: str) -> str:
    return next(section for section in _sections() if section.startswith(heading + "\n"))


def test_the_run_section_leads_the_readme_and_names_the_fetch_command():
    first = _sections()[0]
    assert first.startswith("Run the app\n")
    assert "podman compose run --rm fetch" in first
    assert "docker compose run --rm fetch" in first
    assert first.index("compose run --rm fetch") < first.index("podman compose up")
    assert "localhost:8501" in first


def test_the_manual_route_names_the_released_asset_and_its_checksum_file():
    from exportspredictor import release

    asset = release.asset_name(release.RELEASED_VERSION)
    manual = next(line for line in _sections()[0].splitlines() if "By hand" in line)
    assert asset in manual and asset + release.CHECKSUM_SUFFIX in manual
    assert f"releases/tag/{release.TAG}" in _sections()[0]


def test_the_rebuild_section_says_it_needs_an_nvidia_gpu_and_is_not_needed_to_run_the_app():
    start = _rebuild_section().split("```", 1)[0]
    assert "NVIDIA GPU" in start
    assert "not needed to run the app" in start


def test_the_rebuild_section_says_how_the_containers_pick_up_a_new_version_or_new_code():
    section = _rebuild_section()
    assert "read-only" in section and "podman compose restart api" in section
    assert "pipeline rereach" in section and "format-3" in section
    assert "podman compose up --build" in section


def test_the_served_model_section_describes_the_blend_chosen_on_the_selection_measure():
    section = _section("The served model")
    assert "Blend" in section and "Selection Measure" in section
    assert "This is **v2**" not in section


def test_the_rebuild_table_says_the_notebook_reads_saved_results_and_how_long_it_takes():
    row = next(line for line in _rebuild_section().splitlines() if "notebooks/exports_predictor.ipynb" in line)
    assert "saved results" in row and "minutes" in row
    assert "fits" not in row and "rebuilds" not in row  # it builds no table and fits no benchmark


def test_the_rebuild_section_names_the_command_that_refreshes_the_notebook_s_outputs():
    assert "uv run python scripts/rebuild.py --only notebook" in _rebuild_section()


def test_checklist_states_the_app_is_designed_for_1280_px_and_not_for_phones():
    checklist = _checklist()
    assert "1280 px" in checklist
    assert "phones" in checklist.lower()


def test_checklist_mentions_no_removed_page():
    checklist = _checklist()
    removed = ["Why?", "Product to Markets", "Product → Markets", "Market to Products", "Market → Products"]
    named = [title for title in removed if title in checklist]
    assert not named, f"docs/ui-checklist.md still names removed pages: {named}"


def test_checklist_covers_the_market_reach_sentence_in_the_market_view_and_the_panel():
    lines = [line for line in _checklist().splitlines() if "Market Reach" in line]
    assert any("Choose a Market" in line for line in lines)
    assert any("panel" in line for line in lines)


def _repo_map() -> str:
    text = README.read_text(encoding="utf-8")
    return text.split("## Repo map", 1)[1].split("\n## ", 1)[0]


def test_the_repo_map_names_every_top_level_tracked_folder():
    import subprocess

    names = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout.decode()
    folders = {name.split("/", 1)[0] for name in names.split("\0") if "/" in name}
    repo_map = _repo_map()
    missing = sorted(folder for folder in folders if f"`{folder}/`" not in repo_map)
    assert not missing, f"Add these folders to the README's repo map: {missing}"


def test_the_repo_map_says_to_start_reading_at_the_notebook_then_the_package():
    start = next(line for line in _repo_map().splitlines() if "start reading" in line.lower())
    assert start.index("notebooks/exports_predictor.ipynb") < start.index("src/exportspredictor/")


def test_checklist_covers_the_tie_proof_large_market_and_market_reach_figures():
    checklist = _checklist()
    for figure in ("tie-proof", "old measure", "30 largest", "What Market Reach changes", "by size", "UK pairs"):
        assert figure in checklist, figure
