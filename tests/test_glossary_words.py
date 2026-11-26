"""Words GLOSSARY.md avoids stay out of the prose an Advisor or a marker reads.

Reviews removed "shortfall" twice before it came back, so the check is a test, not a
review rule. It reads the `_Avoid_:` lists from GLOSSARY.md and scans the notebook's markdown
cells, every string literal in `app/` and `api/`, and the package's docstrings.

It checks every avoided phrase of two or more words, and the single words in `DISTINCTIVE`. Most
single avoided words ("trade", "score", "user") have a legitimate everyday sense, so checking them
would flag correct prose; add a single word here once a review finds it misused.

It also checks that every key term the app shows has an explanation in `wording.term_definitions`.
"""

import ast
import json
import re
from pathlib import Path

from exportspredictor.wording import term_definitions

ROOT = Path(__file__).resolve().parent.parent
DISTINCTIVE = {"shortfall", "headroom", "entrepôt", "supplier", "shipment", "commodity", "prospect", "blocker"}
# Uses that are meant, by file, as exact text around the word.
ALLOWED = {
    # A single consignment's noise, not a Flow.
    "notebooks/exports_predictor.ipynb": {"noise of single shipments"},
    "src/exportspredictor/gap_period.py": {"one unusual shipment"},
    # HMRC's own name for its CN8 codes.
    "src/exportspredictor/hmrc.py": {'"commodity codes"', '"commodity code"'},
    # The bootstrap interval of a backtest's estimate, and the statement of what a Potential Range is not.
    "src/exportspredictor/backtests.py": {"95% confidence interval"},
    "src/exportspredictor/conformal.py": {"not a confidence interval", "never a confidence interval"},
}


def avoided_words() -> list[str]:
    glossary = (ROOT / "GLOSSARY.md").read_text(encoding="utf-8")
    words = []
    for line in re.findall(r"^_Avoid_:\s*(.*)$", glossary, re.MULTILINE):
        words += [w.strip() for w in re.sub(r"\([^)]*\)", "", line).split(",") if w.strip()]
    return [w for w in words if " " in w or w in DISTINCTIVE]


def prose() -> list[tuple[str, str]]:
    texts = []
    notebook = json.loads((ROOT / "notebooks/exports_predictor.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            texts.append(("notebooks/exports_predictor.ipynb", "".join(cell["source"])))
    for folder in ("app", "api"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    texts.append((path.relative_to(ROOT).as_posix(), node.value))
    for path in sorted((ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    texts.append((path.relative_to(ROOT).as_posix(), docstring))
    return texts


def test_prose_uses_glossary_terms_not_the_words_it_avoids():
    patterns = {w: re.compile(rf"\b{re.escape(w)}s?\b", re.IGNORECASE) for w in avoided_words()}
    found = []
    for name, text in prose():
        text = " ".join(text.split())  # a docstring wraps lines mid-phrase
        for word, pattern in patterns.items():
            for match in pattern.finditer(text):
                around = text[max(match.start() - 30, 0) : match.end() + 30]
                if not any(allowed in around for allowed in ALLOWED.get(name, ())):
                    found.append((name, word, around))
    assert not found, f"Words GLOSSARY.md avoids; use its term instead: {found}"


# Key terms the app shows, each with the `wording.term_definitions` entries that explain it on hover or behind `?`.
# A term may have one entry per context (a table column, the detail panel), until #195 settles one wording.
EXPLAINED_BY = {
    "Product": ["product"],
    "Market": ["market"],
    "Sector": ["sector"],
    "HS6": ["hs6"],
    "Export Potential": ["export_potential", "baseline_export_potential"],
    "Actual Exports": ["actual_exports", "baseline_actual_exports"],
    "Export Gap": ["export_gap"],
    "Relative Export Gap": ["relative_export_gap", "relative_export_gap_in_panel"],
    "Opportunity Type": ["opportunity_type"],
    "Potential Range": ["potential_range"],
    "Blocked Opportunity": ["blocked_opportunity"],
    "Blocked Opportunity value": ["blocked_opportunity_value", "blocked_opportunity_value_in_panel"],
    "Barrier": ["barrier"],
    "Rise with agreement": ["rise_with_agreement"],
    "Rise share": ["rise_share"],
    "Tariff part": ["tariff_part"],
    "Trade Agreement": ["trade_agreement"],
    "Applied Tariff": ["applied_tariff", "applied_tariff_in_panel"],
    "Scenario Effect": ["scenario_effect"],
    "Market Reach": ["market_reach"],
}
# Key terms the app shows with no explanation yet (listed on #181); #195 adds them.
KNOWN_GAPS = {"Opportunity", "Scenario", "Baseline", "Gap Period", "Illustrative", "Agreement part"}
TRADE_TERMS = {"HS6"}  # key terms outside GLOSSARY.md


def app_text() -> str:
    """Every string literal in `app/` (f-strings' fixed parts included) and every term explanation the app shows."""
    texts = list(term_definitions("2022–2024 average", 2024).values())
    for path in sorted((ROOT / "app").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                texts.append(node.value)
    return "\n".join(texts)


def test_every_key_term_the_app_shows_has_an_explanation():
    definitions = term_definitions("2022–2024 average", 2024)
    text = app_text()
    listed = {*EXPLAINED_BY, *KNOWN_GAPS}

    assert {term for term in listed if re.search(rf"\b{re.escape(term)}s?\b", text)} == listed, (
        "a key term the app no longer shows: drop it here"
    )
    assert not KNOWN_GAPS & set(EXPLAINED_BY), "an explained term is still listed as a gap"
    missing = {key: term for term, keys in EXPLAINED_BY.items() for key in keys if not definitions.get(key)}
    assert not missing, f"Key terms with no explanation in wording.term_definitions: {missing}"


def test_the_key_terms_are_glossary_terms():
    glossary = set(re.findall(r"^\*\*(.+?)\*\*:", (ROOT / "GLOSSARY.md").read_text(encoding="utf-8"), re.MULTILINE))
    assert {*EXPLAINED_BY, *KNOWN_GAPS} <= glossary | TRADE_TERMS


def test_the_glossary_lists_are_read():
    words = avoided_words()
    assert "shortfall" in words and "market share" in words
    assert "trade" not in words  # an everyday word, not checked
