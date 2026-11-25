"""The repo a visitor reads holds together: every relative link in its docs and notebook resolves to a tracked file."""

import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parent.parent


def tracked_paths() -> set[str]:
    """Every path git tracks, as a repo-relative POSIX path."""
    names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return {name for name in names.decode().split("\0") if name}


# Inline links and images, `[text](target "title")`, and reference definitions, `[label]: target`.
_INLINE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\)")
_REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
_FENCE = re.compile(r"^\s*(```|~~~).*?^\s*\1", re.MULTILINE | re.DOTALL)
_CODE_SPAN = re.compile(r"(`+).*?\1", re.DOTALL)
_EXTERNAL = re.compile(r"^(https?:|mailto:|#)", re.IGNORECASE)


def _links(markdown: str) -> list[str]:
    """The relative link targets in a piece of markdown, leaving out code, web links and pure anchors."""
    prose = _CODE_SPAN.sub("", _FENCE.sub("", markdown))
    targets = _INLINE.findall(prose) + _REFERENCE.findall(prose)
    return [target for target in targets if not _EXTERNAL.match(target)]


def _documents(tracked: set[str]) -> dict[str, str]:
    """The markdown a visitor reads, by path: the README, the glossary, the coding standards, everything
    under docs/, and the markdown cells of the notebooks."""
    documents = {}
    for name in sorted(tracked):
        if name in {"README.md", "GLOSSARY.md", "CODING_STANDARDS.md"} or (
            name.startswith("docs/") and name.endswith(".md")
        ):
            documents[name] = (ROOT / name).read_text(encoding="utf-8")
        elif name.startswith("notebooks/") and name.endswith(".ipynb"):
            cells = json.loads((ROOT / name).read_text(encoding="utf-8"))["cells"]
            documents[name] = "\n\n".join(
                "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
            )
    return documents


def _resolves(source: str, target: str, tracked: set[str]) -> bool:
    """Whether a link from `source` reaches a tracked file, or a folder holding one."""
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    parts: list[str] = []
    for part in (PurePosixPath(source).parent / path).parts:
        if part == "..":
            if not parts:
                return False  # outside the repo
            parts.pop()
        elif part not in {".", ""}:
            parts.append(part)
    resolved = "/".join(parts)
    return resolved in tracked or any(name.startswith(resolved + "/") for name in tracked)


# The agent files: instructions, settings, skills, agent docs and run notes, kept in a private repo and linked in.
AGENT_PATHS = ("CLAUDE.md", ".claude/", ".agents/", "docs/agents/", "docs/runs/", "skills-lock.json", "scripts/add-dep.ps1")


def _is_agent_path(name: str) -> bool:
    return PurePosixPath(name).name in {"CLAUDE.md", "AGENTS.md"} or name.startswith(AGENT_PATHS)


def test_no_agent_or_claude_path_is_tracked():
    tracked = sorted(name for name in tracked_paths() if _is_agent_path(name))
    assert not tracked, f"Agent files belong in the private agent repo, not here: {tracked}"


def test_the_agent_paths_are_ignored_so_they_cannot_be_committed_back():
    probes = [path + "x" if path.endswith("/") else path for path in AGENT_PATHS]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=ROOT,
        input="\0".join(probes).encode(),
        capture_output=True,
    ).stdout.decode().split("\0")[:-1]
    assert sorted(ignored) == sorted(probes), f"Add these to .gitignore: {sorted(set(probes) - set(ignored))}"


def test_every_relative_link_in_the_docs_and_notebook_points_at_a_tracked_file():
    tracked = tracked_paths()
    broken = [
        (source, target)
        for source, text in _documents(tracked).items()
        for target in _links(text)
        if not _resolves(source, target, tracked)
    ]
    assert not broken, f"These links point at paths git does not track: {broken}"


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("See [ADR 1](adr/0001-x.md#why) and ![chart](img%20one.png).", ["adr/0001-x.md#why", "img%20one.png"]),
        ('A [titled](../GLOSSARY.md "Glossary") link.', ["../GLOSSARY.md"]),
        ("[ref]: notes/a.md\n", ["notes/a.md"]),
        ("[web](https://example.com) [mail](mailto:a@b.c) [here](#top)", []),
        ("`[code](not/a/link.md)`\n```\n[fenced](nor/this.md)\n```\n", []),
    ],
)
def test_the_link_finder_keeps_relative_targets_and_skips_web_links_anchors_and_code(markdown, expected):
    assert _links(markdown) == expected


def test_a_link_resolves_relative_to_its_file_without_its_fragment_and_decoded():
    tracked = {"docs/adr/0001-x.md", "docs/research/a b.md", "README.md"}
    assert _resolves("docs/adr/0002-y.md", "0001-x.md#why", tracked)
    assert _resolves("docs/adr/0002-y.md", "../research/a%20b.md", tracked)
    assert _resolves("docs/adr/0002-y.md", "../../README.md", tracked)
    assert _resolves("README.md", "docs/adr/", tracked)
    assert not _resolves("README.md", "0001-x.md", tracked)
    assert not _resolves("README.md", "../README.md", tracked)
