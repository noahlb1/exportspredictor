# Agent environment

Gotchas of this machine that the code and config don't reveal. Add one when a run loses time to something the next run would hit again.

## Tooling

- **uv** comes from winget. Shells started before the install don't have it on PATH. `scripts/wt.ps1` finds it either way; elsewhere, use the full path `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe`.
- **Python** comes only through uv. A bare `python` in Git Bash hits the Windows Store stub and hangs until timeout: always go through `uv run`.
- **Printing non-ASCII** (accented Market names, `£`) crashes on cp1252: run Python with `-X utf8`. `python -I` ignores `PYTHONIOENCODING`.
- **`gh` bodies**: PowerShell 5.1 splits a multi-line `--body` into separate arguments. Write the body to a file in the scratchpad and pass `--body-file`.
- **podman** is installed but needs WSL, which needs admin rights and a reboot. Until `podman machine init` works, write compose changes and leave `podman compose up` to the user.

## Worktrees

- **Create them by hand**: `git worktree add C:\CodingProjectsC\ep-wt\<ticket> -b <branch> <base>`. Agent worktree isolation refuses this repo because the session path is `c:\` and git reports `C:\`.
- **Run everything through `scripts/wt.ps1`** (`& <worktree>\scripts\wt.ps1 pytest -q`). It uses the main checkout's `.venv`, so a worktree builds none, and removing it is quick.
- **Adding a dependency**: `uv add --no-sync <pkg>` in the worktree updates `pyproject.toml` and `uv.lock`; then install it into the shared env with `uv pip install --python C:\CodingProjectsC\exportspredictor\.venv <pkg>`.

## Shared data and artefacts

- **One data directory** for every checkout: set `EXPORTSPREDICTOR_DATA_DIR=C:\CodingProjectsC\exportspredictor\data` in a worktree. Raw BACI alone is 1.3 GB; check a file exists before downloading, and write through `exportspredictor.downloads` so nobody reads a half-written file.
- **Artefacts**: set `EXPORTSPREDICTOR_ARTEFACTS_DIR=C:\CodingProjectsC\exportspredictor\artefacts`. The API serves the lexically newest version folder.
- **Build on the saved fold models.** A full real rebuild (`python -m exportspredictor.pipeline`) takes about an hour on the GPU. A ticket that adds a table derives it from the newest artefact's fold models and writes a new version; `python -m exportspredictor.pipeline refresh` copies the newest version with fresh metadata in seconds. An artefact missing a table that `artefacts.load` expects fails to load, so regenerate or refresh after adding one.

## Long GPU runs

- **One GPU, one run.** Before training, check `nvidia-smi`; wait if another run holds the GPU.
- **Background it.** A foreground command times out at 10 minutes. Start training with Bash `run_in_background`, logging to a file, then wait with a second background command that loops until the PID exits (`until ! tasklist //FI "PID eq <pid>" | grep -q <pid>; do sleep 30; done`). You are re-invoked when it ends.
- **Report after the results exist.** A subagent that must stop earlier reports the PID, the log path and the remaining steps, so the orchestrator can resume it.
