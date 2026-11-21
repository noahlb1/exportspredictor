# Agent environment

Gotchas of this machine that the code and config don't reveal. Add one when a run loses time to something the next run would hit again.

## Tooling

- **uv** comes from winget. Shells started before the install don't have it on PATH. `scripts/wt.ps1` finds it either way; elsewhere, use the full path `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe`.
- **Python** comes only through uv. A bare `python` in Git Bash hits the Windows Store stub and hangs until timeout: always go through `uv run`.
- **Encoding**: Windows defaults to cp1252, which crashes on printing `£` or accented Market names and garbles UTF-8 files a tool opens without an encoding. `scripts/wt.ps1` sets `PYTHONUTF8=1`, which subprocesses inherit; `-X utf8` alone does not reach a subprocess such as `jupyter execute`. `tests/test_encoding.py` catches garbled text in any tracked file.
- **`gh` bodies**: PowerShell 5.1 splits a multi-line `--body` into separate arguments. Write the body to a file in the scratchpad and pass `--body-file`.
- **podman** runs in a WSL2 VM, with `docker-compose` as its compose provider. If `podman compose` can't connect after a reboot, run `podman machine start`. `podman compose up -d --build` serves the API on :8000 and the app on :8501; `podman compose down` stops them.

## Worktrees

- **Create them by hand**: `git worktree add C:\CodingProjectsC\ep-wt\<ticket> -b <branch> <base>`. Agent worktree isolation refuses this repo because the session path is `c:\` and git reports `C:\`.
- **Run everything through `scripts/wt.ps1`** (`& <worktree>\scripts\wt.ps1 pytest -q`). It uses the main checkout's `.venv`, so a worktree builds none, and removing it is quick.
- **Every shell call starts in the main checkout**, whatever the last call `cd`'d into. Write with absolute worktree paths and `git -C <worktree>`; a relative path edits the main checkout, which the API and app run from (spec #113 left main's package unable to import this way). `scripts/wt.ps1` warns when the main checkout's working tree changed during its run. A merger checks `git -C C:\CodingProjectsC\exportspredictor status --porcelain` against the run's starting snapshot after each merge.
- **Merging into the integration branch**: after each merge, run `scripts/wt.ps1 ruff check .` and `scripts/wt.ps1 pytest -q` in the merged checkout, and get them green before the next merge. A file that exists on disk but is gitignored passes in a worktree and fails everywhere else, so check `git status --ignored` for test data.
- **Pushes are gated** by `.githooks/pre-push` (ruff, pyright, then the full suite, about 4 minutes). It needs `git config core.hooksPath .githooks` once per clone.
- **Skills live in `.agents/skills/`.** In the main checkout `.claude/skills/<name>` is a junction to it, but git tracks both paths and a worktree gets two plain copies. Edit a skill under `.agents/skills/` and copy the change to `.claude/skills/` in the same commit.
- **Adding a dependency**: from the main checkout, `powershell -File scripts/add-dep.ps1 <worktree> <pkg>` records it in that worktree's `pyproject.toml` and `uv.lock` and installs it into the shared `.venv`. `.claude/settings.json` allows exactly this form, so it runs without a permission prompt; a bare `uv add` or `uv pip install` can be refused.

## Shared data and artefacts

- **One data directory and one artefacts directory** for every checkout, the main checkout's. `scripts/wt.ps1` sets `EXPORTSPREDICTOR_DATA_DIR` and `EXPORTSPREDICTOR_ARTEFACTS_DIR` to them; without it, the real-data tests skip in a worktree. Raw BACI alone is 1.3 GB; check a file exists before downloading, and write through `exportspredictor.downloads` so nobody reads a half-written file.
- **Artefacts**: the API serves the lexically newest version folder.
- **Build on the saved fold models.** A full real rebuild (`python -m exportspredictor.pipeline`) takes about an hour on the GPU. A ticket that adds a table derives it from the newest artefact's fold models and writes a new version; `python -m exportspredictor.pipeline refresh` copies the newest version with fresh metadata in seconds, and `python -m exportspredictor.pipeline rerange` copies it with its Potential Range recomputed from the Baseline year's feature file (a few minutes on the CPU, no retraining). `python -m exportspredictor.pipeline recaveat` copies it with its Baseline marked again with the declared Caveat Tiers (seconds; run it after adding a tier or changing a tier's table). An artefact missing a table that `artefacts.load` expects fails to load, so regenerate or refresh after adding one.
- **Experiments** run through `exportspredictor.loop`. Its package docstring says how to add an arm, where its model cache and results live (`data/derived/loop_V202601/`, about 4 GB) and what to watch for; read it before a modelling experiment.

## Long GPU runs

- **One GPU, one run.** Before training, check `nvidia-smi`; wait if another run holds the GPU.
- **Background it.** A foreground command times out at 10 minutes. Start training with Bash `run_in_background`, logging to a file, then wait with a second background command that loops until the PID exits, under a deadline: `timeout 4h bash -c 'until ! tasklist //FI "PID eq <pid>" | grep -q <pid>; do sleep 30; done'`. You are re-invoked when it ends. Give every wait loop and `tail -f` watcher a `timeout`: a loop waiting for a log line that never comes otherwise polls forever.
- **Stop what you start.** Before reporting, stop every background command you started (wait loops, `tail -f`, Monitor watchers) and check none is left: `powershell -Command "Get-CimInstance Win32_Process | ? { $_.Name -match '^(bash|tail)\.exe$' } | select ProcessId,CreationDate,CommandLine"`. A finished subagent's watcher outlives it and keeps running.
- **Report after the results exist.** A subagent that must stop earlier reports the PID, the log path and the remaining steps, so the orchestrator can resume it.
