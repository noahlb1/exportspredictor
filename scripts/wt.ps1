<#
Runs `uv run <args>` for this checkout against the main checkout's shared .venv.

Worktrees use this instead of plain `uv run`, so they never build a .venv of their own.
PYTHONPATH puts this checkout's src/ and root first, ahead of the shared .venv's
editable install, so imports and subprocesses (notebook kernel, uvicorn) use this
checkout's code. It also points the data and artefact dirs at the main checkout's.

    & C:\CodingProjectsC\ep-wt\42\scripts\wt.ps1 pytest -q
    powershell -File scripts/wt.ps1 python -m exportspredictor.pipeline   # from Git Bash
#>
$checkout = Split-Path $PSScriptRoot -Parent
$main = (git -C $checkout worktree list --porcelain)[0] -replace '^worktree ', ''
$env:UV_PROJECT_ENVIRONMENT = Join-Path $main '.venv'
$env:UV_NO_SYNC = '1'
$env:PYTHONPATH = "$checkout\src;$checkout"
# Data and artefacts are gitignored, so a worktree has none: share the main checkout's.
if (-not $env:EXPORTSPREDICTOR_DATA_DIR) { $env:EXPORTSPREDICTOR_DATA_DIR = Join-Path $main 'data' }
if (-not $env:EXPORTSPREDICTOR_ARTEFACTS_DIR) { $env:EXPORTSPREDICTOR_ARTEFACTS_DIR = Join-Path $main 'artefacts' }

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv*\uv.exe" |
        Select-Object -First 1).FullName
}

Push-Location $checkout
try {
    & $uv run @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
