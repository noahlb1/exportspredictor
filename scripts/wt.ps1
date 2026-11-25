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
# UTF-8 for every Python process, subprocesses included (`jupyter execute`, the notebook kernel):
# a file opened without an encoding is otherwise read as cp1252 and saved back garbled.
$env:PYTHONUTF8 = '1'
# Data and artefacts are gitignored, so a worktree has none: share the main checkout's.
if (-not $env:EXPORTSPREDICTOR_DATA_DIR) { $env:EXPORTSPREDICTOR_DATA_DIR = Join-Path $main 'data' }
if (-not $env:EXPORTSPREDICTOR_ARTEFACTS_DIR) { $env:EXPORTSPREDICTOR_ARTEFACTS_DIR = Join-Path $main 'artefacts' }
# The agent files (instructions and skills) are not in git. The owner keeps them in a private repo beside the main
# checkout, whose link.ps1 links them into this checkout; it is quick, and a no-op once linked. Absent for anyone else.
$agentRepo = if ($env:EXPORTSPREDICTOR_AGENT_DIR) { $env:EXPORTSPREDICTOR_AGENT_DIR } else { "$main-agent" }
$linkAgentFiles = Join-Path $agentRepo 'link.ps1'
if (Test-Path $linkAgentFiles) {
    try { & $linkAgentFiles $checkout -Quiet } catch { Write-Warning "Linking the agent files failed: $_" }
}

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv*\uv.exe" |
        Select-Object -First 1).FullName
}

# Every shell call starts in the main checkout, so a relative path in a worktree's command can write
# there unseen. Warn when this run leaves the main checkout's working tree changed.
$watchMain = (Resolve-Path $checkout).Path -ne (Resolve-Path $main).Path
if ($watchMain) { $mainBefore = @(git -C $main status --porcelain) }

Push-Location $checkout
try {
    & $uv run @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
    if ($watchMain) {
        $changed = @(git -C $main status --porcelain | Where-Object { $mainBefore -notcontains $_ })
        if ($changed) {
            Write-Warning "The main checkout ($main) changed during this run; a command wrote there instead of this worktree:`n$($changed -join "`n")"
        }
    }
}
