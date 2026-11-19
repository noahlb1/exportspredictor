<#
Adds Python dependencies for one checkout: records them in its pyproject.toml and uv.lock
(`uv add --no-sync`) and installs them into the main checkout's shared .venv, which every
worktree runs against (scripts/wt.ps1). .claude/settings.json allows this script, so an
unattended run can add a dependency without a permission prompt. Run it from the main checkout:

    powershell -File scripts/add-dep.ps1 C:\CodingProjectsC\ep-wt\42 matplotlib
#>
param(
    [Parameter(Mandatory)][string]$Checkout,
    [Parameter(Mandatory, ValueFromRemainingArguments)][string[]]$Packages
)
$main = (git -C $Checkout worktree list --porcelain)[0] -replace '^worktree ', ''
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv*\uv.exe" |
        Select-Object -First 1).FullName
}

Push-Location $Checkout
try {
    & $uv add --no-sync @Packages
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & $uv pip install --python (Join-Path $main '.venv') @Packages
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
