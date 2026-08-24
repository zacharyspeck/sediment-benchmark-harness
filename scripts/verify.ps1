# THE ONE COMMAND.
#
#   .\scripts\verify.ps1
#
# Runs the whole thing end to end and tells you whether the harness is sound:
#   1. confirms we are in benchmark-v1 and not the corpus repo
#   2. runs the full test suite (every grader against correct AND incorrect fixtures)
#   3. resolves all 30 question templates against the ledger -- generating nothing
#   4. runs the 30 fixtures x 2 arms end to end and writes a report
#
# Exits non-zero if anything fails. Makes no model calls and touches no network.
[CmdletBinding()]
param(
    [switch]$SkipTemplates,
    [string]$ReportOut
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$failed = @()

function Step($name, [scriptblock]$body) {
    Write-Output ""
    Write-Output "=============================================================="
    Write-Output "  $name"
    Write-Output "=============================================================="
    & $body
    if ($LASTEXITCODE -ne 0) {
        $script:failed += $name
        Write-Output "  --> FAILED ($name), exit $LASTEXITCODE"
    } else {
        Write-Output "  --> ok ($name)"
    }
}

# ---- 0. repo identity ----
$toplevel = (git -C $root rev-parse --show-toplevel 2>$null)
if (-not $toplevel) { Write-Output "FATAL: $root is not a git repository"; exit 2 }
$normTop = ($toplevel -replace '/', '\').TrimEnd('\')
Write-Output "repo: $normTop"
Write-Output "head: $(git -C $root log --oneline -1)"

# ---- python ----
$py = Join-Path $root '.venv/Scripts/python.exe'        # Windows layout
if (-not (Test-Path $py)) { $py = Join-Path $root '.venv/bin/python' }  # POSIX layout
if (-not (Test-Path $py)) {
    Write-Output "no .venv found; falling back to system python (PyYAML optional -- the"
    Write-Output "harness ships a fallback parser). Run scripts/bootstrap_venv.ps1"
    Write-Output "(or: python -m venv .venv) to pin deps."
    $py = if (Get-Command python3 -ErrorAction SilentlyContinue) { 'python3' } else { 'python' }
}
Write-Output "python: $py"
& $py -c "import sys; print('version:', sys.version.split()[0])"

Step "test suite" { & $py -m pytest (Join-Path $root 'tests') -q --no-header }

if (-not $SkipTemplates) {
    Step "template resolution (generates nothing)" {
        & $py -m harness.cli validate-templates
    }
}

Step "item-file validation (shape + statistical power)" {
    & $py -m harness.cli validate-items --items (Join-Path $root 'fixtures' 'fixture_items.yaml') --strict
}

$reportArgs = @('-m', 'harness.cli', 'fixtures')
if ($ReportOut) { $reportArgs += @('--report-out', $ReportOut) }
Step "fixture run, end to end" { & $py @reportArgs }

Write-Output ""
Write-Output "=============================================================="
if ($failed.Count -eq 0) {
    Write-Output "  ALL CHECKS PASSED"
    Write-Output "=============================================================="
    exit 0
}
Write-Output "  FAILED: $($failed -join ', ')"
Write-Output "=============================================================="
exit 1
