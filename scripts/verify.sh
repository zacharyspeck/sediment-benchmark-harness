#!/usr/bin/env bash
# THE ONE COMMAND (macOS / Linux).
#
#   ./scripts/verify.sh
#
# Runs the whole thing end to end and tells you whether the harness is sound:
#   1. runs the full test suite (every grader against correct AND incorrect fixtures)
#   2. resolves every question template against the fixture ledger -- generating nothing
#   3. runs the fixtures end to end through both fixture arms and writes a report
#
# Exits non-zero if anything fails. Makes no model calls and touches no network.
#
# The PowerShell equivalent is scripts/verify.ps1. The two are kept in step.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
failed=()

step() {
    local name="$1"; shift
    printf '\n==============================================================\n'
    printf '  %s\n' "$name"
    printf '==============================================================\n'
    if "$@"; then
        printf '  --> ok (%s)\n' "$name"
    else
        failed+=("$name")
        printf '  --> FAILED (%s)\n' "$name"
    fi
}

# ---- python: repo-local venv if present, else whatever is on PATH ----
# Each candidate is PROBED rather than trusted. On Windows `python3` and
# `python` may resolve to the Microsoft Store shim, which exists on PATH and
# then refuses to run anything -- picking it by name alone fails every step
# with a confusing message instead of falling through to a real interpreter.
py=""
for cand in "$root/.venv/bin/python" \
            "$root/.venv/Scripts/python.exe" \
            python3 python py; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
          >/dev/null 2>&1; then
        py="$cand"
        break
    fi
done
if [ -z "$py" ]; then
    echo "FATAL: no working Python 3.11+ found." >&2
    echo "  tried: .venv/bin/python, .venv/Scripts/python.exe, python3, python, py" >&2
    exit 2
fi
echo "repo:   $root"
echo "python: $py"
"$py" -c "import sys; print('version:', sys.version.split()[0])"

step "test suite"                                 "$py" -m pytest tests -q --no-header
step "template resolution (generates nothing)"    "$py" -m harness.cli validate-templates
step "item-file validation (shape + power)"       "$py" -m harness.cli validate-items \
                                                      --items fixtures/fixture_items.yaml --strict
step "fixture run, end to end"                    "$py" -m harness.cli fixtures

printf '\n==============================================================\n'
if [ ${#failed[@]} -eq 0 ]; then
    echo "  ALL CHECKS PASSED"
    echo "=============================================================="
    exit 0
fi
printf '  FAILED: %s\n' "$(IFS=', '; echo "${failed[*]}")"
echo "=============================================================="
exit 1
