# Creates an isolated .venv inside benchmark-v1 and installs pinned deps.
# Isolated on purpose: the corpus-generation job runs against the system Python and
# must not be perturbed by anything this repo installs.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
Write-Output "bootstrap: root=$root"
if (-not (Test-Path $venv)) {
    python -m venv $venv
    Write-Output "bootstrap: venv created"
} else {
    Write-Output "bootstrap: venv already present"
}
$py = Join-Path $venv 'Scripts/python.exe'
if (-not (Test-Path $py)) { $py = Join-Path $venv 'bin/python' }
& $py -m pip install --disable-pip-version-check --no-input --upgrade pip
& $py -m pip install --disable-pip-version-check --no-input -r (Join-Path $root 'requirements.txt')
& $py -c "import yaml, pytest; print('bootstrap: pyyaml', yaml.__version__, '| pytest', pytest.__version__)"
Write-Output "bootstrap: OK"
