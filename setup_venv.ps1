# PowerShell setup script
# Usage: Right-click 'Run with PowerShell' or run:  .\setup_venv.ps1
$ErrorActionPreference = "Stop"

# Create venv
python -m venv .venv
# Activate for current session
. .\.venv\Scripts\Activate.ps1

# Upgrade pip & install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Register Jupyter kernel
$proj = Split-Path -Leaf (Get-Location)
python -m ipykernel install --user --name "$proj-env" --display-name "Python ($proj)"

Write-Host "✅ Done. Activate later with: .\.venv\Scripts\Activate.ps1"
