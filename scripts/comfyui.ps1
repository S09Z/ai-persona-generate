# scripts/comfyui.ps1 - launch ComfyUI on Windows with the project's .venv and model folders.
# Usage: powershell -File scripts/comfyui.ps1 [-Port 8188] [extra ComfyUI args]
param(
    [int]$Port = 8188,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Extra
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$comfy = Join-Path $root "vendor\ComfyUI"
if (-not (Test-Path $py)) { throw "project venv not found: run 'uv sync' first" }
if (-not (Test-Path $comfy)) { throw "ComfyUI not found at $comfy" }

$env:PYTHONUTF8 = "1"
Set-Location $comfy
& $py main.py --listen 127.0.0.1 --port $Port `
    --extra-model-paths-config (Join-Path $root "configs\comfyui_model_paths.yaml") @Extra
