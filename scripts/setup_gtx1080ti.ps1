$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) is required. Install Python 3.10 x64 first."
}
py -3.10 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 x64 was not found. This candidate environment is pinned to Python 3.10."
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    py -3.10 -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv" }
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

# Candidate Windows wheel for Pascal validation; the preflight checks the actual compiled arch list.
& $Python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed" }

& $Python -m pip install -r (Join-Path $Root "requirements-gtx1080ti.txt")
if ($LASTEXITCODE -ne 0) { throw "Unlimited-OCR Python dependencies failed to install" }

& $Python -m compat.gtx1080ti
if ($LASTEXITCODE -ne 0) {
    throw "GPU preflight failed. Review the report above before attempting model inference."
}

Write-Host ""
Write-Host "Environment ready. First test with one page:"
Write-Host ".\.venv\Scripts\python.exe -m compat.local_runner --input .\path\to\page.png --max-pages 1"
