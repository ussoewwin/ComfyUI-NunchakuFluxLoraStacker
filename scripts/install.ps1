param(
    [switch]$SkipEngine,
    [switch]$Repair,
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

$NodeRoot = Split-Path -Parent $PSScriptRoot
$Outputs = Join-Path $NodeRoot 'outputs'
$Log = Join-Path $Outputs 'install.log'
$Marker = Join-Path $NodeRoot '.ccsr-trt-installed'
Push-Location -LiteralPath $NodeRoot

New-Item -ItemType Directory -Force $Outputs | Out-Null
try { Start-Transcript -Path $Log -Append | Out-Null } catch {}

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host ('== ' + $Message + ' ==') -ForegroundColor Cyan
}

function Find-ComfyPython {
    if ($PythonPath -and (Test-Path -LiteralPath $PythonPath)) {
        return (Resolve-Path $PythonPath).Path
    }

    # 1. ComfyUI python_embeded
    $comfyEmbeded = @(
        'D:\USERFILES\ComfyUI\python_embeded\python.exe',
        'C:\ComfyUI\python_embeded\python.exe',
        (Join-Path $NodeRoot '..\..\..\python_embeded\python.exe'),
        (Join-Path $NodeRoot '..\..\python_embeded\python.exe')
    )
    foreach ($p in $comfyEmbeded) {
        if (Test-Path -LiteralPath $p) { return (Resolve-Path $p).Path }
    }

    # 2. Active Virtual Environment
    if ($env:VIRTUAL_ENV) {
        $activePy = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
        if (Test-Path -LiteralPath $activePy) { return $activePy }
    }

    # 3. PATH Python
    $pythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCmd) { return $pythonCmd.Source }

    return $null
}

try {
    Write-Host 'ComfyUI CCSR TensorRT Installer (ComfyUI-NunchakuFluxLoraStacker)' -ForegroundColor Green
    Write-Host 'Installs TensorRT RTX, Triton, ONNX stack, and CCSR engine artifacts into ComfyUI environment.'

    $TargetPython = Find-ComfyPython
    if (-not $TargetPython) {
        throw 'ComfyUI Python environment was not found. Please specify -PythonPath "path\to\python.exe".'
    }
    Write-Host "Target ComfyUI Python: $TargetPython" -ForegroundColor Cyan

    function Invoke-TargetPip([string[]]$Arguments) {
        & $TargetPython -m pip @Arguments
        if ($LASTEXITCODE -ne 0) { throw "pip failed: $($Arguments -join ' ')" }
    }

    Write-Step 'Verifying ComfyUI PyTorch and CUDA'
    $torchInfo = & $TargetPython -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $torchInfo) {
        throw "Could not execute PyTorch in $TargetPython. Ensure ComfyUI environment is working."
    }
    Write-Host $torchInfo -ForegroundColor Green

    Write-Step 'Running install.py (requirements & TensorRT runtime)'
    & $TargetPython (Join-Path $NodeRoot 'install.py')
    if ($LASTEXITCODE -ne 0) { throw 'install.py failed.' }

    Write-Step 'Final readiness check'
    & $TargetPython (Join-Path $PSScriptRoot 'verify_install.py')
    if ($LASTEXITCODE -ne 0) { throw 'The final installation check failed.' }
    Set-Content -LiteralPath $Marker -Value (Get-Date -Format o) -Encoding ascii

    Write-Host ''
    Write-Host 'ComfyUI CCSR TensorRT setup is complete.' -ForegroundColor Green
}
catch {
    Write-Host ''
    Write-Host ('INSTALLATION FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Detailed log: ' + $Log) -ForegroundColor Yellow
    exit 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    try { Pop-Location } catch {}
}
