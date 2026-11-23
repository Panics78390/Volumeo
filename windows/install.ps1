<#
  Installe Volumeo sur Windows : environnement Python, dépendances, modèle IA
  (TripoSR, TripoSG, modèle d'image + PyTorch) et raccourci sur le Bureau.

  Lancez plutôt « Installer Volumeo.bat » (double-clic). Options :
    -NoAI   n'installe pas PyTorch ni les modèles d'IA (générateur procédural seulement, ~1 Go de moins)
    -Cpu    force PyTorch en version CPU même si une carte NVIDIA est présente
    -Ci     mode automatique : pas de raccourci, pas de pause
#>
param([switch]$NoAI, [switch]$Cpu, [switch]$Ci)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest est bien plus rapide sans barre
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$TripoSrRef = '107cefdc244c39106fa830359024f6a2f1c78871'
$TripoSgRef = 'fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c'

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "La commande a échoué : $exe $($arguments -join ' ')" }
}

# ---------------------------------------------------------------- Python
function Find-Python {
    $candidates = @(
        @('py', '-3.12'), @('py', '-3.11'), @('py', '-3.13'), @('py', '-3.10'), @('python'), @('python3')
    )
    foreach ($c in $candidates) {
        $exe = $c[0]
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $rest = @($c | Select-Object -Skip 1)
        try {
            $v = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(10|11|12|13)$') {
            $found = @($exe) + $rest
            return ,$found
        }
    }
    return $null
}

Step 'Recherche de Python (3.10 à 3.13)'
$py = Find-Python
if (-not $py -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host 'Python introuvable : installation de Python 3.12 avec winget...'
    winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $py = Find-Python
}
if (-not $py) {
    throw 'Python 3.10 à 3.13 est requis. Installez Python 3.12 depuis https://www.python.org/downloads/windows/ (cochez « Add python.exe to PATH »), puis relancez.'
}
Write-Host "Python trouvé : $($py -join ' ')"

# ---------------------------------------------------------------- environnement
Step 'Création de l''environnement Python'
if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
    $pyArgs = @($py | Select-Object -Skip 1) + @('-m', 'venv', $Venv)
    Run $py[0] $pyArgs
}
$VPy = Join-Path $Venv 'Scripts\python.exe'
Run $VPy @('-m', 'pip', 'install', '--upgrade', 'pip', '--quiet')

Step 'Installation des dépendances (quelques minutes)'
Run $VPy @('-m', 'pip', 'install', '-r', (Join-Path $Root 'backend\requirements.txt'), 'pywebview>=5', '--quiet')

# ---------------------------------------------------------------- interface
$Static = Join-Path $Root 'backend\static'
if (-not (Test-Path (Join-Path $Static 'index.html'))) {
    Step 'Compilation de l''interface'
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw 'L''interface compilée est absente. Téléchargez « Volumeo-windows.zip » depuis la page Releases du dépôt GitHub (elle y est incluse), ou installez Node.js puis relancez.'
    }
    Push-Location (Join-Path $Root 'frontend')
    try {
        Run 'npm' @('ci')
        Run 'npm' @('run', 'build')
    } finally { Pop-Location }
    New-Item -ItemType Directory -Force $Static | Out-Null
    Copy-Item (Join-Path $Root 'frontend\dist\*') $Static -Recurse -Force
}

# ---------------------------------------------------------------- IA
if (-not $NoAI) {
    $hasNvidia = $false
    if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        & nvidia-smi *> $null
        $hasNvidia = ($LASTEXITCODE -eq 0)
    }
    # Sous Windows, le PyTorch de PyPI est CPU : la version CUDA vient de l'index PyTorch.
    $indexes = if ($hasNvidia) { @('cu128', 'cu126', 'cu124', 'cpu') } else { @('cpu') }
    if ($hasNvidia) { Step 'Carte NVIDIA détectée : installation de PyTorch avec CUDA (~2,5 Go)' }
    else { Step 'Pas de carte NVIDIA détectée : installation de PyTorch pour CPU' }
    $ok = $false
    foreach ($idx in $indexes) {
        Write-Host "Essai de l'index PyTorch « $idx »..."
        & $VPy -m pip install torch --index-url "https://download.pytorch.org/whl/$idx" --quiet
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
    if (-not $ok) { throw 'Impossible d''installer PyTorch.' }
    Run $VPy @('-m', 'pip', 'install', '-r', (Join-Path $Root 'backend\requirements-ai.txt'), '--quiet')

    $TripoDir = Join-Path $Root 'third_party\TripoSR'
    if (-not (Test-Path (Join-Path $TripoDir 'tsr\system.py'))) {
        Step 'Téléchargement du code TripoSR'
        $tmp = Join-Path $env:TEMP "triposr-$TripoSrRef"
        Invoke-WebRequest "https://github.com/VAST-AI-Research/TripoSR/archive/$TripoSrRef.zip" -OutFile "$tmp.zip"
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive "$tmp.zip" -DestinationPath $tmp
        New-Item -ItemType Directory -Force $TripoDir | Out-Null
        Copy-Item (Join-Path $tmp "TripoSR-$TripoSrRef\tsr") $TripoDir -Recurse -Force
        Copy-Item (Join-Path $tmp "TripoSR-$TripoSrRef\LICENSE") $TripoDir -Force
        Remove-Item $tmp, "$tmp.zip" -Recurse -Force
    }

    $TripoSgDir = Join-Path $Root 'third_party\TripoSG'
    if (-not (Test-Path (Join-Path $TripoSgDir 'triposg\pipelines\pipeline_triposg.py'))) {
        Step 'Téléchargement du code TripoSG'
        $tmp = Join-Path $env:TEMP "triposg-$TripoSgRef"
        Invoke-WebRequest "https://github.com/VAST-AI-Research/TripoSG/archive/$TripoSgRef.zip" -OutFile "$tmp.zip"
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive "$tmp.zip" -DestinationPath $tmp
        New-Item -ItemType Directory -Force $TripoSgDir | Out-Null
        Copy-Item (Join-Path $tmp "TripoSG-$TripoSgRef\triposg") $TripoSgDir -Recurse -Force
        Copy-Item (Join-Path $tmp "TripoSG-$TripoSgRef\LICENSE") $TripoSgDir -Force
        Remove-Item $tmp, "$tmp.zip" -Recurse -Force
    }
}

# ---------------------------------------------------------------- raccourci
if (-not $Ci) {
    Step 'Création du raccourci sur le Bureau'
    $desktop = [Environment]::GetFolderPath('Desktop')
    $lnk = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop 'Volumeo.lnk'))
    $lnk.TargetPath = Join-Path $Venv 'Scripts\pythonw.exe'
    $lnk.Arguments = '-m app.desktop'
    $lnk.WorkingDirectory = Join-Path $Root 'backend'
    $lnk.IconLocation = Join-Path $Root 'windows\volumeo.ico'
    $lnk.Description = 'Volumeo, image vers 3D'
    $lnk.Save()
}

Write-Host "`nVolumeo est installé." -ForegroundColor Green
if (-not $Ci) { Write-Host 'Lancez-le avec le raccourci « Volumeo » du Bureau, ou avec Volumeo.bat.' }
if (-not $NoAI) { Write-Host 'Les poids des modèles d''IA sont téléchargés à leur première utilisation (TripoSR ~1,7 Go, TripoSG quelques Go).' }
