# ============================================================
# KAVACH — Supply Chain Security Installer for Windows
# Supports: Windows 10/11, PowerShell 5.1+
# Run as: iex (irm https://get.kavach.dev/install.ps1)
# ============================================================

$ErrorActionPreference = "Stop"
$KAVACH_VERSION = "1.0.0"
$GITHUB_REPO = "kavach-security/kavach"
$KAVACH_DIR = Join-Path $env:USERPROFILE ".kavach"
$KAVACH_BIN = Join-Path $KAVACH_DIR "bin"
$KAVACH_MODELS = Join-Path $KAVACH_DIR "models"
$KAVACH_VENV = Join-Path $KAVACH_DIR "venv"
$KAVACH_SRC = Join-Path $KAVACH_DIR "src"

# ─── Colors ──────────────────────────────────────────────────────────────────
function Write-Step   { param($msg) Write-Host "`n  → $msg" -ForegroundColor Cyan -NoNewline; Write-Host "" }
function Write-OK     { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn   { param($msg) Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Write-Fail   { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }
function Write-Section{ param($msg) Write-Host "`n$msg" -ForegroundColor White -BackgroundColor DarkBlue; Write-Host "  $('─' * 50)" }

function Write-Banner {
    Write-Host ""
    Write-Host "  ██╗  ██╗ █████╗ ██╗   ██╗ █████╗  ██████╗██╗  ██╗" -ForegroundColor Cyan
    Write-Host "  ██║ ██╔╝██╔══██╗██║   ██║██╔══██╗██╔════╝██║  ██║" -ForegroundColor Cyan
    Write-Host "  █████╔╝ ███████║██║   ██║███████║██║     ███████║" -ForegroundColor Cyan
    Write-Host "  ██╔═██╗ ██╔══██║╚██╗ ██╔╝██╔══██║██║     ██╔══██║" -ForegroundColor Cyan
    Write-Host "  ██║  ██╗██║  ██║ ╚████╔╝ ██║  ██║╚██████╗██║  ██║" -ForegroundColor Cyan
    Write-Host "  ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Agentic Behavioral Shield for Open Source Supply Chain Security" -ForegroundColor White
    Write-Host "  Version $KAVACH_VERSION" -ForegroundColor Gray
    Write-Host ""
}

# ─── Check Requirements ───────────────────────────────────────────────────────
function Check-Requirements {
    Write-Section "Checking Requirements"

    # Python
    $PythonCmd = $null
    foreach ($cmd in @("python3", "python")) {
        try {
            $ver = & $cmd -c "import sys; print(sys.version_info[:2])" 2>$null
            $ok = & $cmd -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonCmd = $cmd
                Write-OK "Python found: $cmd ($ver)"
                break
            }
        } catch {}
    }

    if (-not $PythonCmd) {
        Write-Fail "Python 3.10+ is required but not found."
        Write-Host ""
        Write-Host "  Download Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to open the Python download page..."
        Start-Process "https://www.python.org/downloads/"
        exit 1
    }

    # Git
    try {
        $gitVer = git --version 2>$null
        Write-OK "git found: $gitVer"
    } catch {
        Write-Warn "git not found. Installing via winget..."
        try {
            winget install --id Git.Git -e --source winget --silent
            Write-OK "git installed"
        } catch {
            Write-Fail "Could not install git automatically."
            Write-Host "  Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
            exit 1
        }
    }

    return $PythonCmd
}

# ─── Download KAVACH ──────────────────────────────────────────────────────────
function Download-Kavach {
    Write-Section "Downloading KAVACH"

    New-Item -ItemType Directory -Force -Path $KAVACH_DIR | Out-Null
    New-Item -ItemType Directory -Force -Path $KAVACH_BIN | Out-Null
    New-Item -ItemType Directory -Force -Path $KAVACH_MODELS | Out-Null

    if (Test-Path (Join-Path $KAVACH_SRC ".git")) {
        Write-Step "Updating existing installation..."
        Set-Location $KAVACH_SRC
        git pull --quiet
        Write-OK "Updated"
    } else {
        Write-Step "Downloading KAVACH source..."
        try {
            git clone --quiet --depth=1 "https://github.com/$GITHUB_REPO.git" $KAVACH_SRC 2>$null
            Write-OK "Source downloaded"
        } catch {
            Write-Step "Trying direct download..."
            $zipUrl = "https://github.com/$GITHUB_REPO/archive/refs/heads/main.zip"
            $zipPath = Join-Path $KAVACH_DIR "kavach.zip"
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
            Expand-Archive -Path $zipPath -DestinationPath $KAVACH_DIR -Force
            Rename-Item (Join-Path $KAVACH_DIR "kavach-main") $KAVACH_SRC
            Remove-Item $zipPath
            Write-OK "Source downloaded"
        }
    }
}

# ─── Setup Python venv ────────────────────────────────────────────────────────
function Setup-Venv {
    param($PythonCmd)
    Write-Section "Setting Up Python Environment"

    if (-not (Test-Path $KAVACH_VENV)) {
        Write-Step "Creating virtual environment..."
        & $PythonCmd -m venv $KAVACH_VENV
        Write-OK "Virtual environment created"
    } else {
        Write-OK "Virtual environment already exists"
    }

    $PipExe = Join-Path $KAVACH_VENV "Scripts\pip.exe"
    $PythonExe = Join-Path $KAVACH_VENV "Scripts\python.exe"

    Write-Step "Installing dependencies (this may take 3-5 minutes)..."
    & $PipExe install --quiet --upgrade pip
    & $PipExe install --quiet `
        "numpy==1.26.3" `
        "scikit-learn==1.4.0" `
        "xgboost==2.0.3" `
        "torch" "--index-url" "https://download.pytorch.org/whl/cpu" `
        "sentence-transformers" `
        "httpx" `
        "typer" `
        "rich" `
        "aiofiles" `
        "pydantic"

    Write-OK "Dependencies installed"

    # Install kavach CLI
    $CliPath = Join-Path $KAVACH_SRC "cli"
    if (Test-Path (Join-Path $CliPath "setup.py")) {
        & $PipExe install --quiet -e $CliPath
        Write-OK "KAVACH CLI installed"
    }
}

# ─── Download Models ──────────────────────────────────────────────────────────
function Download-Models {
    Write-Section "Downloading AI Models"

    $Models = @(
        "code_archaeologist.pkl",
        "maintainer_isolation_forest.pkl",
        "behavioral_isolation_forest.pkl",
        "lstm_autoencoder.pt",
        "meta_learner.pkl",
        "meta_learner_scaler.pkl",
        "score_thresholds.json",
        "agent_weights.json",
        "behavioral_metrics_col99.npy",
        "maintainer_profile_col99.npy"
    )

    $ModelBaseUrl = "https://github.com/$GITHUB_REPO/releases/download/v$KAVACH_VERSION/models"

    foreach ($model in $Models) {
        $dest = Join-Path $KAVACH_MODELS $model
        if (-not (Test-Path $dest)) {
            try {
                Invoke-WebRequest -Uri "$ModelBaseUrl/$model" -OutFile $dest -UseBasicParsing
                Write-OK "Downloaded: $model"
            } catch {
                Write-Warn "Could not download $model — will use defaults"
            }
        } else {
            Write-OK "Already have: $model"
        }
    }

    # Copy SBERT if in source
    $SbertSrc = Join-Path $KAVACH_SRC "data\models\sbert_fine_tuned"
    $SbertDst = Join-Path $KAVACH_MODELS "sbert_fine_tuned"
    if ((Test-Path $SbertSrc) -and (-not (Test-Path $SbertDst))) {
        Copy-Item $SbertSrc $SbertDst -Recurse
        Write-OK "SBERT model copied"
    }
}

# ─── Create wrapper batch file ────────────────────────────────────────────────
function Create-Wrapper {
    Write-Section "Creating KAVACH Executable"

    $PythonExe = Join-Path $KAVACH_VENV "Scripts\python.exe"
    $KavachExe = Join-Path $KAVACH_VENV "Scripts\kavach-standalone.exe"
    $WrapperPath = Join-Path $KAVACH_BIN "kavach-standalone.bat"

    @"
@echo off
set KAVACH_MODELS_DIR=$KAVACH_MODELS
"$KavachExe" %*
"@ | Set-Content $WrapperPath -Encoding ASCII

    Write-OK "Wrapper created at $WrapperPath"

    # Also create a .cmd version
    $CmdPath = Join-Path $KAVACH_BIN "kavach-standalone.cmd"
    @"
@echo off
set KAVACH_MODELS_DIR=$KAVACH_MODELS
"$KavachExe" %*
"@ | Set-Content $CmdPath -Encoding ASCII
}

# ─── Setup PowerShell intercepts ──────────────────────────────────────────────
function Setup-Shell {
    Write-Section "Setting Up Shell Intercepts"

    $WrapperBat = Join-Path $KAVACH_BIN "kavach-standalone.bat"

    $KavachBlock = @"

# ─── KAVACH Supply Chain Security ─────────────────────────────────
`$env:PATH = "$KAVACH_BIN;" + `$env:PATH
`$env:KAVACH_MODELS_DIR = "$KAVACH_MODELS"

function npm {
    `$kavach = "$KAVACH_BIN\kavach-standalone.bat"
    if (Test-Path `$kavach) { & `$kavach npm `$args } else { npm.cmd `$args }
}

function pip {
    `$kavach = "$KAVACH_BIN\kavach-standalone.bat"
    if (Test-Path `$kavach) { & `$kavach pip `$args } else { pip.exe `$args }
}

function pip3 {
    `$kavach = "$KAVACH_BIN\kavach-standalone.bat"
    if (Test-Path `$kavach) { & `$kavach pip `$args } else { pip3.exe `$args }
}
# ──────────────────────────────────────────────────────────────────
"@

    # Get PowerShell profile path
    $ProfilePath = $PROFILE.CurrentUserAllHosts
    $ProfileDir = Split-Path $ProfilePath -Parent

    if (-not (Test-Path $ProfileDir)) {
        New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
    }

    if (-not (Test-Path $ProfilePath)) {
        New-Item -ItemType File -Force -Path $ProfilePath | Out-Null
    }

    $ProfileContent = Get-Content $ProfilePath -Raw -ErrorAction SilentlyContinue
    if ($ProfileContent -notlike "*KAVACH Supply Chain Security*") {
        Add-Content -Path $ProfilePath -Value $KavachBlock
        Write-OK "Added intercepts to PowerShell profile: $ProfilePath"
    } else {
        Write-OK "Already configured in PowerShell profile"
    }

    # Add KAVACH_BIN to user PATH permanently
    $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($CurrentPath -notlike "*$KAVACH_BIN*") {
        [Environment]::SetEnvironmentVariable("PATH", "$KAVACH_BIN;$CurrentPath", "User")
        Write-OK "Added $KAVACH_BIN to User PATH"
    } else {
        Write-OK "PATH already configured"
    }

    # Also setup cmd.exe (for developers using cmd)
    $BatchProfile = Join-Path $KAVACH_BIN "kavach-init.bat"
    @"
@echo off
doskey npm="$WrapperBat" npm `$*
doskey pip="$WrapperBat" pip `$*
doskey pip3="$WrapperBat" pip `$*
"@ | Set-Content $BatchProfile -Encoding ASCII

    # Register batch profile for cmd.exe autorun
    try {
        Set-ItemProperty -Path "HKCU:\Software\Microsoft\Command Processor" `
            -Name "AutoRun" `
            -Value "`"$BatchProfile`"" `
            -ErrorAction SilentlyContinue
        Write-OK "CMD.exe intercepts configured"
    } catch {
        Write-Warn "Could not configure CMD.exe intercepts (non-critical)"
    }
}

# ─── Create disable/enable scripts ───────────────────────────────────────────
function Create-ToggleScripts {
    Write-Section "Creating Enable/Disable Commands"

    # kavach-disable.ps1
    $DisablePath = Join-Path $KAVACH_BIN "kavach-disable.ps1"
    @'
# KAVACH Disable Script
$ProfilePath = $PROFILE.CurrentUserAllHosts
if (Test-Path $ProfilePath) {
    $content = Get-Content $ProfilePath -Raw
    $content = $content -replace "(?ms)# ─── KAVACH Supply Chain Security.*?# ──────────────────────────────────────────────────────────────────\r?\n", ""
    Set-Content $ProfilePath $content
    Write-Host "✓ KAVACH disabled from PowerShell profile" -ForegroundColor Green
}

# Remove CMD autorun
try {
    Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Command Processor" -Name "AutoRun" -ErrorAction SilentlyContinue
    Write-Host "✓ KAVACH disabled from CMD.exe" -ForegroundColor Green
} catch {}

Write-Host ""
Write-Host "KAVACH disabled. Restart your terminal to apply." -ForegroundColor Yellow
Write-Host "To re-enable: kavach-enable" -ForegroundColor Yellow
'@ | Set-Content $DisablePath -Encoding UTF8

    # kavach-disable.bat (for cmd users)
    $DisableBat = Join-Path $KAVACH_BIN "kavach-disable.bat"
    @"
@echo off
powershell -ExecutionPolicy Bypass -File "$DisablePath"
"@ | Set-Content $DisableBat -Encoding ASCII

    Write-OK "kavach-disable created"

    # kavach-enable.ps1
    $EnablePath = Join-Path $KAVACH_BIN "kavach-enable.ps1"
    $KavachBinEscaped = $KAVACH_BIN -replace '\\', '\\'
    $KavachModelsEscaped = $KAVACH_MODELS -replace '\\', '\\'
    $WrapperBatEscaped = (Join-Path $KAVACH_BIN "kavach-standalone.bat") -replace '\\', '\\'

    @"
# KAVACH Enable Script
`$ProfilePath = `$PROFILE.CurrentUserAllHosts
`$ProfileDir = Split-Path `$ProfilePath -Parent
if (-not (Test-Path `$ProfileDir)) { New-Item -ItemType Directory -Force -Path `$ProfileDir | Out-Null }
if (-not (Test-Path `$ProfilePath)) { New-Item -ItemType File -Force -Path `$ProfilePath | Out-Null }

`$content = Get-Content `$ProfilePath -Raw -ErrorAction SilentlyContinue
if (`$content -notlike "*KAVACH Supply Chain Security*") {
    Add-Content -Path `$ProfilePath -Value @"
# ─── KAVACH Supply Chain Security ─────────────────────────────────
`\`$env:PATH = "$KAVACH_BIN;" + `\`$env:PATH
function npm { `\`$k = "$WrapperBatEscaped"; if (Test-Path `\`$k) { & `\`$k npm `\`$args } else { npm.cmd `\`$args } }
function pip { `\`$k = "$WrapperBatEscaped"; if (Test-Path `\`$k) { & `\`$k pip `\`$args } else { pip.exe `\`$args } }
function pip3 { `\`$k = "$WrapperBatEscaped"; if (Test-Path `\`$k) { & `\`$k pip `\`$args } else { pip3.exe `\`$args } }
# ──────────────────────────────────────────────────────────────────
"@
    Write-Host "✓ KAVACH re-enabled in PowerShell profile" -ForegroundColor Green
} else {
    Write-Host "KAVACH is already enabled" -ForegroundColor Yellow
}
Write-Host "Restart your terminal to apply." -ForegroundColor Cyan
"@ | Set-Content $EnablePath -Encoding UTF8

    $EnableBat = Join-Path $KAVACH_BIN "kavach-enable.bat"
    @"
@echo off
powershell -ExecutionPolicy Bypass -File "$EnablePath"
"@ | Set-Content $EnableBat -Encoding ASCII

    Write-OK "kavach-enable created"
}

# ─── Verify ───────────────────────────────────────────────────────────────────
function Verify-Installation {
    Write-Section "Verifying Installation"

    $WrapperBat = Join-Path $KAVACH_BIN "kavach-standalone.bat"
    if (Test-Path $WrapperBat) { Write-OK "kavach-standalone.bat: ✓" }
    else { Write-Fail "kavach-standalone.bat missing" }

    $ModelCount = (Get-ChildItem $KAVACH_MODELS -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-OK "Model files present: $ModelCount"

    $PythonExe = Join-Path $KAVACH_VENV "Scripts\python.exe"
    try {
        $check = & $PythonExe -c "import xgboost, sklearn, torch; print('ok')" 2>$null
        if ($check -eq "ok") { Write-OK "Python dependencies: ✓" }
    } catch {
        Write-Warn "Some Python dependencies may be missing"
    }
}

# ─── Success message ──────────────────────────────────────────────────────────
function Print-Success {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║       KAVACH Installation Complete!  🛡️           ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Restart PowerShell, then try:" -ForegroundColor White
    Write-Host ""
    Write-Host "    npm install lodash      " -ForegroundColor Cyan -NoNewline
    Write-Host "→ Should show SAFE" -ForegroundColor Gray
    Write-Host "    npm install yoshi-base  " -ForegroundColor Cyan -NoNewline
    Write-Host "→ Should show CRITICAL and BLOCK" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Commands available:" -ForegroundColor White
    Write-Host "    kavach-disable    " -ForegroundColor Cyan -NoNewline
    Write-Host "Disable KAVACH interception" -ForegroundColor Gray
    Write-Host "    kavach-enable     " -ForegroundColor Cyan -NoNewline
    Write-Host "Re-enable KAVACH" -ForegroundColor Gray
    Write-Host ""
}

# ─── Main ─────────────────────────────────────────────────────────────────────
Write-Banner
$PythonCmd = Check-Requirements
Download-Kavach
Setup-Venv -PythonCmd $PythonCmd
Download-Models
Create-Wrapper
Setup-Shell
Create-ToggleScripts
Verify-Installation
Print-Success
