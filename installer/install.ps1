# ============================================================
# KAVACH — Supply Chain Security Installer for Windows
# Supports: Windows 10/11, PowerShell 5.1+
# ============================================================

$ErrorActionPreference = "Stop"
$KAVACH_VERSION = "1.0.0"
$GITHUB_REPO = "Mohit-20-m/Kavach"
$KAVACH_DIR = Join-Path $env:USERPROFILE ".kavach"
$KAVACH_BIN = Join-Path $KAVACH_DIR "bin"
$KAVACH_MODELS = Join-Path $KAVACH_DIR "models"
$KAVACH_VENV = Join-Path $KAVACH_DIR "venv"
$KAVACH_SRC = Join-Path $KAVACH_DIR "src"

function Write-Step    { param($msg) Write-Host "`n  -> $msg" -ForegroundColor Cyan }
function Write-OK      { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn    { param($msg) Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Write-Fail    { param($msg) Write-Host "  [X] $msg" -ForegroundColor Red }
function Write-Section { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor White }

function Write-Banner {
    Write-Host ""
    Write-Host "  KAVACH - Supply Chain Security" -ForegroundColor Cyan
    Write-Host "  Agentic Behavioral Shield for Open Source Supply Chain Security" -ForegroundColor White
    Write-Host "  Version $KAVACH_VERSION" -ForegroundColor Gray
    Write-Host ""
}

function Check-Requirements {
    Write-Section "Checking Requirements"

    $PythonCmd = $null
    foreach ($cmd in @("python3", "python")) {
        try {
            $result = & $cmd -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonCmd = $cmd
                Write-OK "Python found: $cmd"
                break
            }
        } catch {}
    }

    if (-not $PythonCmd) {
        Write-Fail "Python 3.10+ is required but not found."
        Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  Check 'Add Python to PATH' during install." -ForegroundColor Yellow
        Start-Process "https://www.python.org/downloads/"
        exit 1
    }

    try {
        $null = git --version 2>$null
        Write-OK "git found"
    } catch {
        Write-Warn "git not found. Trying winget..."
        try {
            winget install --id Git.Git -e --source winget --silent
            Write-OK "git installed"
        } catch {
            Write-Fail "Could not install git."
            Write-Host "  Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
            exit 1
        }
    }

    return $PythonCmd
}

function Download-Kavach {
    Write-Section "Downloading KAVACH"

    New-Item -ItemType Directory -Force -Path $KAVACH_DIR | Out-Null
    New-Item -ItemType Directory -Force -Path $KAVACH_BIN | Out-Null
    New-Item -ItemType Directory -Force -Path $KAVACH_MODELS | Out-Null

    if (Test-Path (Join-Path $KAVACH_SRC ".git")) {
        Write-Step "Updating existing installation..."
        Push-Location $KAVACH_SRC
        git pull --quiet
        Pop-Location
        Write-OK "Updated"
    } else {
        Write-Step "Downloading KAVACH source..."
        try {
            git clone --quiet --depth=1 "https://github.com/$GITHUB_REPO.git" $KAVACH_SRC
            Write-OK "Source downloaded"
        } catch {
            Write-Step "Trying direct download..."
            $zipUrl = "https://github.com/$GITHUB_REPO/archive/refs/heads/main.zip"
            $zipPath = Join-Path $KAVACH_DIR "kavach.zip"
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
            Expand-Archive -Path $zipPath -DestinationPath $KAVACH_DIR -Force
            $extractedFolder = Join-Path $KAVACH_DIR "Kavach-main"
            if (Test-Path $extractedFolder) {
                Rename-Item $extractedFolder $KAVACH_SRC
            }
            Remove-Item $zipPath -ErrorAction SilentlyContinue
            Write-OK "Source downloaded"
        }
    }
}

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

    Write-Step "Installing dependencies (3-5 minutes)..."
    & $PipExe install --quiet --upgrade pip
    & $PipExe install --quiet "numpy==1.26.3" "scikit-learn==1.4.0" "xgboost==2.0.3"
    & $PipExe install --quiet "torch" "--index-url" "https://download.pytorch.org/whl/cpu"
    & $PipExe install --quiet "sentence-transformers" "httpx" "typer" "rich" "aiofiles" "pydantic"
    Write-OK "Dependencies installed"

    $CliPath = Join-Path $KAVACH_SRC "cli"
    if (Test-Path (Join-Path $CliPath "setup.py")) {
        & $PipExe install --quiet -e $CliPath
        Write-OK "KAVACH CLI installed"
    }
}

function Download-Models {
    Write-Section "Downloading AI Models"

    $modelCheck = Join-Path $KAVACH_MODELS "code_archaeologist.pkl"
    if (Test-Path $modelCheck) {
        Write-OK "Models already present"
        return
    }

    Write-Step "Downloading models package..."
    $zipUrl = "https://github.com/Mohit-20-m/Kavach/releases/download/v1.0.0/models.zip"
    $zipPath = Join-Path $KAVACH_DIR "models.zip"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing

    Write-Step "Extracting models..."
    Expand-Archive -Path $zipPath -DestinationPath $KAVACH_DIR -Force
    Remove-Item $zipPath -ErrorAction SilentlyContinue
    Write-OK "Models ready"
}

function Create-Wrapper {
    Write-Section "Creating KAVACH Executable"

    $KavachExe = Join-Path $KAVACH_VENV "Scripts\kavach-standalone.exe"
    $WrapperPath = Join-Path $KAVACH_BIN "kavach-standalone.bat"

    $wrapperContent = "@echo off`r`nset KAVACH_MODELS_DIR=" + $KAVACH_MODELS + "`r`n`"" + $KavachExe + "`" %*"
    Set-Content -Path $WrapperPath -Value $wrapperContent -Encoding ASCII
    Write-OK "Wrapper created at $WrapperPath"
}

function Setup-Shell {
    Write-Section "Setting Up Shell Intercepts"

    $WrapperBat = Join-Path $KAVACH_BIN "kavach-standalone.bat"

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
        $block = "`r`n# --- KAVACH Supply Chain Security ---`r`n"
        $block += "`$env:PATH = `"$KAVACH_BIN;`" + `$env:PATH`r`n"
        $block += "`$env:KAVACH_MODELS_DIR = `"$KAVACH_MODELS`"`r`n"
        $block += "function npm { `$k = `"$WrapperBat`"; if (Test-Path `$k) { & `$k npm `$args } else { npm.cmd `$args } }`r`n"
        $block += "function pip { `$k = `"$WrapperBat`"; if (Test-Path `$k) { & `$k pip `$args } else { pip.exe `$args } }`r`n"
        $block += "function pip3 { `$k = `"$WrapperBat`"; if (Test-Path `$k) { & `$k pip `$args } else { pip3.exe `$args } }`r`n"
        $block += "# --- END KAVACH ---`r`n"

        Add-Content -Path $ProfilePath -Value $block
        Write-OK "Added intercepts to PowerShell profile"
    } else {
        Write-OK "Already configured"
    }

    $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($CurrentPath -notlike "*$KAVACH_BIN*") {
        [Environment]::SetEnvironmentVariable("PATH", "$KAVACH_BIN;$CurrentPath", "User")
        Write-OK "Added $KAVACH_BIN to PATH"
    }
}

function Create-ToggleScripts {
    Write-Section "Creating Enable/Disable Commands"

    # kavach-disable.bat
    $DisableBat = Join-Path $KAVACH_BIN "kavach-disable.bat"
    $disableContent = "@echo off`r`npowershell -ExecutionPolicy Bypass -Command `"" +
        "`$p = `$PROFILE.CurrentUserAllHosts; " +
        "`$c = Get-Content `$p -Raw; " +
        "`$c = `$c -replace '(?ms)# --- KAVACH Supply Chain Security ---.*?# --- END KAVACH ---\r?\n', ''; " +
        "Set-Content `$p `$c; " +
        "Write-Host 'KAVACH disabled. Restart PowerShell to apply.' -ForegroundColor Yellow`""
    Set-Content -Path $DisableBat -Value $disableContent -Encoding ASCII
    Write-OK "kavach-disable.bat created"

    # kavach-enable.bat
    $EnableBat = Join-Path $KAVACH_BIN "kavach-enable.bat"
    $WrapperBat = Join-Path $KAVACH_BIN "kavach-standalone.bat"
    $enableContent = "@echo off`r`npowershell -ExecutionPolicy Bypass -Command `"" +
        "`$p = `$PROFILE.CurrentUserAllHosts; " +
        "`$c = Get-Content `$p -Raw -ErrorAction SilentlyContinue; " +
        "if (`$c -notlike '*KAVACH Supply Chain Security*') { " +
        "Add-Content `$p '``r``n# --- KAVACH Supply Chain Security ---'; " +
        "Write-Host 'KAVACH enabled. Restart PowerShell.' -ForegroundColor Green " +
        "} else { Write-Host 'KAVACH already enabled.' -ForegroundColor Yellow }`""
    Set-Content -Path $EnableBat -Value $enableContent -Encoding ASCII
    Write-OK "kavach-enable.bat created"
}

function Verify-Installation {
    Write-Section "Verifying Installation"

    $WrapperBat = Join-Path $KAVACH_BIN "kavach-standalone.bat"
    if (Test-Path $WrapperBat) { Write-OK "kavach-standalone.bat found" }
    else { Write-Fail "kavach-standalone.bat missing" }

    $ModelCount = (Get-ChildItem $KAVACH_MODELS -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-OK "Model files present: $ModelCount"

    $PythonExe = Join-Path $KAVACH_VENV "Scripts\python.exe"
    try {
        $check = & $PythonExe -c "import xgboost, sklearn; print('ok')" 2>$null
        if ($check -eq "ok") { Write-OK "Python dependencies OK" }
    } catch {
        Write-Warn "Some Python dependencies may be missing"
    }
}

function Print-Success {
    Write-Host ""
    Write-Host "=================================================" -ForegroundColor Green
    Write-Host "   KAVACH Installation Complete!" -ForegroundColor Green
    Write-Host "=================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Restart PowerShell, then try:" -ForegroundColor White
    Write-Host "    npm install lodash      -> Should show SAFE" -ForegroundColor Cyan
    Write-Host "    npm install yoshi-base  -> Should show CRITICAL BLOCKED" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  To disable: run kavach-disable" -ForegroundColor Gray
    Write-Host "  To enable:  run kavach-enable" -ForegroundColor Gray
    Write-Host ""
}

# --- Main ---
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
