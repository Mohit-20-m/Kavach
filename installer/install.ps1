# ============================================================
# KAVACH — Supply Chain Security Installer for Windows
# Supports: Windows 10/11, PowerShell 5.1+, CMD, VS Code, Git Bash
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
    $script:PythonCompatible = "no"

    # Check version of whatever python/python3 is available
    foreach ($cmd in @("python3", "python")) {
        try {
            $result = & $cmd -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonCmd = $cmd
                $verCheck = & $cmd -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
                if ($verCheck -eq "3.10" -or $verCheck -eq "3.11") {
                    $script:PythonCompatible = "yes"
                    Write-OK "Python found: $cmd ($verCheck - ideal version)"
                } else {
                    Write-OK "Python found: $cmd ($verCheck)"
                    Write-Warn "Newer Python detected — will use flexible dependency versions"
                }
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
            Write-OK "Source downloaded via git"
        } catch {
            Write-Step "git failed, trying direct download..."
            $zipUrl = "https://github.com/$GITHUB_REPO/archive/refs/heads/main.zip"
            $zipPath = Join-Path $KAVACH_DIR "kavach.zip"
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
            Expand-Archive -Path $zipPath -DestinationPath $KAVACH_DIR -Force
            $extractedFolder = Join-Path $KAVACH_DIR "Kavach-main"
            if (Test-Path $extractedFolder) {
                if (Test-Path $KAVACH_SRC) { Remove-Item $KAVACH_SRC -Recurse -Force }
                Rename-Item $extractedFolder $KAVACH_SRC
            }
            Remove-Item $zipPath -ErrorAction SilentlyContinue
            Write-OK "Source downloaded via zip"
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
    $PythonExe = Join-Path $KAVACH_VENV "Scripts\python.exe"

    Write-Step "Installing dependencies (3-5 minutes)..."
    & $PythonExe -m pip install --quiet --upgrade pip 2>$null

    if ($script:PythonCompatible -eq "yes") {
        Write-OK "Using pinned, tested dependency versions"
        & $PipExe install --quiet "numpy==1.26.3" "scikit-learn==1.4.0" "xgboost==2.0.3"
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Pinned versions failed, falling back to flexible versions"
            & $PipExe install --quiet "numpy>=1.26" "scikit-learn>=1.4" "xgboost>=2.0"
        }
    } else {
        Write-OK "Using flexible dependency versions for this Python version"
        & $PipExe install --quiet "numpy>=1.26" "scikit-learn>=1.4" "xgboost>=2.0"
    }

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
    Write-OK "kavach-standalone.bat created"
}

function Create-NpmPipWrappers {
    Write-Section "Creating npm and pip Wrappers (works in ALL terminals)"

    $KavachStandalone = Join-Path $KAVACH_BIN "kavach-standalone.bat"

    # ── npm.bat — replaces npm in every terminal ──────────────────────────────
    $npmWrapper = "@echo off`r`n"
    $npmWrapper += "rem KAVACH npm wrapper — intercepts npm install`r`n"
    $npmWrapper += "`"$KavachStandalone`" npm %*`r`n"
    Set-Content -Path (Join-Path $KAVACH_BIN "npm.bat") -Value $npmWrapper -Encoding ASCII
    Set-Content -Path (Join-Path $KAVACH_BIN "npm.cmd") -Value $npmWrapper -Encoding ASCII
    Write-OK "npm.bat wrapper created"

    # ── pip.bat — replaces pip in every terminal ──────────────────────────────
    $pipWrapper = "@echo off`r`n"
    $pipWrapper += "rem KAVACH pip wrapper — intercepts pip install`r`n"
    $pipWrapper += "`"$KavachStandalone`" pip %*`r`n"
    Set-Content -Path (Join-Path $KAVACH_BIN "pip.bat") -Value $pipWrapper -Encoding ASCII
    Set-Content -Path (Join-Path $KAVACH_BIN "pip.cmd") -Value $pipWrapper -Encoding ASCII
    Set-Content -Path (Join-Path $KAVACH_BIN "pip3.bat") -Value $pipWrapper -Encoding ASCII
    Set-Content -Path (Join-Path $KAVACH_BIN "pip3.cmd") -Value $pipWrapper -Encoding ASCII
    Write-OK "pip.bat wrapper created"

    # ── Add KAVACH_BIN to FRONT of system PATH ────────────────────────────────
    # This means Windows finds KAVACH's npm.bat BEFORE the real npm
    # Works in CMD, PowerShell, VS Code terminal, Git Bash — every terminal
    $systemPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    if ($systemPath -notlike "*$KAVACH_BIN*") {
        [Environment]::SetEnvironmentVariable("PATH", "$KAVACH_BIN;$systemPath", "Machine")
        Write-OK "KAVACH_BIN added to FRONT of system PATH"
        Write-OK "npm and pip are now intercepted in ALL terminals"
    } else {
        # Make sure it's at the front
        $cleanPath = ($systemPath -split ";" | Where-Object { $_ -ne $KAVACH_BIN }) -join ";"
        [Environment]::SetEnvironmentVariable("PATH", "$KAVACH_BIN;$cleanPath", "Machine")
        Write-OK "KAVACH_BIN moved to front of system PATH"
    }

    # Also add to user PATH as backup
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($userPath -notlike "*$KAVACH_BIN*") {
        [Environment]::SetEnvironmentVariable("PATH", "$KAVACH_BIN;$userPath", "User")
        Write-OK "KAVACH_BIN added to user PATH"
    }

    # ── PowerShell profile (extra layer for PowerShell) ───────────────────────
    $ProfilePath = $PROFILE.CurrentUserAllHosts
    $ProfileDir = Split-Path $ProfilePath -Parent
    if (-not (Test-Path $ProfileDir)) { New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null }
    if (-not (Test-Path $ProfilePath)) { New-Item -ItemType File -Force -Path $ProfilePath | Out-Null }

    $ProfileContent = Get-Content $ProfilePath -Raw -ErrorAction SilentlyContinue
    if ($ProfileContent -notlike "*KAVACH Supply Chain Security*") {
        $block = "`r`n# --- KAVACH Supply Chain Security ---`r`n"
        $block += "`$env:PATH = `"$KAVACH_BIN;`" + `$env:PATH`r`n"
        $block += "`$env:KAVACH_MODELS_DIR = `"$KAVACH_MODELS`"`r`n"
        $block += "# --- END KAVACH ---`r`n"
        Add-Content -Path $ProfilePath -Value $block
        Write-OK "Added to PowerShell profile"
    } else {
        Write-OK "PowerShell profile already configured"
    }

    # ── CMD autorun (extra layer for CMD) ─────────────────────────────────────
    $BatchProfile = Join-Path $KAVACH_BIN "kavach-init.bat"
    $batchContent = "@echo off`r`nset PATH=$KAVACH_BIN;%PATH%`r`nset KAVACH_MODELS_DIR=$KAVACH_MODELS`r`n"
    Set-Content -Path $BatchProfile -Value $batchContent -Encoding ASCII

    try {
        Set-ItemProperty -Path "HKCU:\Software\Microsoft\Command Processor" `
            -Name "AutoRun" `
            -Value "`"$BatchProfile`"" `
            -ErrorAction SilentlyContinue
        Write-OK "CMD autorun configured"
    } catch {
        Write-Warn "Could not configure CMD autorun (non-critical)"
    }
}

function Create-ToggleScripts {
    Write-Section "Creating Enable/Disable Commands"

    $DisablePath = Join-Path $KAVACH_BIN "kavach-disable.bat"
    $disableScript = "@echo off`r`n"
    $disableScript += "echo Disabling KAVACH...`r`n"
    $disableScript += "powershell -ExecutionPolicy Bypass -Command `""
    $disableScript += "`$sys = [Environment]::GetEnvironmentVariable('PATH','Machine'); "
    $disableScript += "`$clean = (`$sys -split ';' | Where-Object { `$_ -notlike '*\.kavach\bin*' }) -join ';'; "
    $disableScript += "[Environment]::SetEnvironmentVariable('PATH', `$clean, 'Machine'); "
    $disableScript += "`$usr = [Environment]::GetEnvironmentVariable('PATH','User'); "
    $disableScript += "`$clean2 = (`$usr -split ';' | Where-Object { `$_ -notlike '*\.kavach\bin*' }) -join ';'; "
    $disableScript += "[Environment]::SetEnvironmentVariable('PATH', `$clean2, 'User'); "
    $disableScript += "Write-Host 'KAVACH disabled. Restart all terminals to apply.' -ForegroundColor Yellow`""
    Set-Content -Path $DisablePath -Value $disableScript -Encoding ASCII
    Write-OK "kavach-disable.bat created"

    $EnablePath = Join-Path $KAVACH_BIN "kavach-enable.bat"
    $enableScript = "@echo off`r`n"
    $enableScript += "echo Enabling KAVACH...`r`n"
    $enableScript += "powershell -ExecutionPolicy Bypass -Command `""
    $enableScript += "`$sys = [Environment]::GetEnvironmentVariable('PATH','Machine'); "
    $enableScript += "if (`$sys -notlike '*\.kavach\bin*') { "
    $enableScript += "[Environment]::SetEnvironmentVariable('PATH', '$KAVACH_BIN;' + `$sys, 'Machine') }; "
    $enableScript += "Write-Host 'KAVACH enabled. Restart all terminals to apply.' -ForegroundColor Green`""
    Set-Content -Path $EnablePath -Value $enableScript -Encoding ASCII
    Write-OK "kavach-enable.bat created"
}

function Verify-Installation {
    Write-Section "Verifying Installation"

    $checks = @(
        (Join-Path $KAVACH_BIN "kavach-standalone.bat"),
        (Join-Path $KAVACH_BIN "npm.bat"),
        (Join-Path $KAVACH_BIN "pip.bat")
    )

    foreach ($f in $checks) {
        if (Test-Path $f) { Write-OK "Found: $(Split-Path $f -Leaf)" }
        else { Write-Fail "Missing: $(Split-Path $f -Leaf)" }
    }

    $ModelCount = (Get-ChildItem $KAVACH_MODELS -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-OK "Model files: $ModelCount"

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
    Write-Host "  IMPORTANT: Close ALL terminals and reopen them." -ForegroundColor Yellow
    Write-Host "  This applies to: PowerShell, CMD, VS Code, Git Bash" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Then test in any terminal:" -ForegroundColor White
    Write-Host "    npm install lodash      -> Should show SAFE" -ForegroundColor Cyan
    Write-Host "    npm install yoshi-base  -> Should show CRITICAL BLOCKED" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  To disable: kavach-disable" -ForegroundColor Gray
    Write-Host "  To enable:  kavach-enable" -ForegroundColor Gray
    Write-Host ""
}

# --- Main ---
Write-Banner
$PythonCmd = Check-Requirements
Download-Kavach
Setup-Venv -PythonCmd $PythonCmd
Download-Models
Create-Wrapper
Create-NpmPipWrappers
Create-ToggleScripts
Verify-Installation
Print-Success
