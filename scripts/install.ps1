<#
.SYNOPSIS
    Install, upgrade, or uninstall Theo OpenVoice on Windows.

.DESCRIPTION
    Downloads uv (if not present) and installs Theo OpenVoice in an isolated
    Python 3.12 virtual environment.

    Quick install:

        irm https://raw.githubusercontent.com/usetheo/theo-openvoice/main/scripts/install.ps1 | iex

    Specific version:

        $env:THEO_VERSION="0.1.0"; irm https://raw.githubusercontent.com/usetheo/theo-openvoice/main/scripts/install.ps1 | iex

    Custom install directory:

        $env:THEO_INSTALL_DIR="D:\Theo"; irm https://raw.githubusercontent.com/usetheo/theo-openvoice/main/scripts/install.ps1 | iex

    Uninstall:

        $env:THEO_UNINSTALL=1; irm https://raw.githubusercontent.com/usetheo/theo-openvoice/main/scripts/install.ps1 | iex

    Environment variables:

        THEO_VERSION       Pin to a specific version (default: latest)
        THEO_INSTALL_DIR   Custom install directory
        THEO_EXTRAS        Pip extras to install (default: server,grpc)
        THEO_UNINSTALL     Set to 1 to uninstall Theo OpenVoice

.EXAMPLE
    irm https://raw.githubusercontent.com/usetheo/theo-openvoice/main/scripts/install.ps1 | iex

.LINK
    https://github.com/usetheo/theo-openvoice
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# --------------------------------------------------------------------------
# Configuration from environment variables
# --------------------------------------------------------------------------

$Version    = if ($env:THEO_VERSION) { $env:THEO_VERSION } else { "" }
$InstallDir = if ($env:THEO_INSTALL_DIR) { $env:THEO_INSTALL_DIR } else { "" }
$Extras     = if ($env:THEO_EXTRAS) { $env:THEO_EXTRAS } else { "server,grpc" }
$Uninstall  = $env:THEO_UNINSTALL -eq "1"

$PythonVersion = "3.12"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

function Write-Step {
    param([string]$Message)
    Write-Host ">>> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Get-TheoDir {
    if ($InstallDir) {
        return $InstallDir
    }
    return Join-Path $env:LOCALAPPDATA "Programs\Theo"
}

function Test-UvAvailable {
    try {
        $null = Get-Command uv -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# --------------------------------------------------------------------------
# Uninstall
# --------------------------------------------------------------------------

function Invoke-Uninstall {
    $theoDir = Get-TheoDir

    if (-not (Test-Path $theoDir)) {
        Write-Host "Theo OpenVoice is not installed at $theoDir."
        return
    }

    Write-Step "Uninstalling Theo OpenVoice from $theoDir..."

    # Remove from PATH
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $scriptsDir = Join-Path $theoDir ".venv\Scripts"
    if ($userPath -and $userPath.Contains($scriptsDir)) {
        $newPath = ($userPath -split ";" | Where-Object { $_ -ne $scriptsDir }) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "  Removed $scriptsDir from user PATH."
    }

    # Remove install directory
    Remove-Item -Recurse -Force $theoDir -ErrorAction SilentlyContinue
    Write-Host "  Removed $theoDir."

    Write-Success "Theo OpenVoice has been uninstalled."
}

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------

function Invoke-Install {
    $theoDir = Get-TheoDir

    Write-Step "Installing Theo OpenVoice..."

    # --- Install uv if not present ---
    if (-not (Test-UvAvailable)) {
        Write-Step "Installing uv (Python package manager)..."
        try {
            Invoke-Expression "& { $(Invoke-RestMethod https://astral.sh/uv/install.ps1) }"
        } catch {
            throw "Failed to install uv. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
        }

        # Refresh PATH for current session
        $uvDir = Join-Path $env:LOCALAPPDATA "uv"
        if (Test-Path $uvDir) {
            $env:PATH = "$uvDir;$env:PATH"
        }
        $cargoDir = Join-Path $env:USERPROFILE ".cargo\bin"
        if (Test-Path $cargoDir) {
            $env:PATH = "$cargoDir;$env:PATH"
        }
    }

    if (-not (Test-UvAvailable)) {
        throw "uv is not available after installation. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
    }

    $uvVersion = & uv --version
    Write-Host "  uv version: $uvVersion"

    # --- Create install directory ---
    Write-Step "Creating install directory at $theoDir..."
    if (-not (Test-Path $theoDir)) {
        New-Item -ItemType Directory -Path $theoDir -Force | Out-Null
    }

    # --- Create venv with Python 3.12 ---
    Write-Step "Creating Python $PythonVersion environment..."
    & uv venv --python $PythonVersion (Join-Path $theoDir ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Python $PythonVersion virtual environment."
    }

    # --- Install theo-openvoice ---
    $theoPkg = "theo-openvoice[$Extras]"
    if ($Version) {
        $theoPkg = "theo-openvoice[$Extras]==$Version"
    }
    Write-Step "Installing $theoPkg..."
    $pythonExe = Join-Path $theoDir ".venv\Scripts\python.exe"
    & uv pip install --python $pythonExe $theoPkg
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $theoPkg."
    }

    # --- Add to PATH ---
    $scriptsDir = Join-Path $theoDir ".venv\Scripts"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath.Contains($scriptsDir)) {
        Write-Step "Adding theo to user PATH..."
        [Environment]::SetEnvironmentVariable("Path", "$scriptsDir;$userPath", "User")
        $env:PATH = "$scriptsDir;$env:PATH"
        Write-Host "  Added $scriptsDir to user PATH."
    }

    # --- GPU detection ---
    try {
        $nvsmi = Get-Command nvidia-smi -ErrorAction Stop
        Write-Step "NVIDIA GPU detected. Installing GPU-accelerated STT engine..."
        & uv pip install --python $pythonExe "theo-openvoice[faster-whisper]"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Failed to install faster-whisper GPU extras. You can install manually later." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  No NVIDIA GPU detected. Theo will run in CPU-only mode."
        Write-Host "  CPU-only mode is fully functional but slower for large models."
    }

    # --- Success ---
    Write-Success "Install complete. Run 'theo serve' to start the API server."
    Write-Host "  API will be available at http://127.0.0.1:8000"
    Write-Host ""
    Write-Host "  Quick start:"
    Write-Host "    theo serve"
    Write-Host "    theo pull faster-whisper-large-v3"
    Write-Host "    theo transcribe audio.wav"
    Write-Host ""
    Write-Host "  NOTE: You may need to restart your terminal for PATH changes to take effect."
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if ($Uninstall) {
    Invoke-Uninstall
} else {
    Invoke-Install
}
