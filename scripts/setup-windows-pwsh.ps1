# DSL-Learning Setup Script for Windows PowerShell
# This script automates the entire setup process including uv, Python dependencies, and frontend build

$ErrorActionPreference = "Stop"

# Navigate to project root (parent of scripts directory)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DSL-Learning - Automated Setup Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Project root: $projectRoot" -ForegroundColor Gray
Write-Host ""

# Step 1: Check and install uv if needed
Write-Host "[1/6] Checking for uv package manager..." -ForegroundColor Yellow
$uvExists = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)

if (-not $uvExists) {
    Write-Host "  uv not found. Installing..." -ForegroundColor Cyan
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    Write-Host "  uv installed successfully!" -ForegroundColor Green
} else {
    Write-Host "  uv is already installed" -ForegroundColor Green
}

# Step 2: Check and install Node.js if needed
Write-Host "[2/6] Checking for Node.js..." -ForegroundColor Yellow
$nodeExists = $null -ne (Get-Command node -ErrorAction SilentlyContinue)

if (-not $nodeExists) {
    Write-Host "  Node.js not found. Installing via winget..." -ForegroundColor Cyan
    winget install OpenJS.NodeJS --accept-source-agreements --accept-package-agreements
    Write-Host "  Node.js installed successfully! Please restart this script." -ForegroundColor Green
    exit 0
} else {
    $nodeVersion = node --version
    Write-Host "  Node.js is installed ($nodeVersion)" -ForegroundColor Green
}

# Step 3: Install Python dependencies with uv
Write-Host "[3/6] Installing Python dependencies with uv..." -ForegroundColor Yellow
try {
    uv sync
    Write-Host "  Python dependencies installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Failed to sync Python dependencies" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Install frontend dependencies and build
Write-Host "[4/5] Installing frontend dependencies..." -ForegroundColor Yellow
try {
    Push-Location frontend
    npm install
    Write-Host "  Frontend dependencies installed successfully!" -ForegroundColor Green
    
    Write-Host "[5/6] Building frontend export..." -ForegroundColor Yellow
    npm run build
    Write-Host "  Frontend build completed successfully!" -ForegroundColor Green
    
    Pop-Location
} catch {
    Write-Host "  ERROR: Failed to install or build frontend" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

# Step 6: Verify setup
Write-Host "[6/6] Verifying setup..." -ForegroundColor Yellow
try {
    $pythonVersion = uv run python --version
    Write-Host "  Python: $pythonVersion" -ForegroundColor Green
    Write-Host "  Node.js: $(node --version)" -ForegroundColor Green
    Write-Host "  npm: $(npm --version)" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Could not verify all components" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, run:" -ForegroundColor Cyan
Write-Host "  uv run .\main.py" -ForegroundColor White
Write-Host ""
