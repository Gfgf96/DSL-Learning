@echo off
REM DSL-Learning Setup Script for Windows Command Prompt
REM This script automates the entire setup process including uv, Python dependencies, and frontend build

setlocal enabledelayedexpansion

REM Navigate to project root (parent of scripts directory)
cd /d "%~dp0.."
set projectRoot=%cd%

echo ============================================================
echo DSL-Learning - Automated Setup Script
echo ============================================================
echo Project root: %projectRoot%
echo.

REM Step 1: Check and install uv if needed
echo [1/5] Checking for uv package manager...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo   uv not found. Installing...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if !errorlevel! neq 0 (
        echo   ERROR: Failed to install uv
        exit /b 1
    )
    echo   uv installed successfully!
) else (
    echo   uv is already installed
)

REM Step 2: Check and install Node.js if needed
echo [2/5] Checking for Node.js...
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo   Node.js not found. Installing via winget...
    winget install OpenJS.NodeJS --accept-source-agreements --accept-package-agreements
    if !errorlevel! neq 0 (
        echo   ERROR: Failed to install Node.js
        exit /b 1
    )
    echo   Node.js installed successfully!
) else (
    for /f "tokens=*" %%i in ('node --version') do set nodeVersion=%%i
    echo   Node.js is installed !nodeVersion!
)

REM Step 3: Install Python dependencies with uv
echo [3/5] Installing Python dependencies with uv...
call uv sync
if %errorlevel% neq 0 (
    echo   ERROR: Failed to sync Python dependencies
    exit /b 1
)
echo   Python dependencies installed successfully!

REM Step 4: Install frontend dependencies
echo [4/5] Installing frontend dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo   ERROR: Failed to install frontend dependencies
    cd ..
    exit /b 1
)
cd ..
echo   Frontend dependencies installed successfully!

REM Step 5: Verify setup
echo [5/5] Verifying setup...
for /f "tokens=*" %%i in ('uv run python --version') do set pythonVersion=%%i
for /f "tokens=*" %%i in ('node --version') do set nodeVersion=%%i
for /f "tokens=*" %%i in ('npm --version') do set npmVersion=%%i
echo   Python: !pythonVersion!
echo   Node.js: !nodeVersion!
echo   npm: !npmVersion!

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To start the application, run:
echo   uv run .\main.py
echo.
