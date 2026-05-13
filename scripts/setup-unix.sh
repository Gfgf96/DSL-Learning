#!/bin/bash

# DSL-Learning Setup Script for Unix/Linux/macOS
# This script automates the entire setup process including uv, Python dependencies, and frontend build

set -e

# Navigate to project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "DSL-Learning - Automated Setup Script"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Step 1: Check and install uv if needed
echo "[1/5] Checking for uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "  uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the uv installation
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "  uv installed successfully!"
else
    echo "  uv is already installed"
fi

# Step 2: Check and install Node.js if needed
echo "[2/5] Checking for Node.js..."
if ! command -v node &> /dev/null; then
    echo "  Node.js not found. Installing..."
    if command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew install node
    elif command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y nodejs npm
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y nodejs npm
    else
        echo "  ERROR: Could not determine package manager to install Node.js"
        echo "  Please install Node.js manually: https://nodejs.org/"
        exit 1
    fi
    echo "  Node.js installed successfully!"
else
    nodeVersion=$(node --version)
    echo "  Node.js is installed ($nodeVersion)"
fi

# Step 3: Install Python dependencies with uv
echo "[3/5] Installing Python dependencies with uv..."
if ! uv sync; then
    echo "  ERROR: Failed to sync Python dependencies"
    exit 1
fi
echo "  Python dependencies installed successfully!"

# Step 4: Install frontend dependencies
echo "[4/5] Installing frontend dependencies..."
if ! (cd frontend && npm install); then
    echo "  ERROR: Failed to install frontend dependencies"
    exit 1
fi
echo "  Frontend dependencies installed successfully!"

# Step 5: Verify setup
echo "[5/5] Verifying setup..."
pythonVersion=$(uv run python --version)
echo "  Python: $pythonVersion"
echo "  Node.js: $(node --version)"
echo "  npm: $(npm --version)"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To start the application, run:"
echo "  uv run ./main.py"
echo ""
