# Setup Scripts

This directory contains automated setup scripts for different platforms. Run the appropriate script for your operating system to automatically install all dependencies and configure the project.

## Quick Start

You can run the scripts from **either the project root or the scripts directory** - they automatically navigate to the project root.

**From project root - Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\scripts\setup-windows-pwsh.ps1
```

**From project root - Windows Command Prompt:**
```cmd
scripts\setup-windows-cmd.bat
```

**From project root - macOS/Linux:**
```bash
chmod +x scripts/setup-unix.sh
./scripts/setup-unix.sh
```

**OR from the scripts directory:**
```bash
# PowerShell
.\setup-windows-pwsh.ps1

# Command Prompt
setup-windows-cmd.bat

# Bash
./setup-unix.sh
```

After setup completes successfully, start the app with:
```bash
uv run ./main.py
```

## Detailed Documentation

All setup scripts perform the following steps:

1. Install `uv` package manager (if not present)
2. Install Node.js (if not present)
3. Install Python dependencies via `uv sync`
4. Install frontend dependencies via `npm install`
5. Verify the complete setup

## Platform-Specific Notes

### Windows PowerShell

Requires execution policy adjustment:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\scripts\setup-windows-pwsh.ps1
```

### Windows Command Prompt

Run directly:
```cmd
scripts\setup-windows-cmd.bat
```

### macOS/Linux

Make executable and run:
```bash
chmod +x scripts/setup-unix.sh
./scripts/setup-unix.sh
```

## Troubleshooting

If a script fails:

1. Ensure you have Administrator/sudo access
2. Check that your internet connection is stable
3. Try manually running the steps documented in the script
4. Check the error message for specific component failures

For detailed error messages, you can run the scripts with verbose output:
- **PowerShell**: `./scripts/setup-windows-pwsh.ps1 -Verbose`
- **Bash**: `bash -x ./scripts/setup-unix.sh`
