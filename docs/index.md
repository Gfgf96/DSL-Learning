# Sign Language Learning App

Welcome to the documentation for the **DSL Learning** project! This application is designed to help users learn the NGT (Nederlandse Gebarentaal / Dutch Sign Language) alphabet using real-time computer vision and deep learning.

## Quick Links
- [GitHub Repository](https://github.com/Gfgf96/DSL-Learning)
- [Deployment Guide](deployment.md)

## Application Overview

The app uses MediaPipe to detect hand landmarks and passes them to PyTorch models for classification:
1. **Static CNN Model**: Predicts letters A-I, K-Y.
2. **Dynamic LSTM Model**: Predicts letters with movement (J, Z) by evaluating a sequence of 30 frames.

## Getting Started Locally

Please refer to the setup scripts located in the `scripts/` directory of the repository to automatically install dependencies and start the app:

```bash
# On Windows (PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\scripts\setup-windows-pwsh.ps1

# On macOS/Linux
./scripts/setup-unix.sh
```

Then start the application:
```bash
uv run ./main.py
```
