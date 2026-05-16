# Deployment & Architecture

This guide explains the architecture of the DSL Learning app and how it is deployed.

## Local Development Architecture

During local development (and currently in production), the app uses a tightly coupled architecture:

1. **FastAPI Backend (`main.py`)**:
   - Handles the WebSocket connections (`/ws`) for real-time model inference.
   - Provides REST APIs (`/api/*`) for fetching assets and managing sessions.
   - Serves the compiled Next.js frontend as static files.

2. **Next.js Frontend (`frontend/`)**:
   - Built as a Static HTML Export (`output: 'export'` in `next.config.ts`).
   - The compiled files in `frontend/out/` are mounted by FastAPI and served directly.

When you run `main.py`, it automatically checks for model weights, ensures the frontend is built, and starts the Uvicorn server on `localhost:8000`.

## Production Deployment

For a highly scalable production environment, the application is containerized using **Docker** and continuously integrated via **GitHub Actions**.

### Docker Architecture
The project uses a **multi-stage Docker build** to keep image sizes small and builds incredibly fast:
1. **Node.js Stage**: Installs npm dependencies and builds the Next.js static export (`frontend/out`).
2. **Python Stage**: Uses the highly optimized `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` base image. It syncs the Python environment perfectly using `uv sync --frozen --no-dev`, copies the frontend export, and sets the entrypoint.

> **Note:** The MediaPipe and PyTorch models are explicitly **not baked into the Docker image**. The container dynamically downloads them at runtime using the logic inside `main.py` (`check_and_download_models`). This ensures the core Docker image remains small and easy to pull.

### Continuous Integration (CI/CD)

The project leverages GitHub Actions for CI/CD. The workflows can be found in `.github/workflows/`:

1. **`docs.yml`**: Automatically builds and deploys this MkDocs documentation to GitHub Pages whenever changes are merged to the `main` branch.
2. **`ci.yml`**: The main CI/CD pipeline.
   - **On Pull Request**: Runs tests and linters for the Next.js frontend (using Node.js) and the FastAPI backend (using `uv`).
   - **On Push to `main`**: Runs the tests/linters, and if successful, builds the multi-stage Docker image and pushes it directly to the **GitHub Container Registry (ghcr.io)**. 

### How to use the Docker Image

Since the image is automatically pushed to `ghcr.io`, anyone (or any deployment platform like Google Cloud Run or Render) can pull and run the application:

```bash
docker pull ghcr.io/gfgf96/dsl-learning:latest
docker run -p 8000:8000 ghcr.io/gfgf96/dsl-learning:latest
```