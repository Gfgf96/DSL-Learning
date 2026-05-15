# ---------------------------------------------------------
# STAGE 1: Frontend Builder
# ---------------------------------------------------------
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy package files and install
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------
# STAGE 2: Production Runner
# ---------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install python dependencies using uv (production only, strictly matching lockfile)
RUN uv sync --frozen --no-dev

# Copy application source code
COPY . .

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/out /app/frontend/out

# Ensure models directory exists with read/write permissions
# so the app can download model weights dynamically at startup
RUN mkdir -p models && chmod -R 777 models

# Expose FastAPI port
EXPOSE 8000

# Start the application using the uv-managed environment
# main.py handles model downloading and launching uvicorn
CMD ["uv", "run", "main.py"]
