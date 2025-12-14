# Chatbot BKSI - Multi-stage Dockerfile
# Using uv for fast dependency management

# ============================================
# Stage 1: Base image with uv
# ============================================
FROM python:3.11-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# ============================================
# Stage 2: Dependencies
# ============================================
FROM base AS dependencies

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies (without dev dependencies)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ============================================
# Stage 3: Application
# ============================================
FROM base AS app

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Create necessary directories
RUN mkdir -p data/raw data/processed lancedb_data logs .cache

# ============================================
# Stage 4: API Service
# ============================================
FROM app AS api

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 5: Gradio Service
# ============================================
FROM app AS gradio

EXPOSE 7860

CMD ["python", "-m", "ui.gradio_app"]

# ============================================
# Stage 6: Streamlit Service
# ============================================
FROM app AS streamlit

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

# ============================================
# Stage 7: CLI (default)
# ============================================
FROM app AS cli

ENTRYPOINT ["python", "-m", "scripts.cli"]
CMD ["--help"]
