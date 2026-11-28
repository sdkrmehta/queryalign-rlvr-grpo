# Multi-stage Dockerfile for SQL-GRPO Local Inference Serving
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels .

# Runtime Stage
FROM python:3.11-slim-bookworm AS runner

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install built package and local CPU inference + serving dependencies
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    /wheels/* \
    "torch>=2.2.0" \
    "transformers>=4.48.0" \
    "peft>=0.18.1" \
    "accelerate>=0.30.0" \
    "safetensors>=0.4.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.34.0" \
    "streamlit>=1.41.0" \
    "pydantic>=2.8.0" \
    "pydantic-settings>=2.4.0" \
    "pyyaml>=6.0" \
    "sqlglot>=26.0.0" \
    "psutil>=6.0.0" \
    "pandas>=2.0.0" \
    "numpy>=1.26.0"

# Create non-root user and writable cache/output directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/artifacts /app/outputs /app/.cache/huggingface && \
    chown -R appuser:appuser /app

ENV HF_HOME=/app/.cache/huggingface

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser configs ./configs

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "sql_grpo.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
