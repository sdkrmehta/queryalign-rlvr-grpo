# Multi-stage Dockerfile for SQL-GRPO Serving
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels .

# Runtime Stage
FROM python:3.11-slim AS runner

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install built wheel and serving dependencies
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* fastapi uvicorn streamlit pydantic pyyaml sqlglot

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/outputs && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser configs ./configs

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "sql_grpo.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
