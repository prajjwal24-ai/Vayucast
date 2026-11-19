FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by LightGBM on Debian slim images
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install ONLY runtime dependencies (no mlflow, evidently, dvc, or pytest)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

# Copy config and latest promoted artifacts/cache for zero-config startup
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY data/processed/ ./data/processed/
COPY reports/ ./reports/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "vayucast.api:app", "--host", "0.0.0.0", "--port", "8000"]