# ── Build stage: install dependencies ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder (keeps final image lean)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source and train the model (synthetic data, no preprocessing needed)
COPY src/      ./src/

RUN python src/train_model.py

# Don't run as root — basic security practice
RUN useradd -m apiuser
USER apiuser

EXPOSE 7860

# Health check so Docker/Kubernetes knows when the container is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
