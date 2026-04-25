# =============================================================================
# Stage 1: Builder — install Python dependencies
# =============================================================================
FROM public.ecr.aws/docker/library/python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip

# Install Python deps into a prefix we can copy later
COPY <<EOF requirements-deploy.txt
boto3
fastapi
uvicorn
pymupdf
pypdf
Pillow
rapidfuzz
strands-agents
python-multipart
gunicorn
pydantic
EOF

RUN pip install --no-cache-dir --prefix=/install -r requirements-deploy.txt

# =============================================================================
# Stage 2: Runtime — lean image with only what we need
# =============================================================================
FROM public.ecr.aws/docker/library/python:3.11-slim AS runtime

# System deps (poppler for pdf2image, curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY local_agent/ .

# Explicitly copy KB markdown files (baked into image)
COPY local_agent/data/kb/ ./data/kb/

# Environment defaults
ENV STORAGE_BACKEND=aws \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "api:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "600"]
