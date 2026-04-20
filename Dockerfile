# ── Stage: runtime ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (separate layer so Docker can cache it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Build the vector store at image build time.
# The PDF must be present in ./data/ before running docker build.
# The OPENAI_API_KEY must be passed as a build arg:
#   docker build --build-arg OPENAI_API_KEY=sk-... -t promtiorbot .
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN python ingest.py

# Railway injects the PORT env var; we default to 8000 locally
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
