# Dockerfile tailored for Hugging Face Spaces
# - HF Spaces expect the service to listen on port 7860
# - HF runs as a non-root user; everything must be writable by that user
# - CORS is opened wide so the Vercel portal can hit /api/analyze from the browser

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# HF Spaces best practice: run as the "user" UID 1000 account
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
