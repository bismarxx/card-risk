# syntax=docker/dockerfile:1

FROM python:3.11-slim

# --- System setup -----------------------------------------------------
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Dependencies -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code -----------------------------------------------------
COPY . .

# Train the model at build time so the image is self-contained and ready
# to serve predictions as soon as the container starts.
# (Skips re-downloading if a model artifact was already committed/copied in.)
RUN python train.py

# --- Runtime configuration -----------------------------------------------
# Render injects $PORT at runtime; Hugging Face Spaces defaults to 7860.
# We default to 8501 (Streamlit's standard port) and let the platform override it.
ENV PORT=8501
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Use sh -c so ${PORT} is resolved at container start (works on Render, HF Spaces, etc.)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
