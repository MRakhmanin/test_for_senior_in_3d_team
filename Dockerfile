FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime dependencies for Python wheels and medical imaging stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
RUN pip install --upgrade pip && \
    pip install \
      fastapi \
      uvicorn \
      python-multipart \
      numpy \
      SimpleITK \
      torch

COPY api /app/api
COPY model /app/model
COPY utils.py /app/utils.py

EXPOSE 8000

# Provide your trained checkpoint path via env at runtime.
# Example:
# docker run -e MODEL_CHECKPOINT=/app/checkpoints/best.pt -p 8000:8000 ...
ENV MODEL_CHECKPOINT=/app/checkpoints/best.pt
ENV DEVICE=auto

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

