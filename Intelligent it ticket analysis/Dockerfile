FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt requirements-dashboard.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt -r requirements-dashboard.txt

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
