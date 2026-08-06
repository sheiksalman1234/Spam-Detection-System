FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Set cache directories for model downloads inside the container
ENV XDG_CACHE_HOME=/app/.cache
ENV HF_HOME=/app/.cache/huggingface

# Models will be downloaded on first request (lazy loading)

COPY . .

RUN mkdir -p /app/backend/uploads

WORKDIR /app/backend

ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
