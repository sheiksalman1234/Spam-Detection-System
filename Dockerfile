FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set cache directories for model downloads inside the container
ENV XDG_CACHE_HOME=/app/.cache
ENV HF_HOME=/app/.cache/huggingface

# Pre-download models at build time so Render/Railway cold-starts are instant
# Whisper tiny model (~39 MB)
RUN python -c "import whisper; whisper.load_model('tiny')"

# HuggingFace spam-call-detector model
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('Salmansheik/spam-call-detector'); AutoModelForSequenceClassification.from_pretrained('Salmansheik/spam-call-detector')"

COPY . .

WORKDIR /app/backend

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
