FROM python:3.11-slim

# Install ffmpeg and git-lfs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose Render port
EXPOSE 10000

WORKDIR /app/backend

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]