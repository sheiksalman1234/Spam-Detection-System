FROM python:3.11-slim

# Install system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better build caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Move into backend
WORKDIR /app/backend

# Expose Render's port
EXPOSE 10000

# Start FastAPI
CMD ["sh", "-c", "gunicorn --workers 1 --bind 0.0.0.0:${PORT:-10000} app:app"]