# HybridMind - Vector + Graph Native Database
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

# Run with uvicorn
# Note: Health check is defined in docker-compose.yml to avoid conflicts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
