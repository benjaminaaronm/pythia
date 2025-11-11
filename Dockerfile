FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Default command: run API
CMD ["uvicorn", "pythia.api:app", "--host", "0.0.0.0", "--port", "8000"]