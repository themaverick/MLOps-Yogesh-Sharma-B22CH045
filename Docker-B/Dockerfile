FROM python:3.11-slim

# Install system deps if needed (e.g. for image libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY train.py evaluate.py ./

# Default: run training (override when running container)
CMD ["python", "train.py"]