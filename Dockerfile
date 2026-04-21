# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies (needed for FAISS and uv)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies globally into the system Python
RUN uv pip install --system -e .

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the absolute path to streamlit
CMD ["python", "-m", "streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
