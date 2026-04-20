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
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# Note: we use --system to install into the image's python environment
RUN uv sync --frozen --no-dev --system

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the absolute path to streamlit
CMD ["python", "-m", "streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
