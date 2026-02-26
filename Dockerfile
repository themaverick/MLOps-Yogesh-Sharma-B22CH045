# Assignment 3: Training environment - Goodreads genre classification
# Build: docker build -t goodreads-train .
# Run:   docker run --env-file .env -v $(pwd)/results:/app/results goodreads-train

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Task 2.6: Verify Python and required libraries
RUN python -c "import torch; import transformers; import sklearn; print('Python and libraries OK')"

# Copy source
COPY src/ src/
COPY ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb .

# Default: run training (CPU). Override with docker run ... goodreads-train python -m src.eval ...
ENV PYTHONPATH=/app
CMD ["python", "-m", "src.train", "--epochs", "2", "--batch-size", "4", "--push-to-hub"]
