FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY translate.py .
COPY input.txt .
COPY reference.txt .

CMD ["python", "translate.py"]
