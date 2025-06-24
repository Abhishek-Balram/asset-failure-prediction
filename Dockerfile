FROM python:3.11-slim AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .                          

CMD ["gunicorn", "serve:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4"]