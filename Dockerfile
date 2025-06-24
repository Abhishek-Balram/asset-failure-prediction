FROM python:3.11-slim AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .                          # code + artifacts
ENV PORT=8080 PYTHONUNBUFFERED=1

# Using 1 worker is OK for 256-512 MB free instances.
CMD ["gunicorn", "serve:app", "--bind", "0.0.0.0:${PORT}", "--workers", "1", "--threads", "4"]