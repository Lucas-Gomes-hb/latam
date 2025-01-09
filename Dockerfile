FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-test.txt

COPY . .

ENV ENVIRONMENT=production

EXPOSE 8000

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]