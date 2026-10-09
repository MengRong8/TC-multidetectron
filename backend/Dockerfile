FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/requirements.intent.txt /app/backend/requirements.intent.txt
RUN pip install --upgrade pip \
 && pip install -r /app/backend/requirements.txt \
 && pip install -r /app/backend/requirements.intent.txt

EXPOSE 5000

CMD ["python", "backend/app.py", "--host", "0.0.0.0", "--port", "5000"]
