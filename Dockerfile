FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    espeak \
    espeak-ng \
    ffmpeg \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.12.0/rhubarb-lip-sync-1.12.0-linux.zip && \
    unzip -q rhubarb-lip-sync-1.12.0-linux.zip && \
    cp Rhubarb-Lip-Sync-1.12.0-Linux/rhubarb /usr/local/bin/rhubarb && \
    cp -r Rhubarb-Lip-Sync-1.12.0-Linux/res /usr/local/bin/res && \
    chmod +x /usr/local/bin/rhubarb && \
    rm -rf rhubarb-lip-sync-1.12.0-linux.zip Rhubarb-Lip-Sync-1.12.0-Linux

COPY . .

CMD gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT main:app --access-logfile - --error-logfile -