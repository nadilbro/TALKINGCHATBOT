FROM python:3.11-slim

# Full root access — install system deps
RUN apt-get update && apt-get install -y \
    espeak \
    espeak-ng \
    ffmpeg \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Rhubarb
RUN wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip && \
    unzip -q rhubarb-lip-sync-1.13.0-linux.zip && \
    cp Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb /usr/local/bin/rhubarb && \
    chmod +x /usr/local/bin/rhubarb && \
    rm -rf rhubarb-lip-sync-1.13.0-linux.zip Rhubarb-Lip-Sync-1.13.0-Linux

# Copy project files
COPY . .

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:$PORT", "main:app", "--access-logfile", "-", "--error-logfile", "-"]
```