# Dockerfile — Rastreamento e Análise de Tráfego
# Build: docker build -t traffic-analysis .
# Run:   docker run --rm traffic-analysis vehicles --source /videos/traffic.mp4 --save-video

FROM python:3.11-slim AS builder

WORKDIR /app

# Dependências de sistema para OpenCV e torch (CPU).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Estágio final: imagem menor, sem ferramentas de build.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . /app

# Cria diretórios de entrada/saída.
RUN mkdir -p /app/output /app/videos

# Variáveis de ambiente úteis para Docker.
ENV TRAFFIC_DEFAULT_SOURCE=/videos/traffic.mp4
ENV TRAFFIC_OUTPUT_DIR=/app/output
ENV TRAFFIC_NO_SHOW=true
ENV TRAFFIC_DEVICE=cpu

ENTRYPOINT ["python", "main.py"]
CMD ["vehicles", "--source", "/videos/traffic.mp4"]
