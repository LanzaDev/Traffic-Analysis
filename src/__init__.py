"""Pacote modular de Rastreamento e Análise de Tráfego.

Módulos:
    config          - Configurações, dataclasses e carregamento de JSON.
    video_source    - Captura de vídeo (arquivo, webcam, RTSP/HTTP) + frame skipping.
    detector        - Detecção e rastreamento YOLO (Ultralytics) + ByteTrack/BoT-SORT.
    zones           - Regiões de interesse (polígonos e linhas) e teste de travessia.
    counter         - Contagem de objetos por IDs únicos (sets) em cada zona.
    speed           - Estimação de velocidade (Bird's Eye View / perspectiva).
    visualization   - Interface visual (boxes, IDs, ROI, contadores, velocidade).
    pipeline        - Orquestração dos modos de execução.
"""

__version__ = "1.0.0"
