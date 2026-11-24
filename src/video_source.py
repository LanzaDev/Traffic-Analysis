"""Fonte de vídeo unificada: arquivos locais, webcam e streams IP/RTSP/HTTP.

Recursos:
    - Detecção automática do tipo de fonte (índice, URL ou caminho).
    - Frame skipping para manter fluidez em tempo real (CCTV).
    - Timestamps por frame (baseado no FPS para arquivos; relógio para ao vivo).
    - Reconexão automática para streams ao vivo que caem.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoSource:
    """Abstrai cv2.VideoCapture para arquivo/webcam/stream com frame skipping."""

    def __init__(
        self,
        source: str,
        skip_frames: int = 1,
        max_reconnects: int = 5,
        reconnect_delay: float = 2.0,
    ) -> None:
        self.raw = str(source)
        self.skip_frames = max(1, int(skip_frames))
        self.max_reconnects = max_reconnects
        self.reconnect_delay = reconnect_delay

        self.kind = self._classify(self.raw)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0

    @staticmethod
    def _classify(source: str) -> str:
        if source.isdigit():
            return "webcam"
        low = source.lower()
        if low.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return "stream"
        return "file"

    @property
    def is_live(self) -> bool:
        return self.kind in ("webcam", "stream")

    def open(self) -> "VideoSource":
        if self.kind == "webcam":
            # DirectShow (CAP_DSHOW) é mais confiável que o padrão MSMF no Windows
            # para captura de webcam (evita falhas de grab/OnReadSample).
            self.cap = cv2.VideoCapture(int(self.raw), cv2.CAP_DSHOW)
        elif self.kind == "stream":
            self.cap = cv2.VideoCapture(self.raw, cv2.CAP_FFMPEG)
            # Buffer mínimo para reduzir latência em tempo real.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            p = Path(self.raw)
            if not p.is_absolute():
                from .config import PROJECT_ROOT

                p = PROJECT_ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"Vídeo não encontrado: {p}")
            self.cap = cv2.VideoCapture(str(p))

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte: {self.raw}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps <= 0 or self.fps > 240:
            self.fps = 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Gera (índice_absoluto, timestamp_s, frame) já aplicando frame skipping.

        Todos os frames são lidos com read() (necessário para avançar no arquivo
        e para manter o stream ao vivo em dia); apenas 1 a cada `skip_frames` é
        entregue. Usa-se read() em vez de grab()/retrieve() porque o backend
        MSMF do Windows falha no grab() separado (erro OnReadSample).
        """
        if self.cap is None:
            self.open()

        assert self.cap is not None, "VideoCapture não foi inicializado"

        abs_idx = 0
        reconnects = 0
        start = time.perf_counter()

        while True:
            ret, frame = self.cap.read()
            abs_idx += 1

            # Trata falha de leitura antes do skipping, para que uma fonte morta
            # acione a reconexão/encerramento mesmo em frames ignorados.
            if not ret or frame is None:
                if self.is_live and reconnects < self.max_reconnects:
                    reconnects += 1
                    print(
                        f"[fonte] falha na leitura; reconectando "
                        f"({reconnects}/{self.max_reconnects})..."
                    )
                    time.sleep(self.reconnect_delay)
                    assert self.cap is not None
                    self.cap.release()
                    self.open()
                    start = time.perf_counter()
                    continue
                break

            # Frame skipping: entrega apenas 1 a cada N frames lidos.
            if (abs_idx - 1) % self.skip_frames != 0:
                continue

            if self.is_live:
                timestamp = time.perf_counter() - start
            else:
                timestamp = (abs_idx - 1) / self.fps

            yield abs_idx - 1, timestamp, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
