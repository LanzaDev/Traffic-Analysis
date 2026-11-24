"""Estimação de velocidade de veículos em km/h.

Estratégia principal: transformação de perspectiva (cv2.getPerspectiveTransform)
para uma vista "Bird's Eye View" métrica. O ponto de contato do veículo com o
chão (bottom center da bbox) é projetado no plano do chão em metros; o
deslocamento entre timestamps é convertido de m/s para km/h (× 3.6).

Uma suavização exponencial (EMA) reduz ruído, e um teto de sanidade evita
picos espúrios causados por trocas de ID ou detecções instáveis.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .config import SpeedConfig, VEHICLE_CLASSES
from .detector import Detection


class SpeedEstimator:
    """Calcula e suaviza a velocidade (km/h) por ID de veículo rastreado."""

    def __init__(self, cfg: SpeedConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.enabled) and len(cfg.source_points) == 4
        self.history: dict[int, deque] = {}
        self.speeds: dict[int, float] = {}
        self.last_seen: dict[int, float] = {}
        self.M: Optional[np.ndarray] = None
        self.ppm: float = cfg.pixels_per_meter

        if self.enabled:
            src = np.asarray(cfg.source_points, dtype=np.float32)
            w = cfg.real_width_m * self.ppm
            h = cfg.real_height_m * self.ppm
            dst = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            self.M = cv2.getPerspectiveTransform(src, dst)

    def to_world(self, pt: tuple[float, float]) -> tuple[float, float]:
        """Projeta um ponto da imagem para coordenadas métricas (metros)."""
        assert self.M is not None, "Matriz de perspectiva não inicializada"
        p = np.asarray([[[pt[0], pt[1]]]], dtype=np.float32)
        t = cv2.perspectiveTransform(p, self.M)[0][0]
        return float(t[0]) / self.ppm, float(t[1]) / self.ppm

    def update(self, detections: list[Detection], timestamp: float) -> None:
        if not self.enabled:
            return
        for d in detections:
            if not d.is_vehicle or d.track_id is None:
                continue
            tid = d.track_id
            wx, wy = self.to_world(d.bottom_center)
            dq = self.history.setdefault(tid, deque(maxlen=20))
            dq.append((timestamp, wx, wy))
            self.last_seen[tid] = timestamp
            self._compute(tid, dq)
        self._cleanup(timestamp)

    def _compute(self, tid: int, dq: deque) -> None:
        if len(dq) < 2:
            return
        t1, x1, y1 = dq[-1]
        # Usa a amostra mais antiga dentro da janela temporal como referência.
        ref = dq[0]
        for sample in dq:
            if t1 - sample[0] <= self.cfg.window_s:
                ref = sample
                break
        t0, x0, y0 = ref
        dt = t1 - t0
        if dt <= 1e-3:
            return
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < self.cfg.min_displacement_m:
            instant = 0.0
        else:
            instant = (dist / dt) * 3.6  # m/s -> km/h
        instant = min(instant, self.cfg.max_speed_kmh)

        prev = self.speeds.get(tid)
        a = self.cfg.smoothing_alpha
        self.speeds[tid] = instant if prev is None else a * instant + (1 - a) * prev

    def _cleanup(self, timestamp: float, ttl: float = 3.0) -> None:
        stale = [tid for tid, t in self.last_seen.items() if timestamp - t > ttl]
        for tid in stale:
            self.history.pop(tid, None)
            self.speeds.pop(tid, None)
            self.last_seen.pop(tid, None)

    def get(self, tid: Optional[int]) -> Optional[float]:
        if tid is None:
            return None
        return self.speeds.get(tid)

    def stats(self) -> dict:
        """Estatísticas agregadas das velocidades atuais (para relatório)."""
        values = list(self.speeds.values())
        if not values:
            return {"vehicles_with_speed": 0}
        return {
            "vehicles_with_speed": len(values),
            "avg_kmh": round(sum(values) / len(values), 1),
            "max_kmh": round(max(values), 1),
        }
