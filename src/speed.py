"""Estimação de velocidade de veículos em km/h.

Estratégia principal: transformação de perspectiva (cv2.getPerspectiveTransform)
para uma vista "Bird's Eye View" métrica. O ponto de contato do veículo com o
chão (bottom center da bbox) é projetado no plano do chão em metros; o
deslocamento entre timestamps é convertido de m/s para km/h (× 3.6).

Um filtro de Kalman unidimensional reduz ruído nas medições instantâneas e
produz uma curva de km/h mais estável e responsiva, evitando picos espúrios
causados por trocas de ID ou detecções instáveis.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .config import SpeedConfig, VEHICLE_CLASSES
from .detector import Detection


class KalmanSpeedFilter:
    """Filtro de Kalman unidimensional para suavização de velocidade (km/h).

    Estado: velocidade (km/h).
    Medição: velocidade instantânea calculada a partir do deslocamento.
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        initial_uncertainty: float = 1.0,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.x: Optional[float] = None  # estimativa de velocidade (km/h)
        self.p: float = initial_uncertainty  # incerteza da estimativa

    def predict(self) -> None:
        """Passo de predição: velocidade tende a permanecer constante."""
        # F = 1 (modelo de velocidade constante)
        # x = F * x  (x inalterado)
        # P = F * P * F' + Q
        if self.x is not None:
            self.p = self.p + self.process_noise

    def update(self, measurement: float) -> float:
        """Passo de atualização com uma nova medição instantânea (km/h).

        Retorna a velocidade filtrada (km/h).
        """
        if self.x is None:
            # Primeira medição: inicializa diretamente.
            self.x = measurement
            self.p = self.measurement_noise
            return self.x

        # Predição.
        self.predict()

        # Ganho de Kalman: K = P * H' / (H * P * H' + R), com H = 1.
        k = self.p / (self.p + self.measurement_noise)

        # Atualização do estado.
        self.x = self.x + k * (measurement - self.x)

        # Atualização da incerteza.
        self.p = (1.0 - k) * self.p

        return self.x

    def get(self) -> Optional[float]:
        """Retorna a última velocidade filtrada (km/h)."""
        return self.x

    def reset(self) -> None:
        """Reseta o filtro."""
        self.x = None
        self.p = 1.0


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
        self.filters: dict[int, KalmanSpeedFilter] = {}
        # Contagem de frames consecutivos e soma de confiança por ID.
        self._consecutive_frames: dict[int, int] = {}
        self._confidence_sum: dict[int, float] = {}

        if self.enabled:
            src = np.asarray(cfg.source_points, dtype=np.float32)
            w = cfg.real_width_m * self.ppm
            h = cfg.real_height_m * self.ppm
            dst = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            self.M = cv2.getPerspectiveTransform(src, dst)

    def _get_filter(self, tid: int) -> KalmanSpeedFilter:
        """Retorna o filtro de Kalman do veículo, criando se necessário."""
        return self.filters.setdefault(
            tid,
            KalmanSpeedFilter(
                process_noise=self.cfg.kalman_process_noise,
                measurement_noise=self.cfg.kalman_measurement_noise,
            ),
        )

    def is_confirmed(self, tid: Optional[int]) -> bool:
        """Retorna True se o ID tem confiança suficiente para exibir velocidade.

        Critério: pelo menos min_detection_frames frames consecutivos E
        confiança média >= min_confidence.
        """
        if tid is None:
            return False
        frames = self._consecutive_frames.get(tid, 0)
        if frames < self.cfg.min_detection_frames:
            return False
        total_conf = self._confidence_sum.get(tid, 0.0)
        if frames > 0 and (total_conf / frames) < self.cfg.min_confidence:
            return False
        return True

    def to_world(self, pt: tuple[float, float]) -> tuple[float, float]:
        """Projeta um ponto da imagem para coordenadas métricas (metros)."""
        assert self.M is not None, "Matriz de perspectiva não inicializada"
        p = np.asarray([[[pt[0], pt[1]]]], dtype=np.float32)
        t = cv2.perspectiveTransform(p, self.M)[0][0]
        return float(t[0]) / self.ppm, float(t[1]) / self.ppm

    def update(self, detections: list[Detection], timestamp: float) -> None:
        if not self.enabled:
            return
        detected_tids: set[int] = set()
        for d in detections:
            if not d.is_vehicle or d.track_id is None:
                continue
            tid = d.track_id
            detected_tids.add(tid)
            wx, wy = self.to_world(d.bottom_center)
            dq = self.history.setdefault(tid, deque(maxlen=20))
            dq.append((timestamp, wx, wy))
            self.last_seen[tid] = timestamp

            # Atualiza contagem de frames consecutivos e soma de confiança.
            self._consecutive_frames[tid] = self._consecutive_frames.get(tid, 0) + 1
            self._confidence_sum[tid] = (
                self._confidence_sum.get(tid, 0.0) + float(d.confidence)
            )

            self._compute(tid, dq)

        # Reseta contagem de frames consecutivos para IDs que não foram detectados neste frame.
        for tid in list(self._consecutive_frames.keys()):
            if tid not in detected_tids:
                self._consecutive_frames[tid] = 0

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

        # Filtro de Kalman substitui a EMA.
        kf = self._get_filter(tid)
        filtered = kf.update(instant)
        self.speeds[tid] = filtered

    def _cleanup(self, timestamp: float, ttl: float = 3.0) -> None:
        stale = [tid for tid, t in self.last_seen.items() if timestamp - t > ttl]
        for tid in stale:
            self.history.pop(tid, None)
            self.speeds.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.filters.pop(tid, None)
            self._consecutive_frames.pop(tid, None)
            self._confidence_sum.pop(tid, None)

    def get(self, tid: Optional[int]) -> Optional[float]:
        if tid is None:
            return None
        if not self.is_confirmed(tid):
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
