"""Lógica pura de análise de tráfego (sem I/O).

Esta camada não depende de OpenCV para exibição (cv2.imshow / waitKey),
não grava vídeo (cv2.VideoWriter) e não imprime no terminal (print).
Recebe frames e timestamps, processa detecção/rastreamento, contagem por
zonas ROI e estimação de velocidade, e expõe métodos para consultar
métricas e gerar o dicionário de relatório.

Uso típico:
    analyzer = TrafficAnalyzer(cfg)
    for frame, ts in video_frames():
        result = analyzer.process_frame(frame, ts)
    report = analyzer.get_report()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    AppConfig,
    MODE_LABELS,
    classes_for_mode,
    load_speed,
    load_zones,
    mode_uses_speed,
    resolve_path,
)
from .counter import ObjectCounter
from .detector import Detector, Detection
from .speed import SpeedEstimator
from .visualization import Visualizer
from .zones import Zone, build_zones


@dataclass
class FrameResult:
    """Resultado do processamento de um único frame."""

    frame_index: int
    timestamp: float
    detections: list[Detection]
    count_events: list
    processing_time_s: float


@dataclass
class AnalyzerStats:
    """Estatísticas acumuladas durante a execução."""

    frames_processed: int = 0
    total_processing_s: float = 0.0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


class TrafficAnalyzer:
    """Executa a análise de tráfego de forma pura (sem I/O de vídeo/janela)."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.classes = classes_for_mode(cfg.mode)

        model_path = resolve_path(cfg.model_path)
        self.detector = Detector(
            model_path=str(model_path),
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            device=cfg.device,
            tracker=cfg.tracker,
            classes=self.classes,
        )

        self.zones: list[Zone] = build_zones(load_zones(cfg.zones_file))
        self.counter = ObjectCounter(self.zones)

        speed_cfg = load_speed(cfg.speed_file)
        if not mode_uses_speed(cfg.mode):
            speed_cfg.enabled = False
        self.speed = SpeedEstimator(speed_cfg)

        self.visualizer = Visualizer(show_speed=self.speed.enabled)

        self.stats = AnalyzerStats()
        self._wall_start: Optional[float] = None
        self._video_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    def process_frame(
        self, frame, timestamp: float, frame_index: int
    ) -> FrameResult:
        """Processa um único frame: detecta, conta, estima velocidade.

        Não desenha na tela nem grava vídeo. Retorna um FrameResult com
        detecções, eventos de contagem e tempo de processamento.
        """
        t0 = time.perf_counter()

        detections = self.detector.track(frame)
        count_events = self.counter.update(detections, timestamp)
        self.speed.update(detections, timestamp)

        dt = time.perf_counter() - t0
        self.stats.frames_processed += 1
        self.stats.total_processing_s += dt
        if self.stats.started_at is None:
            self.stats.started_at = t0
        self.stats.ended_at = time.perf_counter()

        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            detections=detections,
            count_events=count_events,
            processing_time_s=dt,
        )

    # ------------------------------------------------------------------ #
    def annotate_frame(
        self,
        frame,
        result: FrameResult,
        fps: float,
        frame_info: str = "",
    ):
        """Desenha anotações no frame (boxes, IDs, ROI, HUD).

        Retorna o frame anotado. Não exibe janela nem grava vídeo.
        """
        return self.visualizer.draw(
            frame,
            result.detections,
            self.zones,
            self.counter,
            self.speed if self.speed.enabled else None,
            MODE_LABELS[self.cfg.mode],
            fps,
            frame_info,
        )

    # ------------------------------------------------------------------ #
    def get_report(self, output_video_path: Optional[str] = None) -> dict:
        """Gera o dicionário de relatório final (sem salvar em disco)."""
        elapsed = 0.0
        if self.stats.started_at is not None and self.stats.ended_at is not None:
            elapsed = self.stats.ended_at - self.stats.started_at

        processed = self.stats.frames_processed
        return {
            "mode": self.cfg.mode,
            "mode_label": MODE_LABELS[self.cfg.mode],
            "source": self.cfg.source,
            "tracker": self.cfg.tracker,
            "skip_frames": self.cfg.skip_frames,
            "frames_processed": processed,
            "elapsed_s": round(elapsed, 2),
            "processing_fps": round(
                processed / max(1e-6, elapsed), 2
            ),
            "counting": self.counter.summary(),
            "speed": self.speed.stats() if self.speed.enabled else {"enabled": False},
            "speed_calibration": {
                "enabled": self.speed.enabled,
                "real_width_m": self.speed.cfg.real_width_m,
                "real_height_m": self.speed.cfg.real_height_m,
                "pixels_per_meter": self.speed.cfg.pixels_per_meter,
            },
            "output_video": output_video_path,
            "timestamp": self._iso_now(),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _iso_now() -> str:
        from datetime import datetime

        return datetime.now().isoformat(timespec="seconds")
