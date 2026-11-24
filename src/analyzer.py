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

        # Buffer para processamento em lote (batch inference).
        self._pending_frames: list = []
        self._pending_timestamps: list[float] = []
        self._pending_indices: list[int] = []

    # ------------------------------------------------------------------ #
    def process_frame(
        self, frame, timestamp: float, frame_index: int
    ) -> FrameResult | None:
        """Processa um único frame: detecta, conta, estima velocidade.

        Quando batch_size > 1, o frame é acumulado e processado em lote.
        Retorna um FrameResult quando o frame é processado imediatamente
        (batch_size == 1) ou quando completa um lote; caso contrário, retorna None
        e o caller deve chamar flush() no final para processar lotes incompletos.
        Não desenha na tela nem grava vídeo.
        """
        t0 = time.perf_counter()

        # Sem batch: processamento direto (comportamento original).
        if self.cfg.batch_size <= 1:
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

        # Com batch: acumula e processa quando o lote estiver completo.
        self._pending_frames.append(frame)
        self._pending_timestamps.append(timestamp)
        self._pending_indices.append(frame_index)

        if len(self._pending_frames) >= self.cfg.batch_size:
            return self._flush_batch(t0)

        return None

    # ------------------------------------------------------------------ #
    def _flush_batch(self, batch_start_t: float) -> FrameResult | None:
        """Processa o lote acumulado via inferência em batch."""
        if not self._pending_frames:
            return None

        frames = self._pending_frames
        timestamps = self._pending_timestamps
        indices = self._pending_indices

        # Inferência em lote (batch): todos os frames de uma vez.
        detections_per_frame = self.detector.track_batch(frames)

        # Processa cada frame individualmente para contagem e velocidade.
        results: list[FrameResult] = []
        t_per_frame = (time.perf_counter() - batch_start_t) / max(len(frames), 1)

        for i, (frame, timestamp, frame_idx, detections) in enumerate(
            zip(frames, timestamps, indices, detections_per_frame)
        ):
            count_events = self.counter.update(detections, timestamp)
            self.speed.update(detections, timestamp)

            self.stats.frames_processed += 1
            self.stats.total_processing_s += t_per_frame

            results.append(
                FrameResult(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    detections=detections,
                    count_events=count_events,
                    processing_time_s=t_per_frame,
                )
            )

        if self.stats.started_at is None:
            self.stats.started_at = batch_start_t
        self.stats.ended_at = time.perf_counter()

        # Limpa o buffer.
        self._pending_frames = []
        self._pending_timestamps = []
        self._pending_indices = []

        # Retorna o último resultado do lote (o caller exibe o frame mais recente).
        return results[-1] if results else None

    # ------------------------------------------------------------------ #
    def flush(self) -> list[FrameResult]:
        """Processa qualquer lote incompleto restante no buffer.

        Deve ser chamado no final da execução quando batch_size > 1.
        Retorna a lista de FrameResult dos frames processados no flush.
        """
        if not self._pending_frames:
            return []

        t0 = time.perf_counter()
        result = self._flush_batch(t0)
        return [result] if result else []

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
