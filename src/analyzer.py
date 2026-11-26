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

import numpy as np

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
from .reid import ReIDDetector
from .speed import SpeedEstimator
from .stopped_vehicle import IncidentDetector, StoppedVehicleDetector, StoppedVehicleEvent
from .visualization import Visualizer
from .zones import Zone, build_zones
from .metrics import evaluate_detections, _load_ground_truth, report_to_dict


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

        model_path = resolve_path(cfg.engine_path if cfg.engine_path else cfg.model_path)
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
        # Sobrescreve parâmetros de confiança vindos do AppConfig (CLI/env).
        speed_cfg.min_detection_frames = cfg.min_detection_frames
        speed_cfg.min_confidence = cfg.min_confidence
        self.speed = SpeedEstimator(speed_cfg)

        self.visualizer = Visualizer(show_speed=self.speed.enabled)

        self.stopped_detector = StoppedVehicleDetector(
            min_displacement_m=0.05,
            min_stopped_frames=cfg.min_detection_frames,
            min_stopped_s=cfg.min_stopped_s,
        )
        self.incident_detector = IncidentDetector(self.zones)
        self.stopped_events: list[StoppedVehicleEvent] = []
        self.incident_events: list[IncidentEvent] = []

        # Re-ID detector (reidentificação de objetos que somem e reaparecem)
        self.reid_detector = ReIDDetector(
            max_distance_px=cfg.reid_max_distance_m,
            max_time_s=cfg.reid_max_time_s,
            color_tolerance=cfg.reid_color_tolerance,
        )
        self.reid_enabled = cfg.reid_enabled
        # Histórico de objetos vistos no frame anterior: tid -> (pos_px, color_hist, timestamp, class_name)
        self._prev_frame_objects: dict[int, tuple[tuple[float, float], Optional[np.ndarray], float, str]] = {}

        # Avaliação (modo --eval)
        self.eval_mode = cfg.eval_mode
        self.gt_path = cfg.gt_path
        self.gt_by_frame: dict[int, list[tuple[np.ndarray, int]]] = {}
        self.detections_by_frame: dict[int, list[tuple[np.ndarray, int, float]]] = {}
        if self.eval_mode and self.gt_path:
            self.gt_by_frame = _load_ground_truth(self.gt_path)

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

            # Extrair cor de cada detecção para Re-ID
            if self.reid_enabled:
                for d in detections:
                    if d.track_id is not None:
                        d.color_hist = Detector.extract_color_hist(frame, d.bbox)

            # --- Re-ID: tentar reidentificar objetos que sumiram ---
            if self.reid_enabled:
                detections = self._apply_reid(detections, frame, timestamp)

            count_events = self.counter.update(detections, timestamp)
            self.speed.update(detections, timestamp)

            # Detecção de veículos parados e incidentes.
            stopped_events = self.stopped_detector.update(detections, timestamp, self.speed if self.speed.enabled else None)
            incident_events = self.incident_detector.update(
                stopped_events, detections, self.counter, timestamp
            )
            self.stopped_events.extend(stopped_events)
            self.incident_events.extend(incident_events)

            # Coletar detecções para avaliação (modo --eval).
            if self.eval_mode:
                det_list = [
                    (d.bbox, d.class_id, d.confidence)
                    for d in detections
                ]
                self.detections_by_frame[frame_index] = det_list

            # Registrar objetos vistos neste frame para Re-ID no próximo frame
            self._update_prev_frame_objects(detections, timestamp)

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
            # Extrair cor de cada detecção para Re-ID
            if self.reid_enabled:
                for d in detections:
                    if d.track_id is not None:
                        d.color_hist = Detector.extract_color_hist(frame, d.bbox)

            # --- Re-ID: tentar reidentificar objetos que sumiram ---
            if self.reid_enabled:
                detections = self._apply_reid(detections, frame, timestamp)

            count_events = self.counter.update(detections, timestamp)
            self.speed.update(detections, timestamp)

            # Detecção de veículos parados e incidentes.
            stopped_events = self.stopped_detector.update(detections, timestamp, self.speed if self.speed.enabled else None)
            incident_events = self.incident_detector.update(
                stopped_events, detections, self.counter, timestamp
            )
            self.stopped_events.extend(stopped_events)
            self.incident_events.extend(incident_events)

            # Coletar detecções para avaliação (modo --eval).
            if self.eval_mode:
                det_list = [
                    (d.bbox, d.class_id, d.confidence)
                    for d in detections
                ]
                self.detections_by_frame[frame_idx] = det_list

            # Registrar objetos vistos neste frame para Re-ID no próximo frame
            self._update_prev_frame_objects(detections, timestamp)

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
    def _apply_reid(
        self,
        detections: list[Detection],
        frame,
        timestamp: float,
    ) -> list[Detection]:
        """Tenta reidentificar objetos que sumiram e reapareceram com novo ID.

        Para cada detecção com track_id novo (não visto antes), verifica se
        corresponde a um objeto perdido. Se houver match, substitui o track_id
        pelo antigo para evitar contagem duplicada.

        Usa distância em pixels (não metros) para comparação, já que a
        transformação de perspectiva só está disponível no SpeedEstimator.
        """
        current_ids = {d.track_id for d in detections if d.track_id is not None}

        # Converter max_distance_m para pixels baseado na resolução da imagem
        frame_h, frame_w = frame.shape[:2]
        # Assumir que max_distance_m em metros corresponde aproximadamente a
        # uma fração da largura da imagem (ex: 15m em uma pista de ~30m de largura
        # = 50% da largura da imagem)
        max_dist_px = self.cfg.reid_max_distance_m / 30.0 * frame_w

        for d in detections:
            if d.track_id is None:
                continue

            # Se o ID já foi visto neste frame, não precisa de re-ID
            if d.track_id in self._prev_frame_objects:
                continue

            # Extrair cor da bbox para comparação
            color_hist = Detector.extract_color_hist(frame, d.bbox)
            print(f"[REID DEBUG] Novo ID={d.track_id}, classe={d.class_name}, pos={d.bottom_center}, tem_histograma={color_hist is not None}, tam_hist={len(color_hist) if color_hist is not None else 0}")

            # Mostrar candidatos perdidos
            print(f"[REID DEBUG]   Candidatos perdidos disponíveis:")
            for tid, obj in self.reid_detector.lost_objects.items():
                if obj.class_name != d.class_name:
                    print(f"[REID DEBUG]     ID={tid}, classe={obj.class_name} (classe diferente, ignorado)")
                    continue
                if obj.is_expired(timestamp, self.reid_detector.max_time_s):
                    print(f"[REID DEBUG]     ID={tid}, classe={obj.class_name} (expirado, ignorado)")
                    continue
                dx = d.bottom_center[0] - obj.last_pos_px[0]
                dy = d.bottom_center[1] - obj.last_pos_px[1]
                dist_px = (dx * dx + dy * dy) ** 0.5
                cd = self.reid_detector._color_distance(obj.color_hist, color_hist) if obj.color_hist is not None and color_hist is not None else 1.0
                print(f"[REID DEBUG]     ID={tid}, classe={obj.class_name}, dist={dist_px:.1f}px, cor_dist={cd:.3f}, dentro_limite={dist_px <= max_dist_px}, cor_ok={cd <= self.reid_detector.color_tolerance}")

            # Tentar encontrar correspondência com objeto perdido
            old_tid = self.reid_detector.find_match(
                new_track_id=d.track_id,
                class_name=d.class_name,
                pos_px=d.bottom_center,  # usar coordenadas de pixels
                color_hist=color_hist,
                timestamp=timestamp,
                max_dist_px=max_dist_px,
            )

            if old_tid is not None:
                # Re-ID bem-sucedido: substituir ID novo pelo antigo
                new_tid = d.track_id
                d.track_id = old_tid
                self.reid_detector.match_and_update(new_track_id=new_tid, old_track_id=old_tid)
                print(f"[REID DEBUG]   MATCH ENCONTRADO: novo ID={new_tid} -> antigo ID={old_tid}")
            else:
                print(f"[REID DEBUG]   Nenhum match encontrado para ID={d.track_id}")

        # Registrar objetos que sumiram (estavam no frame anterior mas não estão neste)
        current_tids = {d.track_id for d in detections if d.track_id is not None}
        for prev_tid in list(self._prev_frame_objects.keys()):
            if prev_tid not in current_tids:
                # Objeto sumiu - registrar para possível re-ID futuro
                pos_px, color_hist, _, class_name = self._prev_frame_objects[prev_tid]
                print(f"[REID DEBUG] Objeto sumiu: tid={prev_tid}, classe={class_name}, pos={pos_px}, tem_histograma={color_hist is not None}, tam_hist={len(color_hist) if color_hist is not None else 0}")
                self.reid_detector.register_lost(
                    track_id=prev_tid,
                    class_name=class_name,
                    pos_px=pos_px,
                    timestamp=timestamp,
                    color_hist=color_hist,
                )

        # Limpar objetos expirados
        self.reid_detector.cleanup_expired(timestamp)

        return detections

    def _update_prev_frame_objects(
        self,
        detections: list[Detection],
        timestamp: float,
    ) -> None:
        """Registra objetos vistos neste frame para Re-ID no próximo frame."""
        self._prev_frame_objects.clear()
        for d in detections:
            if d.track_id is None:
                continue
            self._prev_frame_objects[d.track_id] = (
                d.bottom_center,
                d.color_hist,
                timestamp,
                d.class_name,
            )

    def _get_class_name_for_tid(self, tid: int, detections: list[Detection]) -> str:
        """Obtém o nome da classe de um track_id nas detecções atuais."""
        for d in detections:
            if d.track_id == tid:
                return d.class_name
        return "unknown"

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
            "stopped_vehicles": {
                "count": len(self.stopped_detector.get_stopped_ids()),
                "ids": sorted(self.stopped_detector.get_stopped_ids()),
            },
            "incidents": [
                {
                    "type": inc.incident_type,
                    "track_id": inc.track_id,
                    "zone": inc.zone,
                    "timestamp": inc.timestamp,
                    "description": inc.description,
                }
                for inc in self.incident_detector.get_active_incidents()
            ],
            "reid": self.reid_detector.get_stats(),
            "evaluation": self._get_evaluation_report(),
            "output_video": output_video_path,
            "timestamp": self._iso_now(),
        }

    # ------------------------------------------------------------------ #
    def _get_evaluation_report(self) -> dict:
        """Gera relatório de avaliação (mAP, precisão, recall, F1, erro de contagem)."""
        if not self.eval_mode or not self.gt_by_frame:
            return {"enabled": False}

        report = evaluate_detections(
            detections_by_frame=self.detections_by_frame,
            gt_by_frame=self.gt_by_frame,
            iou_threshold=0.5,
            class_names=self.detector.names,
        )
        return report_to_dict(report)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _iso_now() -> str:
        from datetime import datetime

        return datetime.now().isoformat(timespec="seconds")
