"""Runner de I/O para análise de tráfego.

Cuida de tudo que envolve entrada/saída: abrir fonte de vídeo, loop de
frames, exibição em janela, gravação de vídeo anotado, captura de teclas
(q/ESC), salvamento de relatórios JSON/CSV em disco e logs no terminal.

A lógica pura de detecção/contagem/velocidade fica em TrafficAnalyzer
(src/analyzer.py).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from .analyzer import TrafficAnalyzer, FrameResult
from .config import AppConfig, MODE_LABELS, resolve_path

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Orquestra execução com I/O: fonte, janela, vídeo de saída, relatórios."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.analyzer = TrafficAnalyzer(cfg)
        self.source = __import__("src.video_source", fromlist=["VideoSource"]).VideoSource(
            cfg.source, skip_frames=cfg.skip_frames
        )

        self.output_dir = resolve_path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        self._video_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        cfg = self.cfg
        self.source.open()
        logger.info("=" * 64)
        logger.info("  %s", MODE_LABELS[cfg.mode])
        logger.info("  Fonte : %s  (%s)", cfg.source, self.source.kind)
        logger.info(
            "  Vídeo : %sx%s @ %.1f FPS",
            self.source.width,
            self.source.height,
            self.source.fps,
        )
        logger.info(
            "  Tracker: %s | skip: %s | device: %s",
            cfg.tracker,
            cfg.skip_frames,
            self.analyzer.detector.device,
        )
        logger.info(
            "  Zonas ROI: %d | Velocidade: %s",
            len(self.analyzer.zones),
            "sim" if self.analyzer.speed.enabled else "não",
        )
        logger.info("=" * 64)
        if cfg.show:
            logger.info("  Janela de vídeo: pressione 'q' ou ESC para encerrar.")

        writer = self._make_writer()
        window = "Analise de Trafego"
        wall_start = time.perf_counter()
        last_log = wall_start

        try:
            for abs_idx, timestamp, frame in self.source.frames():
                result = self.analyzer.process_frame(frame, timestamp, abs_idx)

                proc_fps = (
                    self.analyzer.stats.frames_processed
                    / max(1e-6, time.perf_counter() - wall_start)
                )
                total = self.source.frame_count
                info = (
                    f"Frame {abs_idx + 1}/{total}" if total > 0 else f"Frame {abs_idx + 1}"
                )

                annotated = self.analyzer.annotate_frame(
                    frame, result, proc_fps, info
                )

                if writer is not None:
                    writer.write(annotated)

                if cfg.show:
                    cv2.imshow(window, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        logger.info("[runner] encerrado pelo usuário.")
                        break

                now = time.perf_counter()
                if now - last_log >= 3.0:
                    self._log(proc_fps)
                    last_log = now

                if cfg.max_frames and self.analyzer.stats.frames_processed >= cfg.max_frames:
                    logger.info(
                        "[runner] limite de %d frames atingido.", cfg.max_frames
                    )
                    break
        except KeyboardInterrupt:
            logger.info("\n[runner] interrompido (Ctrl+C).")
        finally:
            self.source.release()
            if writer is not None:
                writer.release()
            if cfg.show:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - wall_start
        report = self.analyzer.get_report(output_video_path=self._video_path)
        self._save_report(report)
        self._print_summary(report)
        return report

    # ------------------------------------------------------------------ #
    def _make_writer(self):
        if not self.cfg.save_video:
            return None
        fps_out = max(1.0, self.source.fps / self.source.skip_frames)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.cfg.run_name or self.cfg.mode
        path = self.output_dir / f"{name}_{stamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path), fourcc, fps_out, (self.source.width, self.source.height)
        )
        if not writer.isOpened():
            logger.warning("[runner] não foi possível criar o vídeo de saída.")
            return None
        self._video_path = str(path)
        logger.info("[runner] gravando vídeo em: %s", path)
        return writer

    def _log(self, fps: float) -> None:
        s = self.analyzer.counter.summary()
        parts = [
            f"frames={self.analyzer.stats.frames_processed}",
            f"fps={fps:.1f}",
        ]
        parts.append(f"pessoas={s['by_class'].get('person', 0)}")
        veh = sum(
            s["by_class"].get(n, 0)
            for n in ("car", "truck", "bus", "motorcycle")
        )
        parts.append(f"veiculos={veh}")
        for zname, zdata in s["zones"].items():
            parts.append(f"{zname}={zdata['count']}")
        logger.info("[proc] %s", " | ".join(parts))

    def _save_report(self, report: dict) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.cfg.run_name or self.cfg.mode
        json_path = self.output_dir / f"{name}_{stamp}_report.json"
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        csv_path = self.output_dir / f"{name}_{stamp}_zones.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["zona", "tipo", "contagem", "in", "out"])
            for zname, zdata in report["counting"]["zones"].items():
                w.writerow(
                    [zname, zdata["type"], zdata["count"], zdata["in"], zdata["out"]]
                )
        logger.info("[runner] relatório: %s", json_path)
        logger.info("[runner] CSV zonas: %s", csv_path)

    def _print_summary(self, report: dict) -> None:
        c = report["counting"]
        print("\n" + "=" * 64)
        print("  RESUMO FINAL")
        print("=" * 64)
        print(f"  Frames processados : {report['frames_processed']}")
        print(f"  Tempo decorrido    : {report['elapsed_s']} s")
        print(f"  FPS de processamento: {report['processing_fps']}")
        print(f"  Objetos únicos     : {c['total_unique_objects']}")
        for cls, n in c["by_class"].items():
            print(f"    - {cls}: {n}")
        if c["zones"]:
            print("  Contagem por zona:")
            for zname, zdata in c["zones"].items():
                print(
                    f"    - {zname} ({zdata['type']}): {zdata['count']} "
                    f"[IN {zdata['in']} / OUT {zdata['out']}]"
                )
        if report["speed"].get("vehicles_with_speed"):
            print(
                f"  Velocidade média: {report['speed']['avg_kmh']} km/h | "
                f"máxima: {report['speed']['max_kmh']} km/h"
            )
        print("=" * 64)
