"""Orquestração dos modos de execução (pipeline principal).

Integra fonte de vídeo, detector/rastreador, zonas ROI, contador, estimador de
velocidade e visualização em um único loop, com gravação opcional de vídeo e
relatório final (JSON + CSV).
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

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
from .detector import Detector
from .speed import SpeedEstimator
from .video_source import VideoSource
from .visualization import Visualizer
from .zones import build_zones


class TrafficAnalyzer:
    """Executa a análise de tráfego de ponta a ponta para um AppConfig."""

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

        self.zones = build_zones(load_zones(cfg.zones_file))
        self.counter = ObjectCounter(self.zones)

        speed_cfg = load_speed(cfg.speed_file)
        if not mode_uses_speed(cfg.mode):
            speed_cfg.enabled = False
        self.speed = SpeedEstimator(speed_cfg)

        self.visualizer = Visualizer(show_speed=self.speed.enabled)
        self.source = VideoSource(cfg.source, skip_frames=cfg.skip_frames)

        self.output_dir = resolve_path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        cfg = self.cfg
        self.source.open()
        print("=" * 64)
        print(f"  {MODE_LABELS[cfg.mode]}")
        print(f"  Fonte : {cfg.source}  ({self.source.kind})")
        print(
            f"  Vídeo : {self.source.width}x{self.source.height} @ "
            f"{self.source.fps:.1f} FPS"
        )
        print(f"  Tracker: {cfg.tracker} | skip: {cfg.skip_frames} | device: {self.detector.device}")
        print(f"  Zonas ROI: {len(self.zones)} | Velocidade: {'sim' if self.speed.enabled else 'não'}")
        print("=" * 64)
        if cfg.show:
            print("  Janela de vídeo: pressione 'q' ou ESC para encerrar.")

        writer = self._make_writer()
        window = "Analise de Trafego"
        processed = 0
        wall_start = time.perf_counter()
        last_log = wall_start
        report: dict = {}

        try:
            for abs_idx, timestamp, frame in self.source.frames():
                detections = self.detector.track(frame)
                self.counter.update(detections, timestamp)
                self.speed.update(detections, timestamp)

                proc_fps = processed / max(1e-6, time.perf_counter() - wall_start)
                total = self.source.frame_count
                info = (
                    f"Frame {abs_idx + 1}/{total}" if total > 0 else f"Frame {abs_idx + 1}"
                )
                annotated = self.visualizer.draw(
                    frame,
                    detections,
                    self.zones,
                    self.counter,
                    self.speed if self.speed.enabled else None,
                    MODE_LABELS[cfg.mode],
                    proc_fps,
                    info,
                )

                if writer is not None:
                    writer.write(annotated)

                if cfg.show:
                    cv2.imshow(window, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("[pipeline] encerrado pelo usuário.")
                        break

                processed += 1
                now = time.perf_counter()
                if now - last_log >= 3.0:
                    self._log(processed, proc_fps)
                    last_log = now

                if cfg.max_frames and processed >= cfg.max_frames:
                    print(f"[pipeline] limite de {cfg.max_frames} frames atingido.")
                    break
        except KeyboardInterrupt:
            print("\n[pipeline] interrompido (Ctrl+C).")
        finally:
            self.source.release()
            if writer is not None:
                writer.release()
            if cfg.show:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - wall_start
        report = self._build_report(processed, elapsed)
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
            print("[pipeline] aviso: não foi possível criar o vídeo de saída.")
            return None
        self._video_path = path
        print(f"[pipeline] gravando vídeo em: {path}")
        return writer

    def _log(self, processed: int, fps: float) -> None:
        s = self.counter.summary()
        parts = [f"frames={processed}", f"fps={fps:.1f}"]
        parts.append(f"pessoas={s['by_class'].get('person', 0)}")
        veh = sum(s["by_class"].get(n, 0) for n in ("car", "truck", "bus", "motorcycle"))
        parts.append(f"veiculos={veh}")
        for zname, zdata in s["zones"].items():
            parts.append(f"{zname}={zdata['count']}")
        print("[proc] " + " | ".join(parts))

    def _build_report(self, processed: int, elapsed: float) -> dict:
        return {
            "mode": self.cfg.mode,
            "mode_label": MODE_LABELS[self.cfg.mode],
            "source": self.cfg.source,
            "tracker": self.cfg.tracker,
            "skip_frames": self.cfg.skip_frames,
            "frames_processed": processed,
            "elapsed_s": round(elapsed, 2),
            "processing_fps": round(processed / max(1e-6, elapsed), 2),
            "counting": self.counter.summary(),
            "speed": self.speed.stats() if self.speed.enabled else {"enabled": False},
            "speed_calibration": {
                "enabled": self.speed.enabled,
                "real_width_m": self.speed.cfg.real_width_m,
                "real_height_m": self.speed.cfg.real_height_m,
                "pixels_per_meter": self.speed.cfg.pixels_per_meter,
            },
            "output_video": str(getattr(self, "_video_path", "")) or None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def _save_report(self, report: dict) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.cfg.run_name or self.cfg.mode
        json_path = self.output_dir / f"{name}_{stamp}_report.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        csv_path = self.output_dir / f"{name}_{stamp}_zones.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["zona", "tipo", "contagem", "in", "out"])
            for zname, zdata in report["counting"]["zones"].items():
                w.writerow([zname, zdata["type"], zdata["count"], zdata["in"], zdata["out"]])
        print(f"[pipeline] relatório: {json_path}")
        print(f"[pipeline] CSV zonas: {csv_path}")

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
