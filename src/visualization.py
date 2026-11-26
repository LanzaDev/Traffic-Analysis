"""Interface visual: bounding boxes, IDs, zonas ROI, contadores e velocidade.

Desenha sobre o frame usando Supervision (sv) quando disponível:
    - Polígonos/linhas de ROI com nome e contagem (sv.Zone, sv.LineZone).
    - Bounding boxes coloridas por classe, com ID e confiança (sv.BoxAnnotator).
    - Rótulo de velocidade (km/h) abaixo dos veículos.
    - Painel HUD com modo, FPS, totais e contagens por zona.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    import supervision as sv

    _HAS_SUPERVISION = True
except ImportError:  # pragma: no cover - supervision opcional
    sv = None  # type: ignore
    _HAS_SUPERVISION = False

from .config import PERSON_CLASS
from .counter import ObjectCounter
from .detector import Detection
from .speed import SpeedEstimator
from .zones import Zone

# Paleta por classe (B, G, R).
CLASS_COLORS = {
    PERSON_CLASS: (0, 220, 0),  # pessoa: verde
    2: (255, 160, 0),  # car: azul-alaranjado
    3: (0, 200, 255),  # motorcycle: amarelo
    5: (200, 0, 200),  # bus: magenta
    7: (0, 120, 255),  # truck: laranja
}
DEFAULT_COLOR = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    def __init__(self, show_speed: bool = True) -> None:
        self.show_speed = show_speed
        self._box_annotator = (
            sv.BoxAnnotator(thickness=2)
            if _HAS_SUPERVISION
            else None
        )

    def draw(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        zones: list[Zone],
        counter: ObjectCounter,
        speed: Optional[SpeedEstimator],
        mode_label: str,
        fps: float,
        frame_info: str = "",
    ) -> np.ndarray:
        out = frame.copy()
        self._draw_zones(out, zones, counter)
        self._draw_detections(out, detections, speed)
        self._draw_hud(out, mode_label, fps, counter, speed, frame_info)
        return out

    # ------------------------------------------------------------------ #
    def _draw_zones(self, img: np.ndarray, zones: list[Zone], counter: ObjectCounter) -> None:
        overlay = img.copy()
        for z in zones:
            if z.type == "polygon":
                cv2.fillPoly(overlay, [z.contour], z.color)
                cv2.polylines(img, [z.contour], True, z.color, 2)
                label_pos = tuple(z.points[0].astype(int))
                ins, outs = counter.zone_in_out(z.name)
                text = f"{z.name}: {counter.zone_count(z.name)}"
                if ins or outs:
                    text += f" (IN {ins}/OUT {outs})"
                self._label(img, text, label_pos, z.color)
            else:
                p1 = tuple(z.p1.astype(int))
                p2 = tuple(z.p2.astype(int))
                cv2.line(img, p1, p2, z.color, 3)
                cv2.circle(img, p1, 5, z.color, -1)
                cv2.circle(img, p2, 5, z.color, -1)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
                ins, outs = counter.zone_in_out(z.name)
                text = f"{z.name}: {counter.zone_count(z.name)} (IN {ins}/OUT {outs})"
                self._label(img, text, mid, z.color)
        # Mistura o preenchimento translúcido dos polígonos.
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    def _draw_detections(
        self, img: np.ndarray, detections: list[Detection], speed: Optional[SpeedEstimator]
    ) -> None:
        if _HAS_SUPERVISION and self._box_annotator is not None and detections:
            # Converter detecções para formato supervision.
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            confidences = np.array([d.confidence for d in detections], dtype=np.float32)
            class_ids = np.array([d.class_id for d in detections], dtype=np.int32)
            tracker_ids = np.array(
                [d.track_id if d.track_id is not None else -1 for d in detections],
                dtype=np.int32,
            )

            detections_sv = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids,
                tracker_id=tracker_ids,
            )

            # Anotar boxes com supervision (sem rótulos de texto).
            annotated = self._box_annotator.annotate(
                scene=img.copy(), detections=detections_sv
            )
            img[:] = annotated[:]

            # Desenhar rótulos de ID/confiança com OpenCV (supervision não aceita labels neste método).
            for d in detections:
                color = CLASS_COLORS.get(d.class_id, DEFAULT_COLOR)
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                tid = d.track_id if d.track_id is not None else "?"
                label = f"ID{tid} {d.class_name} {d.confidence:.2f}"
                self._label(img, label, (x1, max(0, y1 - 6)), color)

                # Rótulo de velocidade abaixo do veículo.
                if self.show_speed and speed is not None and d.is_vehicle:
                    kmh = speed.get(d.track_id)
                    if kmh is not None:
                        self._label(
                            img,
                            f"{kmh:.0f} km/h",
                            (x1, min(img.shape[0] - 4, y2 + 16)),
                            (0, 255, 255),
                        )
        else:
            # Fallback: desenho puro OpenCV.
            for d in detections:
                color = CLASS_COLORS.get(d.class_id, DEFAULT_COLOR)
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                tid = d.track_id if d.track_id is not None else "?"
                label = f"ID{tid} {d.class_name} {d.confidence:.2f}"
                self._label(img, label, (x1, max(0, y1 - 6)), color)

                # Rótulo de velocidade abaixo do veículo.
                if self.show_speed and speed is not None and d.is_vehicle:
                    kmh = speed.get(d.track_id)
                    if kmh is not None:
                        self._label(
                            img,
                            f"{kmh:.0f} km/h",
                            (x1, min(img.shape[0] - 4, y2 + 16)),
                            (0, 255, 255),
                        )

    def _draw_hud(
        self,
        img: np.ndarray,
        mode_label: str,
        fps: float,
        counter: ObjectCounter,
        speed: Optional[SpeedEstimator],
        frame_info: str,
    ) -> None:
        lines = [
            f"Modo: {mode_label}",
            f"FPS proc: {fps:.1f}",
        ]
        if frame_info:
            lines.append(frame_info)

        people = counter.class_count("person")
        vehicles = sum(
            counter.class_count(n) for n in ("car", "truck", "bus", "motorcycle")
        )
        lines.append(f"Pessoas (unicas): {people}")
        lines.append(f"Veiculos (unicos): {vehicles}")
        for z in counter.zones:
            lines.append(f"  {z.name}: {counter.zone_count(z.name)}")
        if speed is not None and speed.enabled:
            st = speed.stats()
            if st.get("vehicles_with_speed"):
                lines.append(
                    f"Velocidade media: {st['avg_kmh']} km/h | max: {st['max_kmh']}"
                )

        pad = 8
        line_h = 22
        width = 360
        height = pad * 2 + line_h * len(lines)
        cv2.rectangle(img, (10, 10), (10 + width, 10 + height), (20, 20, 20), -1)
        cv2.rectangle(img, (10, 10), (10 + width, 10 + height), (90, 90, 90), 1)
        for i, text in enumerate(lines):
            y = 10 + pad + line_h * (i + 1) - 6
            cv2.putText(img, text, (18, y), FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    @staticmethod
    def _label(img: np.ndarray, text: str, pos: tuple[int, int], color: tuple) -> None:
        (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
        x, y = pos
        x = max(0, min(x, img.shape[1] - tw - 2))
        y = max(th + 2, min(y, img.shape[0] - 2))
        cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + baseline), color, -1)
        cv2.putText(img, text, (x + 1, y - 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
