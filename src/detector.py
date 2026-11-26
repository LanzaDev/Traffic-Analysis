"""Detecção e rastreamento de objetos com YOLO (Ultralytics) + ByteTrack/BoT-SORT.

Envolve o modelo YOLO e expõe uma interface limpa que retorna objetos
`Detection` com ID persistente, classe, confiança e caixa delimitadora.
O rastreamento frame a frame usa `persist=True` para manter os IDs entre frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import CONCESSIONARIA_CLASS_NAMES, VEHICLE_CLASSES, tracker_yaml


@dataclass
class Detection:
    """Um objeto detectado e rastreado em um frame."""

    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    bottom_center: tuple  # (cx, y2) — ponto de contato com o chão
    color_hist: Optional[np.ndarray] = None  # histograma de cor da região da bbox

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASSES


class Detector:
    """Carrega o modelo YOLO e executa detecção + rastreamento."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu",
        tracker: str = "bytetrack",
        classes: Optional[list[int]] = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.tracker = tracker_yaml(tracker)
        self.device = self._resolve_device(device)
        self.names = self.model.names

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.lower() in ("auto", ""):
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        return device

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Detecta + rastreia objetos no frame, retornando a lista de Detection."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )
        results_list = list(results)
        return self._parse(results_list[0])

    def track_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detecta + rastreia objetos em um lote de frames.

        Retorna uma lista de listas de Detection (uma por frame do lote).
        O rastreamento com persist=True mantém IDs entre frames e lotes.
        """
        results = self.model.track(
            source=frames,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )
        return [self._parse(r) for r in results]

    def export_model(self, fmt: str) -> str:
        """Exporta o modelo YOLO para um formato otimizado.

        Suportado: onnx, engine (TensorRT), openvino, ncnn.
        Retorna o caminho do arquivo exportado.
        """
        exported = self.model.export(format=fmt, imgsz=self.imgsz, device=self.device)
        return str(exported)

    @staticmethod
    def extract_color_hist(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """Extrai histograma de cor HSV da região da bbox para Re-ID."""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            hist = cv2.calcHist([roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except Exception:
            return None

    def _parse(self, result) -> list[Detection]:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        coords = boxes.xyxy.cpu().numpy()

        for tid, cls_id, conf, (x1, y1, x2, y2) in zip(ids, clss, confs, coords):
            name = self.names.get(cls_id, str(cls_id))
            cx = (x1 + x2) / 2.0
            detections.append(
                Detection(
                    track_id=int(tid) if tid is not None else None,
                    class_id=int(cls_id),
                    class_name=name,
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    bottom_center=(float(cx), float(y2)),
                    color_hist=None,  # preenchido posteriormente pelo analyzer
                )
            )
        return detections
