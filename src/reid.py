"""Detecção de Re-ID (Re-identificação) de objetos.

Quando um objeto some da detecção (por oclusão, distância, imagem borrada),
o rastreador pode perder o ID e criar um novo quando reaparecer.
Este módulo tenta reidentificar objetos "desaparecidos" com base em:
    - proximidade espacial (última posição conhecida em pixels)
    - similaridade de cor (histograma HSV da região da bbox)
    - janela temporal (quanto tempo sumiu)
    - classe do objeto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class LostObject:
    """Um objeto que sumiu da detecção e está sendo monitorado para re-ID."""

    track_id: int
    class_name: str
    last_pos_px: tuple[float, float]  # última posição em pixels (x, y)
    last_seen_s: float  # timestamp em que foi visto pela última vez
    color_hist: Optional[np.ndarray]  # histograma de cor da última detecção
    frames_lost: int = 0  # quantos frames está sumido

    def is_expired(self, current_time: float, max_time_s: float) -> bool:
        """Verifica se o objeto expirou (sumiu há muito tempo)."""
        return (current_time - self.last_seen_s) > max_time_s


class ReIDDetector:
    """Detecta quando um objeto "desaparecido" reaparece e reatribui o ID antigo."""

    def __init__(
        self,
        max_distance_px: float = 150.0,  # distância máxima em pixels para considerar mesmo objeto
        max_time_s: float = 5.0,  # tempo máximo (s) que um objeto pode ficar "desaparecido"
        color_tolerance: float = 0.3,  # tolerância de diferença de cor (0-1, menor = mais rígido)
    ) -> None:
        self.max_distance_px = max_distance_px
        self.max_time_s = max_time_s
        self.color_tolerance = color_tolerance

        # Objetos desaparecidos aguardando re-ID
        self.lost_objects: dict[int, LostObject] = {}
        # Mapeamento: novo_id -> antigo_id (para rastrear re-ID realizados)
        self.id_mapping: dict[int, int] = {}
        # Contador de re-ID bem-sucedidos
        self.reid_count: int = 0

    def register_lost(
        self,
        track_id: int,
        class_name: str,
        pos_px: tuple[float, float],
        timestamp: float,
        color_hist: Optional[np.ndarray] = None,
    ) -> None:
        """Registra um objeto que sumiu da detecção."""
        self.lost_objects[track_id] = LostObject(
            track_id=track_id,
            class_name=class_name,
            last_pos_px=pos_px,
            last_seen_s=timestamp,
            color_hist=color_hist,
        )

    def update_lost(
        self,
        track_id: int,
        pos_px: tuple[float, float],
        timestamp: float,
        color_hist: Optional[np.ndarray] = None,
    ) -> None:
        """Atualiza a posição de um objeto desaparecido."""
        if track_id in self.lost_objects:
            obj = self.lost_objects[track_id]
            obj.last_pos_px = pos_px
            obj.last_seen_s = timestamp
            if color_hist is not None:
                obj.color_hist = color_hist
            obj.frames_lost += 1

    def remove_lost(self, track_id: int) -> None:
        """Remove um objeto da lista de perdidos (ex: foi reidentificado ou expirou)."""
        self.lost_objects.pop(track_id, None)

    def cleanup_expired(self, current_time: float) -> list[int]:
        """Remove objetos que excederam o tempo máximo de desaparecimento."""
        expired = [
            tid
            for tid, obj in self.lost_objects.items()
            if obj.is_expired(current_time, self.max_time_s)
        ]
        for tid in expired:
            self.remove_lost(tid)
        return expired

    def find_match(
        self,
        new_track_id: int,
        class_name: str,
        pos_px: tuple[float, float],
        color_hist: Optional[np.ndarray],
        timestamp: float,
        max_dist_px: float | None = None,
    ) -> Optional[int]:
        """Tenta encontrar um objeto desaparecido que corresponda ao novo.

        Retorna o track_id antigo se houver match, ou None.
        """
        max_dist = max_dist_px if max_dist_px is not None else self.max_distance_px
        best_tid: Optional[int] = None
        best_score: float = float("inf")  # menor score = melhor match

        for tid, obj in self.lost_objects.items():
            # Verificar classe compatível
            if obj.class_name != class_name:
                continue

            # Verificar tempo máximo
            if obj.is_expired(timestamp, self.max_time_s):
                continue

            # Distância euclidiana em pixels
            dx = pos_px[0] - obj.last_pos_px[0]
            dy = pos_px[1] - obj.last_pos_px[1]
            dist_px = (dx * dx + dy * dy) ** 0.5
            if dist_px > max_dist:
                continue

            # Similaridade de cor
            color_dist = 1.0  # neutro (sem informação de cor)
            if obj.color_hist is not None and color_hist is not None:
                color_dist = self._color_distance(obj.color_hist, color_hist)
                if color_dist > self.color_tolerance:
                    continue  # cores muito diferentes

            # Score combinado: distância normalizada + cor
            dist_score = dist_px / max_dist  # normalizar para 0-1
            score = dist_score + color_dist * 0.5  # pesos ajustáveis

            if score < best_score:
                best_score = score
                best_tid = tid

        if best_tid is not None:
            print(f"[REID] Match encontrado: novo_id={new_track_id} -> antigo_id={best_tid}, score={best_score:.3f}")
        else:
            # Debug: mostrar por que nenhum match foi encontrado
            candidates = 0
            for tid, obj in self.lost_objects.items():
                if obj.class_name != class_name:
                    continue
                if obj.is_expired(timestamp, self.max_time_s):
                    continue
                candidates += 1
                dx = pos_px[0] - obj.last_pos_px[0]
                dy = pos_px[1] - obj.last_pos_px[1]
                dist_px = (dx * dx + dy * dy) ** 0.5
                color_info = ""
                if obj.color_hist is not None and color_hist is not None:
                    cd = self._color_distance(obj.color_hist, color_hist)
                    color_info = f", cor_dist={cd:.3f}"
                    if cd > self.color_tolerance:
                        color_info += " (cor muito diferente)"
                if dist_px > max_dist:
                    color_info = f", dist={dist_px:.1f}px (muito longe, max={max_dist:.1f}px){color_info}"
                else:
                    color_info = f", dist={dist_px:.1f}px{color_info}"
                print(f"[REID] Candidato {tid}:{color_info}")

        return best_tid

    def _color_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Calcula distância entre dois histogramas de cor.
        Retorna 0 = idênticos, 1 = muito diferentes.
        """
        try:
            # cv2.compareHist com HISTCMP_CORREL retorna 1 para idênticos, -1 para opostos
            corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
            return 1.0 - max(0.0, corr)  # 0 = idênticos, 1 = muito diferentes
        except Exception:
            return 1.0

    def match_and_update(
        self,
        new_track_id: int,
        old_track_id: int,
    ) -> None:
        """Marca um novo ID como re-ID de um antigo."""
        self.id_mapping[new_track_id] = old_track_id
        self.remove_lost(old_track_id)
        self.reid_count += 1

    def get_original_id(self, track_id: int) -> int:
        """Retorna o ID original se o track_id foi reidentificado."""
        return self.id_mapping.get(track_id, track_id)

    def get_stats(self) -> dict:
        """Retorna estatísticas do re-ID."""
        return {
            "lost_objects": len(self.lost_objects),
            "reid_count": self.reid_count,
            "active_mappings": len(self.id_mapping),
        }