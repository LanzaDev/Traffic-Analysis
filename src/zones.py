"""Regiões de Interesse (ROI): polígonos fechados e linhas de travessia.

Fornece testes geométricos:
    - polygon.contains(pt): ponto dentro do polígono (cv2.pointPolygonTest).
    - line.crossing(prev, curr): detects se o segmento de movimento cruza a
      linha e em qual sentido ("in"/"out"), usando produto vetorial.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import (
    LINE_DIRECTION_BIDIRECTIONAL,
    LINE_DIRECTION_INBOUND,
    LINE_DIRECTION_OUTBOUND,
    ZoneConfig,
)

Point = tuple[float, float]

# Cores padrão (B, G, R) caso a zona não defina uma.
DEFAULT_POLYGON_COLOR = (0, 200, 255)
DEFAULT_LINE_COLOR = (0, 255, 128)


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Produto vetorial (b-a) x (c-a); sinal indica o lado de c em relação a ab."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Testa se os segmentos p1p2 e p3p4 se intersectam (caso geral)."""
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    return False


class Zone:
    """Geometria de uma região de interesse."""

    def __init__(self, cfg: ZoneConfig) -> None:
        self.name = cfg.name
        self.type = cfg.type
        self.zone_type = cfg.zone_type  # "road", "sidewalk", "avoid"
        self.line_direction = cfg.line_direction  # "inbound", "outbound", "bidirectional"
        self.points = np.array(cfg.points, dtype=np.float32)
        if cfg.color:
            self.color = tuple(int(c) for c in cfg.color)
        else:
            self.color = (
                DEFAULT_POLYGON_COLOR if self.type == "polygon" else DEFAULT_LINE_COLOR
            )

        if self.type == "polygon":
            self.contour = self.points.astype(np.int32).reshape(-1, 1, 2)
        else:
            self.p1 = self.points[0]
            self.p2 = self.points[1]

    def contains(self, pt: Point) -> bool:
        """True se o ponto está dentro (ou na borda) do polígono."""
        if self.type != "polygon":
            return False
        return (
            cv2.pointPolygonTest(self.contour, (float(pt[0]), float(pt[1])), False) >= 0
        )

    def crossing(self, prev: Point, curr: Point) -> Optional[str]:
        """Retorna 'in'/'out' se o movimento prev->curr cruza a linha; senão None.

        O sentido é definido pelo lado do ponto em relação à linha (p1->p2):
        negativo -> positivo é 'in'; positivo -> negativo é 'out'.
        """
        if self.type != "line":
            return None
        if not _segments_intersect(prev, curr, self.p1, self.p2):
            return None
        side_prev = _orientation(self.p1, self.p2, prev)
        side_curr = _orientation(self.p1, self.p2, curr)
        if side_prev < 0 <= side_curr:
            return "in"
        if side_prev > 0 >= side_curr:
            return "out"
        return "in"

    def crossing_direction(self, prev: Point, curr: Point) -> Optional[str]:
        """Retorna 'inbound'/'outbound' se o movimento cruza a linha no sentido esperado.

        Retorna None se não houver cruzamento ou se a linha for bidirecional.
        """
        if self.type != "line":
            return None
        crossing = self.crossing(prev, curr)
        if crossing is None:
            return None
        if self.line_direction == LINE_DIRECTION_BIDIRECTIONAL:
            return crossing  # não há sentido definido
        # Para linhas unidirecionais, o sentido depende da direção da linha
        # Se p1->p2 é "inbound", então crossing "in" = inbound, "out" = outbound
        if self.line_direction == LINE_DIRECTION_INBOUND:
            return "inbound" if crossing == "in" else "outbound"
        else:  # LINE_DIRECTION_OUTBOUND
            return "outbound" if crossing == "in" else "inbound"


def build_zones(configs: list[ZoneConfig]) -> list[Zone]:
    return [Zone(c) for c in configs]
