"""Módulo de contagem baseado em IDs únicos rastreados (sets).

Cada zona mantém um conjunto (set) de IDs já contados, garantindo que um objeto
seja contabilizado apenas na PRIMEIRA travessia/presença, evitando duplicidade
entre frames. Também mantém contagens globais únicas por classe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .detector import Detection
from .zones import Zone


@dataclass
class CountEvent:
    """Um evento de contagem (objeto cruzou/entrou em uma zona)."""

    zone: str
    track_id: int
    class_name: str
    direction: str  # "in" | "out"
    timestamp: float


class ObjectCounter:
    """Contabiliza objetos por zona e globalmente usando sets de IDs."""

    def __init__(self, zones: list[Zone]) -> None:
        self.zones = zones
        # IDs já contados em cada zona (primeira travessia apenas).
        self.zone_ids: dict[str, set[int]] = {z.name: set() for z in zones}
        self.zone_direction: dict[tuple[str, int], str] = {}
        # Última posição (bottom center) por ID, para detectar cruzamento de linha.
        self.last_pos: dict[int, tuple[float, float]] = {}
        # Contagens globais únicas.
        self.global_ids: set[int] = set()
        self.class_ids: dict[str, set[int]] = {}

    def update(self, detections: list[Detection], timestamp: float) -> list[CountEvent]:
        """Atualiza contadores com as detecções do frame; retorna novos eventos."""
        events: list[CountEvent] = []
        for d in detections:
            tid = d.track_id
            if tid is None:
                continue

            self.global_ids.add(tid)
            self.class_ids.setdefault(d.class_name, set()).add(tid)

            pt = d.bottom_center
            prev = self.last_pos.get(tid)

            for z in self.zones:
                if tid in self.zone_ids[z.name]:
                    continue  # já contado nesta zona
                direction: Optional[str] = None
                if z.type == "polygon":
                    if z.contains(pt):
                        direction = "in"
                elif z.type == "line" and prev is not None:
                    direction = z.crossing(prev, pt)

                if direction:
                    self.zone_ids[z.name].add(tid)
                    self.zone_direction[(z.name, tid)] = direction
                    events.append(CountEvent(z.name, tid, d.class_name, direction, timestamp))

            self.last_pos[tid] = pt

        return events

    def zone_count(self, name: str) -> int:
        return len(self.zone_ids.get(name, set()))

    def zone_in_out(self, name: str) -> tuple[int, int]:
        ins = sum(1 for (z, _), d in self.zone_direction.items() if z == name and d == "in")
        outs = sum(1 for (z, _), d in self.zone_direction.items() if z == name and d == "out")
        return ins, outs

    def class_count(self, name: str) -> int:
        return len(self.class_ids.get(name, set()))

    def total_unique(self) -> int:
        return len(self.global_ids)

    def summary(self) -> dict:
        """Resumo serializável para relatório."""
        return {
            "total_unique_objects": self.total_unique(),
            "by_class": {k: len(v) for k, v in sorted(self.class_ids.items())},
            "zones": {
                z.name: {
                    "count": self.zone_count(z.name),
                    "in": self.zone_in_out(z.name)[0],
                    "out": self.zone_in_out(z.name)[1],
                    "type": z.type,
                }
                for z in self.zones
            },
        }
