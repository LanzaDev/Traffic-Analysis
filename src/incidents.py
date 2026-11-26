"""Eventos de incidentes de tráfego: contramão e pedestre na pista."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class WrongWayEvent:
    """Evento de veículo em contramão."""

    track_id: int
    class_name: str
    zone: str  # nome da zona/linha onde ocorreu
    direction: str  # "inbound" ou "outbound" (sentido detectado)
    timestamp: float


@dataclass
class PedestrianOnRoadEvent:
    """Evento de pedestre detectado na pista."""

    track_id: int
    zone: str  # nome da zona onde foi detectado
    timestamp: float