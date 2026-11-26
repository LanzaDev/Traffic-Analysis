"""Detecção de veículos parados e incidentes de tráfego.

Acompanha a posição de cada ID rastreado ao longo do tempo e classifica
um veículo como parado quando seu deslocamento fica abaixo de um limiar
dentro de uma janela temporal. Incidentes são gerados quando veículos
parados permanecem em zonas críticas (faixas de rolamento) ou quando
muitos veículos param simultaneamente (tráfego parado).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import AppConfig
from .counter import ObjectCounter
from .detector import Detection
from .speed import SpeedEstimator
from .zones import Zone


@dataclass
class StoppedVehicleEvent:
    """Evento de veículo parado detectado."""

    track_id: int
    class_name: str
    zone: str  # nome da zona onde está parado
    started_at: float  # timestamp em que começou a ser considerado parado
    last_updated: float  # último timestamp em que ainda estava parado
    duration_s: float  # duração atual da parada


@dataclass
class IncidentEvent:
    """Evento de incidente de tráfego."""

    incident_type: str  # "stopped_vehicle", "traffic_stopped", "wrong_way"
    track_id: Optional[int]  # None se for incidente de tráfego geral
    zone: str
    timestamp: float
    description: str


class StoppedVehicleDetector:
    """Detecta veículos parados com base no deslocamento entre frames."""

    def __init__(
        self,
        min_displacement_m: float = 0.05,  # abaixo disso considera movimento insignificante
        min_stopped_frames: int = 10,  # nº mínimo de frames consecutivos parados
        min_stopped_s: float = 2.0,  # tempo mínimo para considerar parada (segundos)
    ) -> None:
        self.min_displacement_m = min_displacement_m
        self.min_stopped_frames = min_stopped_frames
        self.min_stopped_s = min_stopped_s

        # Histórico de posições por track_id: lista de (timestamp, x_m, y_m)
        self._history: dict[int, list[tuple[float, float, float]]] = {}
        # Contador de frames consecutivos parados por ID
        self._stopped_frame_count: dict[int, int] = {}
        # Timestamp em que cada ID começou a ser considerado parado
        self._stopped_since: dict[int, float] = {}
        # IDs atualmente parados
        self._stopped_ids: set[int] = set()

    def update(
        self, detections: list[Detection], timestamp: float, speed: Optional[SpeedEstimator] = None
    ) -> list[StoppedVehicleEvent]:
        """Processa detecções e retorna eventos de veículos parados iniciados/atualizados."""
        events: list[StoppedVehicleEvent] = []
        detected_tids: set[int] = set()

        for d in detections:
            if d.track_id is None:
                continue
            tid = d.track_id
            detected_tids.add(tid)

            # Usar posição em metros se disponível (via speed estimator), senão pixels
            wx, wy = self._get_world_pos(d, speed)
            h = self._history.setdefault(tid, [])
            h.append((timestamp, wx, wy))

            # Manter histórico limitado (últimos 60 frames)
            if len(h) > 60:
                h.pop(0)

            # Verificar se está parado
            is_stopped = self._is_stopped(tid, timestamp)

            if is_stopped:
                self._stopped_frame_count[tid] = self._stopped_frame_count.get(tid, 0) + 1
                if tid not in self._stopped_ids and self._stopped_frame_count[tid] >= self.min_stopped_frames:
                    # Veículo acabou de ser classificado como parado
                    self._stopped_ids.add(tid)
                    self._stopped_since[tid] = timestamp
                    events.append(
                        StoppedVehicleEvent(
                            track_id=tid,
                            class_name=d.class_name,
                            zone="",  # será preenchido pelo IncidentDetector
                            started_at=timestamp,
                            last_updated=timestamp,
                            duration_s=0.0,
                        )
                    )
                elif tid in self._stopped_ids:
                    # Atualizar evento existente
                    duration = timestamp - self._stopped_since[tid]
                    events.append(
                        StoppedVehicleEvent(
                            track_id=tid,
                            class_name=d.class_name,
                            zone="",
                            started_at=self._stopped_since[tid],
                            last_updated=timestamp,
                            duration_s=duration,
                        )
                    )
            else:
                # Veículo se moveu, remover de parado
                if tid in self._stopped_ids:
                    self._stopped_ids.discard(tid)
                    self._stopped_frame_count.pop(tid, None)
                    self._stopped_since.pop(tid, None)

        # Limpar IDs que não foram detectados neste frame (opcional: manter por alguns frames)
        # Não removemos imediatamente para não perder parada temporária por oclusão

        return events

    def _get_world_pos(self, d: Detection, speed: Optional[SpeedEstimator]) -> tuple[float, float]:
        """Retorna posição em metros (via speed estimator) ou pixels (fallback)."""
        if speed is not None and speed.enabled:
            try:
                return speed.to_world(d.bottom_center)
            except Exception:
                pass
        cx, cy = d.bottom_center
        return cx, cy

    def _is_stopped(self, tid: int, timestamp: float) -> bool:
        """Verifica se o veículo está parado com base no deslocamento recente."""
        h = self._history.get(tid, [])
        if len(h) < 2:
            return False

        # Usar primeiro e último ponto da janela para calcular deslocamento total
        t0, x0, y0 = h[0]
        t1, x1, y1 = h[-1]

        dt = t1 - t0
        if dt <= 1e-3:
            return False

        # Deslocamento em metros (ou pixels, se não houver calibração)
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

        # Também verificar tempo mínimo de parada
        if dt < self.min_stopped_s:
            return False

        return dist < self.min_displacement_m

    def get_stopped_ids(self) -> set[int]:
        """Retorna IDs atualmente parados."""
        return set(self._stopped_ids)

    def get_stopped_duration(self, tid: int) -> Optional[float]:
        """Retorna duração da parada em segundos, ou None se não estiver parado."""
        if tid not in self._stopped_since:
            return None
        # Usar o último timestamp conhecido
        return None  # será preenchido pelo chamador se necessário


class IncidentDetector:
    """Detecta incidentes de tráfego com base em veículos parados e zonas."""

    def __init__(self, zones: list, min_parked_vehicles: int = 3) -> None:
        self.zones = {z.name: z for z in zones}
        self.min_parked_vehicles = min_parked_vehicles
        # Histórico de incidentes ativos
        self.active_incidents: dict[str, IncidentEvent] = {}

    def update(
        self,
        stopped_events: list[StoppedVehicleEvent],
        detections: list[Detection],
        counter: ObjectCounter,
        timestamp: float,
    ) -> list[IncidentEvent]:
        """Processa eventos de parada e retorna incidentes ativos."""
        incidents: list[IncidentEvent] = []

        # Mapear track_id -> classe para eventos de parada
        stopped_by_id: dict[int, StoppedVehicleEvent] = {}
        for ev in stopped_events:
            stopped_by_id[ev.track_id] = ev

        # Detectar veículos parados em zonas críticas
        for ev in stopped_events:
            zone_name = self._find_zone_for_tid(ev.track_id, detections)
            ev.zone = zone_name

            incident_key = f"stopped_{ev.track_id}"
            if incident_key not in self.active_incidents:
                self.active_incidents[incident_key] = IncidentEvent(
                    incident_type="stopped_vehicle",
                    track_id=ev.track_id,
                    zone=zone_name,
                    timestamp=ev.started_at,
                    description=f"Veículo {ev.class_name} (ID {ev.track_id}) parado na zona {zone_name}",
                )
                incidents.append(self.active_incidents[incident_key])
            else:
                # Atualizar duração
                inc = self.active_incidents[incident_key]
                inc.timestamp = timestamp
                inc.description = f"Veículo {ev.class_name} (ID {ev.track_id}) parado na zona {zone_name} por {ev.duration_s:.1f}s"

        # Detectar tráfego parado (muitos veículos parados na mesma zona)
        zone_stopped_count: dict[str, int] = {}
        for ev in stopped_events:
            if ev.zone:
                zone_stopped_count[ev.zone] = zone_stopped_count.get(ev.zone, 0) + 1

        for zone_name, count in zone_stopped_count.items():
            if count >= self.min_parked_vehicles:
                incident_key = f"traffic_stopped_{zone_name}"
                if incident_key not in self.active_incidents:
                    self.active_incidents[incident_key] = IncidentEvent(
                        incident_type="traffic_stopped",
                        track_id=None,
                        zone=zone_name,
                        timestamp=timestamp,
                        description=f"Tráfego parado: {count} veículos na zona {zone_name}",
                    )
                    incidents.append(self.active_incidents[incident_key])
                else:
                    inc = self.active_incidents[incident_key]
                    inc.timestamp = timestamp
                    inc.description = f"Tráfego parado: {count} veículos na zona {zone_name}"

        return incidents

    def _find_zone_for_tid(self, tid: int, detections: list[Detection]) -> str:
        """Encontrar a zona onde o veículo está parado (baseado na última posição)."""
        for d in detections:
            if d.track_id == tid:
                pt = d.bottom_center
                for z in self.zones.values():
                    if z.contains(pt):
                        return z.name
        return "desconhecida"

    def get_active_incidents(self) -> list[IncidentEvent]:
        """Retorna todos os incidentes ativos."""
        return list(self.active_incidents.values())