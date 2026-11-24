"""Configurações centrais, dataclasses e utilitários de carregamento.

Centraliza os modos de operação, o mapeamento de classes COCO e a leitura
dos arquivos de configuração de zonas (ROI) e de calibração de velocidade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Classes COCO utilizadas pelo modelo pré-treinado (yolo11n / yolov8n).
# --------------------------------------------------------------------------- #
PERSON_CLASS = 0
# COCO usa "motorcycle"; aceitamos também o alias "motorbike" citado no enunciado.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_ALIASES = {"motorbike": "motorcycle"}

MODES = ("people", "vehicles", "speed", "full")

MODE_LABELS = {
    "people": "Contagem de Pessoas",
    "vehicles": "Contagem de Veículos",
    "speed": "Estimação de Velocidade",
    "full": "Execução Completa (contagem + velocidade)",
}

# Raiz do projeto (diretório que contém main.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def classes_for_mode(mode: str) -> list[int]:
    """Retorna os IDs de classe COCO a detectar/rastrear para cada modo."""
    if mode == "people":
        return [PERSON_CLASS]
    if mode in ("vehicles", "speed"):
        return list(VEHICLE_CLASSES.keys())
    if mode == "full":
        return [PERSON_CLASS] + list(VEHICLE_CLASSES.keys())
    raise ValueError(f"Modo inválido: {mode!r}. Use um de {MODES}.")


def mode_uses_speed(mode: str) -> bool:
    return mode in ("speed", "full")


def mode_counts_people(mode: str) -> bool:
    return mode in ("people", "full")


def mode_counts_vehicles(mode: str) -> bool:
    return mode in ("vehicles", "speed", "full")


def resolve_path(value: str | Path | None) -> Optional[Path]:
    """Resolve caminhos relativos em relação à raiz do projeto.

    URLs (rtsp://, http://, https://) e índices de webcam são retornados como
    string pelo caller; aqui tratamos apenas caminhos de arquivo.
    """
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def tracker_yaml(name: str) -> str:
    """Converte o nome amigável do tracker no arquivo YAML do Ultralytics."""
    mapping = {
        "bytetrack": "bytetrack.yaml",
        "botsort": "botsort.yaml",
        "deepocsort": "deepocsort.yaml",
        "ocsort": "ocsort.yaml",
    }
    return mapping.get(name.lower(), "bytetrack.yaml")


# --------------------------------------------------------------------------- #
# Dataclasses de configuração.
# --------------------------------------------------------------------------- #
@dataclass
class ZoneConfig:
    """Define uma região de interesse (polígono fechado ou linha de travessia)."""

    name: str
    type: str  # "polygon" | "line"
    points: list  # lista de [x, y]
    color: Optional[list] = None  # [B, G, R] opcional

    def __post_init__(self) -> None:
        self.type = self.type.lower()
        if self.type not in ("polygon", "line"):
            raise ValueError(f"Tipo de zona inválido: {self.type}")
        if self.type == "polygon" and len(self.points) < 3:
            raise ValueError(f"Polígono '{self.name}' precisa de >= 3 pontos.")
        if self.type == "line" and len(self.points) != 2:
            raise ValueError(f"Linha '{self.name}' precisa de exatamente 2 pontos.")


@dataclass
class SpeedConfig:
    """Calibração para estimação de velocidade via transformação de perspectiva.

    source_points: 4 pontos [x, y] na imagem, na ordem
        topo-esquerda, topo-direita, base-direita, base-esquerda,
        delimitando um trecho da pista no plano do chão.
    real_width_m / real_height_m: dimensões reais (metros) desse trecho.
    pixels_per_meter: escala da vista "Bird's Eye View" gerada.
    """

    enabled: bool = False
    source_points: list = field(default_factory=list)
    real_width_m: float = 8.0
    real_height_m: float = 20.0
    pixels_per_meter: float = 20.0
    smoothing_alpha: float = 0.35  # suavização EMA (0..1)
    max_speed_kmh: float = 200.0  # teto de sanidade
    min_displacement_m: float = 0.05  # abaixo disso considera parado
    window_s: float = 0.5  # janela temporal para cálculo do deslocamento


@dataclass
class AppConfig:
    """Configuração completa de uma execução."""

    source: str = "assets/traffic.mp4"
    mode: str = "full"
    model_path: str = "models/yolo11n.pt"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device: str = "cpu"
    tracker: str = "bytetrack"
    skip_frames: int = 1
    zones_file: Optional[str] = "config/zones.json"
    speed_file: Optional[str] = "config/speed.json"
    output_dir: str = "output"
    save_video: bool = False
    show: bool = True
    max_frames: int = 0  # 0 = processar tudo
    run_name: str = ""
    json_output: bool = False
    report_format: str = "pretty"


# --------------------------------------------------------------------------- #
# Carregamento de JSON.
# --------------------------------------------------------------------------- #
def load_zones(path: str | Path | None) -> list[ZoneConfig]:
    """Carrega zonas de um JSON. Retorna lista vazia se arquivo não existir."""
    if not path:
        return []
    p = resolve_path(path)
    if not p or not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    zones = []
    for item in data.get("zones", []):
        zones.append(
            ZoneConfig(
                name=item["name"],
                type=item["type"],
                points=item["points"],
                color=item.get("color"),
            )
        )
    return zones


def load_speed(path: str | Path | None) -> SpeedConfig:
    """Carrega calibração de velocidade. Retorna config desabilitada se ausente."""
    if not path:
        return SpeedConfig()
    p = resolve_path(path)
    if not p or not p.exists():
        return SpeedConfig()
    data = json.loads(p.read_text(encoding="utf-8"))
    return SpeedConfig(
        enabled=bool(data.get("enabled", False)),
        source_points=data.get("source_points", []),
        real_width_m=float(data.get("real_width_m", 8.0)),
        real_height_m=float(data.get("real_height_m", 20.0)),
        pixels_per_meter=float(data.get("pixels_per_meter", 20.0)),
        smoothing_alpha=float(data.get("smoothing_alpha", 0.35)),
        max_speed_kmh=float(data.get("max_speed_kmh", 200.0)),
        min_displacement_m=float(data.get("min_displacement_m", 0.05)),
        window_s=float(data.get("window_s", 0.5)),
    )
