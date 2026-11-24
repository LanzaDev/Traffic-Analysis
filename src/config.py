"""Configurações centrais, dataclasses e utilitários de carregamento.

Centraliza os modos de operação, o mapeamento de classes COCO e a leitura
dos arquivos de configuração de zonas (ROI) e de calibração de velocidade.
"""

from __future__ import annotations

import json
import os
import sys
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def has_display() -> bool:
    """Detecta se há um display gráfico disponível para exibição de janelas.

    - Windows: tenta criar uma janela OpenCV invisível (cv2.namedWindow).
    - Linux/Unix: verifica a variável de ambiente DISPLAY.
    - Outros: retorna False (assume headless por segurança).
    """
    import sys

    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))

    if sys.platform == "win32":
        try:
            import cv2

            # Tenta criar uma janela invisível; se falhar, assume headless.
            cv2.namedWindow("_display_check", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_display_check")
            return True
        except Exception:
            return False

    return False


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
# Carregamento de variáveis de ambiente (.env + SO).
# --------------------------------------------------------------------------- #
ENV_FILE = PROJECT_ROOT / ".env"


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "sim", "on")


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def load_env_config() -> dict:
    """Carrega configurações de ambiente (.env + variáveis do SO).

    Prioridade: variáveis do SO sobrescrevem o arquivo .env.
    Retorna um dicionário com os valores encontrados (apenas chaves presentes).
    """
    env_values = {}

    # Carrega .env se existir (formato simples KEY=VALUE, ignora comentários e linhas vazias).
    if ENV_FILE.exists():
        try:
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                env_values[key.strip()] = val.strip()
        except OSError:
            pass

    # Variáveis do SO sobrescrevem o .env.
    for key in (
        "TRAFFIC_MODEL_PATH",
        "TRAFFIC_DEFAULT_SOURCE",
        "TRAFFIC_TRACKER",
        "TRAFFIC_OUTPUT_DIR",
        "TRAFFIC_SKIP_FRAMES",
        "TRAFFIC_CONF",
        "TRAFFIC_IOU",
        "TRAFFIC_IMG_SIZE",
        "TRAFFIC_DEVICE",
        "TRAFFIC_ZONES_FILE",
        "TRAFFIC_SPEED_CONFIG_FILE",
        "TRAFFIC_SAVE_VIDEO",
        "TRAFFIC_NO_SHOW",
        "TRAFFIC_MAX_FRAMES",
        "TRAFFIC_RUN_NAME",
        "TRAFFIC_JSON_OUTPUT",
        "TRAFFIC_REPORT_FORMAT",
        "TRAFFIC_ADAPTIVE_SKIP",
        "TRAFFIC_TARGET_FPS",
        "TRAFFIC_MIN_SKIP",
        "TRAFFIC_MAX_SKIP",
        "TRAFFIC_BATCH_SIZE",
        "TRAFFIC_EXPORT_FORMAT",
        "TRAFFIC_ENGINE_PATH",
        "TRAFFIC_KALMAN_PROCESS_NOISE",
        "TRAFFIC_KALMAN_MEASUREMENT_NOISE",
        "TRAFFIC_MIN_DETECTION_FRAMES",
        "TRAFFIC_MIN_CONFIDENCE",
    ):
        val = os.environ.get(key) or ""
        if val:
            env_values[key] = val

    return env_values


def env_overrides_for_appconfig(env: dict | None = None) -> dict:
    """Converte variáveis de ambiente em kwargs compatíveis com AppConfig.

    Retorna apenas as chaves que estão presentes no dicionário de ambiente,
    permitindo que o caller aplique precedência (CLI > env > default).
    """
    if env is None:
        env = load_env_config()

    overrides: dict = {}

    if "TRAFFIC_MODEL_PATH" in env:
        overrides["model_path"] = env["TRAFFIC_MODEL_PATH"]
    if "TRAFFIC_DEFAULT_SOURCE" in env:
        overrides["source"] = env["TRAFFIC_DEFAULT_SOURCE"]
    if "TRAFFIC_TRACKER" in env:
        overrides["tracker"] = env["TRAFFIC_TRACKER"]
    if "TRAFFIC_OUTPUT_DIR" in env:
        overrides["output_dir"] = env["TRAFFIC_OUTPUT_DIR"]
    if "TRAFFIC_SKIP_FRAMES" in env:
        overrides["skip_frames"] = _parse_int(env["TRAFFIC_SKIP_FRAMES"], 1)
    if "TRAFFIC_CONF" in env:
        overrides["conf"] = _parse_float(env["TRAFFIC_CONF"], 0.25)
    if "TRAFFIC_IOU" in env:
        overrides["iou"] = _parse_float(env["TRAFFIC_IOU"], 0.5)
    if "TRAFFIC_IMG_SIZE" in env:
        overrides["imgsz"] = _parse_int(env["TRAFFIC_IMG_SIZE"], 640)
    if "TRAFFIC_DEVICE" in env:
        overrides["device"] = env["TRAFFIC_DEVICE"]
    if "TRAFFIC_ZONES_FILE" in env:
        overrides["zones_file"] = env["TRAFFIC_ZONES_FILE"]
    if "TRAFFIC_SPEED_CONFIG_FILE" in env:
        overrides["speed_file"] = env["TRAFFIC_SPEED_CONFIG_FILE"]
    if "TRAFFIC_SAVE_VIDEO" in env:
        overrides["save_video"] = _parse_bool(env["TRAFFIC_SAVE_VIDEO"], False)
    if "TRAFFIC_NO_SHOW" in env:
        overrides["show"] = not _parse_bool(env["TRAFFIC_NO_SHOW"], False)
    if "TRAFFIC_MAX_FRAMES" in env:
        overrides["max_frames"] = _parse_int(env["TRAFFIC_MAX_FRAMES"], 0)
    if "TRAFFIC_RUN_NAME" in env:
        overrides["run_name"] = env["TRAFFIC_RUN_NAME"]
    if "TRAFFIC_JSON_OUTPUT" in env:
        overrides["json_output"] = _parse_bool(env["TRAFFIC_JSON_OUTPUT"], False)
    if "TRAFFIC_REPORT_FORMAT" in env:
        overrides["report_format"] = env["TRAFFIC_REPORT_FORMAT"]
    if "TRAFFIC_ADAPTIVE_SKIP" in env:
        overrides["adaptive_skip"] = _parse_bool(env["TRAFFIC_ADAPTIVE_SKIP"], False)
    if "TRAFFIC_TARGET_FPS" in env:
        overrides["target_fps"] = _parse_float(env["TRAFFIC_TARGET_FPS"], 15.0)
    if "TRAFFIC_MIN_SKIP" in env:
        overrides["min_skip"] = _parse_int(env["TRAFFIC_MIN_SKIP"], 1)
    if "TRAFFIC_MAX_SKIP" in env:
        overrides["max_skip"] = _parse_int(env["TRAFFIC_MAX_SKIP"], 10)
    if "TRAFFIC_BATCH_SIZE" in env:
        overrides["batch_size"] = _parse_int(env["TRAFFIC_BATCH_SIZE"], 1)
    if "TRAFFIC_EXPORT_FORMAT" in env:
        overrides["export_format"] = env["TRAFFIC_EXPORT_FORMAT"]
    if "TRAFFIC_ENGINE_PATH" in env:
        overrides["engine_path"] = env["TRAFFIC_ENGINE_PATH"]
    if "TRAFFIC_KALMAN_PROCESS_NOISE" in env:
        overrides["kalman_process_noise"] = _parse_float(
            env["TRAFFIC_KALMAN_PROCESS_NOISE"], 1.0
        )
    if "TRAFFIC_KALMAN_MEASUREMENT_NOISE" in env:
        overrides["kalman_measurement_noise"] = _parse_float(
            env["TRAFFIC_KALMAN_MEASUREMENT_NOISE"], 10.0
        )
    if "TRAFFIC_MIN_DETECTION_FRAMES" in env:
        overrides["min_detection_frames"] = _parse_int(
            env["TRAFFIC_MIN_DETECTION_FRAMES"], 3
        )
    if "TRAFFIC_MIN_CONFIDENCE" in env:
        overrides["min_confidence"] = _parse_float(
            env["TRAFFIC_MIN_CONFIDENCE"], 0.3
        )

    return overrides


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
    smoothing_alpha: float = 0.35  # suavização EMA (0..1) — mantido para retrocompatibilidade
    max_speed_kmh: float = 200.0  # teto de sanidade
    min_displacement_m: float = 0.05  # abaixo disso considera parado
    window_s: float = 0.5  # janela temporal para cálculo do deslocamento
    kalman_process_noise: float = 1.0  # ruído do processo do filtro de Kalman (Q)
    kalman_measurement_noise: float = 10.0  # ruído da medição do filtro de Kalman (R)
    min_detection_frames: int = 3  # mínimo de frames consecutivos para calcular velocidade
    min_confidence: float = 0.3  # confiança média mínima para exibir velocidade


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
    adaptive_skip: bool = False
    target_fps: float = 15.0
    min_skip: int = 1
    max_skip: int = 10
    batch_size: int = 1
    export_format: Optional[str] = None  # onnx, engine, openvino, ncnn
    engine_path: Optional[str] = None  # caminho para modelo TensorRT já exportado
    min_detection_frames: int = 3  # mínimo de frames consecutivos para exibir velocidade
    min_confidence: float = 0.3  # confiança média mínima para exibir velocidade


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
