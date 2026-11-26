"""Ponto de entrada: Rastreamento e Análise de Tráfego.

Oferece seleção de funcionalidade por subcomandos CLI (argparse) ou por menu
interativo no terminal (quando nenhum subcomando é informado).

Subcomandos:
    people   - Contagem de Pessoas (classe 'person')
    vehicles - Contagem de Veículos (car, truck, bus, motorcycle/motorbike)
    speed    - Estimação de Velocidade de Veículos (km/h)
    full     - Execução Completa (contagem + velocidade simultaneamente)

Exemplos:
    python main.py people   --source assets/traffic.mp4
    python main.py vehicles --source assets/traffic.mp4 --save-video
    python main.py speed    --source assets/traffic.mp4 --speed-config config/speed.json
    python main.py full     --source assets/traffic.mp4 --skip-frames 2 --save-video
    python main.py                          # abre o menu interativo
    python main.py vehicles --help          # ajuda específica do modo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.config import AppConfig, MODES, MODE_LABELS, env_overrides_for_appconfig, load_env_config, classes_for_mode, has_display
from src.runner import AnalysisRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _common_parser() -> argparse.ArgumentParser:
    """Parser com argumentos compartilhados por todos os modos.

    Argumentos que podem vir de variáveis de ambiente usam default=None para
    que possamos distinguir "não informado pelo CLI" de "valor padrão do código".
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--source", default=None,
                   help="Caminho de vídeo (.mp4/.avi), URL RTSP/HTTP ou índice de webcam (0,1,...).")
    p.add_argument("--model", default=None,
                   help="Modelo YOLO (Ultralytics). Compatível com yolo11n.pt, yolo_traffic.pt, etc.")
    p.add_argument("--conf", type=float, default=None, help="Confiança mínima de detecção.")
    p.add_argument("--iou", type=float, default=None, help="Limiar de IoU para NMS.")
    p.add_argument("--imgsz", type=int, default=None, help="Tamanho de inferência (pixels).")
    p.add_argument("--device", default=None, help="Dispositivo: cpu, 0 (GPU), ou auto.")
    p.add_argument("--tracker", default=None, choices=["bytetrack", "botsort"],
                   help="Algoritmo de rastreamento multi-objeto.")
    p.add_argument("--skip-frames", type=int, default=None,
                   help="Processa 1 a cada N frames (desempenho em tempo real/CCTV).")
    p.add_argument("--zones", default=None,
                   help="JSON de zonas ROI (polígonos/linhas).")
    p.add_argument("--output", default=None, help="Diretório de saída (vídeo/relatórios).")
    p.add_argument("--save-video", action="store_true",
                   help="Grava o vídeo anotado em output/.")
    p.add_argument("--show", action="store_const", const=True, default=None,
                   help="Força exibição de janela (útil quando há display detectado como headless).")
    p.add_argument("--no-show", action="store_const", const=True, default=None,
                   help="Não exibe janela (modo headless).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limita o nº de frames processados (0 = todos). Útil para testes.")
    p.add_argument("--run-name", default=None, help="Nome base para os arquivos de saída.")
    p.add_argument("--json", action="store_true",
                   help="Imprime o relatório final em JSON no stdout (útil para encanear com outras ferramentas).")
    p.add_argument("--report-format", default=None, choices=["pretty", "compact"],
                   help="Formato da saída JSON no stdout (ignorado se --json não for usado).")
    p.add_argument("--adaptive-skip", action="store_true",
                   help="Ajusta automaticamente o salto de frames para manter o FPS alvo.")
    p.add_argument("--target-fps", type=float, default=None,
                   help="FPS alvo para o skip adaptativo (padrão 15.0).")
    p.add_argument("--min-skip", type=int, default=None,
                   help="Salto mínimo no modo adaptativo (padrão 1).")
    p.add_argument("--max-skip", type=int, default=None,
                   help="Salto máximo no modo adaptativo (padrão 10).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Tamanho do lote para inferência em batch (padrão 1 = sem batch).")
    p.add_argument("--min-detection-frames", type=int, default=None,
                   help="Mínimo de frames consecutivos para exibir velocidade (padrão 3).")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="Confiança média mínima para exibir velocidade (padrão 0.3).")
    p.add_argument("--min-stopped-s", type=float, default=None,
                   help="Tempo mínimo (s) para considerar veículo parado (padrão 2.0).")
    p.add_argument("--reid-enabled", action="store_true",
                   help="Habilitar re-identificação de objetos (evitar ID duplicado quando objeto reaparece).")
    p.add_argument("--reid-max-distance-m", type=float, default=None,
                   help="Distância máxima em metros para re-ID (padrão 15.0).")
    p.add_argument("--reid-max-time-s", type=float, default=None,
                   help="Tempo máximo (s) para considerar objeto desaparecido (padrão 5.0).")
    p.add_argument("--reid-color-tolerance", type=float, default=None,
                   help="Tolerância de diferença de cor para re-ID 0-1 (padrão 0.3).")
    p.add_argument("--export", default=None, choices=["onnx", "engine", "openvino", "ncnn"],
                   help="Exporta o modelo para o formato especificado e sai.")
    p.add_argument("--engine", default=None,
                   help="Caminho para modelo TensorRT já exportado (.engine).")
    p.add_argument("--eval", action="store_true",
                   help="Modo de avaliação: calcula métricas (mAP, precisão, recall, F1, erro de contagem) comparando com ground truth.")
    p.add_argument("--gt", default=None,
                   help="Caminho para arquivo de ground truth (COCO JSON ou YOLO TXT) para avaliação.")
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rastreamento e Análise de Tráfego (YOLO + ByteTrack).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", help="Funcionalidade a executar.")

    common = _common_parser()

    # people — contagem de pessoas, sem calibração de velocidade.
    sp_people = sub.add_parser(
        "people", parents=[common], help="Contagem de Pessoas (classe person)."
    )
    sp_people.set_defaults(mode="people")

    # vehicles — contagem de veículos, sem calibração de velocidade.
    sp_vehicles = sub.add_parser(
        "vehicles",
        parents=[common],
        help="Contagem de Veículos (car, truck, bus, motorcycle/motorbike).",
    )
    sp_vehicles.set_defaults(mode="vehicles")

    # speed — estimação de velocidade, requer calibração.
    sp_speed = sub.add_parser(
        "speed",
        parents=[common],
        help="Estimação de Velocidade de Veículos (km/h).",
    )
    sp_speed.add_argument(
        "--speed-config",
        default=None,
        help="JSON de calibração de velocidade (perspectiva).",
    )
    sp_speed.set_defaults(mode="speed")

    # full — contagem + velocidade simultaneamente.
    sp_full = sub.add_parser(
        "full",
        parents=[common],
        help="Execução Completa (contagem + velocidade).",
    )
    sp_full.add_argument(
        "--speed-config",
        default=None,
        help="JSON de calibração de velocidade (perspectiva).",
    )
    sp_full.set_defaults(mode="full")

    return p


def interactive_menu() -> dict:
    """Menu de seleção no terminal; retorna overrides para o AppConfig."""
    print("\n" + "=" * 64)
    print("  RASTREAMENTO E ANÁLISE DE TRÁFEGO")
    print("=" * 64)
    print("  Selecione a funcionalidade:")
    print("    1 - Contagem de Pessoas")
    print("    2 - Contagem de Veículos")
    print("    3 - Estimação de Velocidade de Veículos (km/h)")
    print("    4 - Execução Completa (contagem + velocidade)")
    print("=" * 64)

    choice = input("  Opção [1-4] (padrão 4): ").strip() or "4"
    mode = {"1": "people", "2": "vehicles", "3": "speed", "4": "full"}.get(choice, "full")

    source = input(f"  Fonte de vídeo (padrão assets/traffic.mp4): ").strip()
    source = source or "assets/traffic.mp4"

    skip = input("  Salto de frames / frame skipping (padrão 1): ").strip()
    try:
        skip_frames = max(1, int(skip)) if skip else 1
    except ValueError:
        skip_frames = 1

    save = input("  Gravar vídeo anotado? [s/N]: ").strip().lower()
    save_video = save in ("s", "sim", "y", "yes")

    print(f"\n  -> Modo: {MODE_LABELS[mode]} | Fonte: {source} | skip: {skip_frames}\n")
    return {"mode": mode, "source": source, "skip_frames": skip_frames, "save_video": save_video}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Carrega variáveis de ambiente (.env + SO).
    env_overrides = env_overrides_for_appconfig(load_env_config())

    # Precedência: código default → env → CLI (CLI vence sempre que informado).

    # Nenhum subcomando informado -> menu interativo.
    if getattr(args, "command", None) is None:
        overrides = interactive_menu()
        mode = overrides["mode"]
        source = overrides["source"]
        skip_frames = overrides["skip_frames"]
        save_video = overrides["save_video"]
        speed_config = (
            getattr(args, "speed_config", None)
            if getattr(args, "speed_config", None) is not None
            else env_overrides.get("speed_file", "config/speed.json")
        )
    else:
        mode = args.mode
        # CLI vence env, que vence default do código.
        source = args.source if args.source is not None else env_overrides.get("source", "assets/traffic.mp4")
        skip_frames = (
            args.skip_frames if args.skip_frames is not None else env_overrides.get("skip_frames", 1)
        )
        # Para flags booleanas, só aplicamos se o usuário passou explicitamente (True).
        save_video = args.save_video or env_overrides.get("save_video", False)
        speed_config = (
            getattr(args, "speed_config", None)
            if getattr(args, "speed_config", None) is not None
            else env_overrides.get("speed_file", "config/speed.json")
        )

    # Resolução de show com precedência: CLI > env > detecção automática de display.
    cli_show = None
    if getattr(args, "show", None) is True:
        cli_show = True
    elif getattr(args, "no_show", None) is True:
        cli_show = False

    if cli_show is not None:
        show = cli_show
    elif "TRAFFIC_NO_SHOW" in os.environ:
        env_val = os.environ["TRAFFIC_NO_SHOW"].strip().lower()
        show = env_val not in ("1", "true", "yes", "y", "sim", "on")
    else:
        show = has_display()

    cfg = AppConfig(
        source=source,
        mode=mode,
        model_path=getattr(args, "model", None)
        if getattr(args, "model", None) is not None
        else env_overrides.get("model_path", "models/yolo11m.pt"),
        conf=getattr(args, "conf", None)
        if getattr(args, "conf", None) is not None
        else env_overrides.get("conf", 0.25),
        iou=getattr(args, "iou", None)
        if getattr(args, "iou", None) is not None
        else env_overrides.get("iou", 0.5),
        imgsz=getattr(args, "imgsz", None)
        if getattr(args, "imgsz", None) is not None
        else env_overrides.get("imgsz", 640),
        device=getattr(args, "device", None)
        if getattr(args, "device", None) is not None
        else env_overrides.get("device", "cpu"),
        tracker=getattr(args, "tracker", None)
        if getattr(args, "tracker", None) is not None
        else env_overrides.get("tracker", "bytetrack"),
        skip_frames=skip_frames,
        zones_file=getattr(args, "zones", None)
        if getattr(args, "zones", None) is not None
        else env_overrides.get("zones_file", "config/zones.json"),
        speed_file=speed_config,
        output_dir=getattr(args, "output", None)
        if getattr(args, "output", None) is not None
        else env_overrides.get("output_dir", "output"),
        save_video=save_video,
        show=show,
        max_frames=getattr(args, "max_frames", None)
        if getattr(args, "max_frames", None) is not None
        else env_overrides.get("max_frames", 0),
        run_name=getattr(args, "run_name", None)
        if getattr(args, "run_name", None) is not None
        else env_overrides.get("run_name", ""),
        json_output=getattr(args, "json", False) or env_overrides.get("json_output", False),
        report_format=getattr(args, "report_format", None)
        if getattr(args, "report_format", None) is not None
        else env_overrides.get("report_format", "pretty"),
        adaptive_skip=getattr(args, "adaptive_skip", False) or env_overrides.get("adaptive_skip", False),
        target_fps=getattr(args, "target_fps", None)
        if getattr(args, "target_fps", None) is not None
        else env_overrides.get("target_fps", 15.0),
        min_skip=getattr(args, "min_skip", None)
        if getattr(args, "min_skip", None) is not None
        else env_overrides.get("min_skip", 1),
        max_skip=getattr(args, "max_skip", None)
        if getattr(args, "max_skip", None) is not None
        else env_overrides.get("max_skip", 10),
        batch_size=getattr(args, "batch_size", None)
        if getattr(args, "batch_size", None) is not None
        else env_overrides.get("batch_size", 1),
        export_format=getattr(args, "export", None)
        if getattr(args, "export", None) is not None
        else env_overrides.get("export_format"),
        engine_path=getattr(args, "engine", None)
        if getattr(args, "engine", None) is not None
        else env_overrides.get("engine_path"),
        min_detection_frames=getattr(args, "min_detection_frames", None)
        if getattr(args, "min_detection_frames", None) is not None
        else env_overrides.get("min_detection_frames", 3),
        min_confidence=getattr(args, "min_confidence", None)
        if getattr(args, "min_confidence", None) is not None
        else env_overrides.get("min_confidence", 0.3),
        min_stopped_s=getattr(args, "min_stopped_s", None)
        if getattr(args, "min_stopped_s", None) is not None
        else env_overrides.get("min_stopped_s", 2.0),
        eval_mode=getattr(args, "eval", False) or env_overrides.get("eval_mode", False),
        gt_path=getattr(args, "gt", None)
        if getattr(args, "gt", None) is not None
        else env_overrides.get("gt_path"),
        reid_enabled=getattr(args, "reid_enabled", False)
        or env_overrides.get("reid_enabled", True),
        reid_max_distance_m=getattr(args, "reid_max_distance_m", None)
        if getattr(args, "reid_max_distance_m", None) is not None
        else env_overrides.get("reid_max_distance_m", 15.0),
        reid_max_time_s=getattr(args, "reid_max_time_s", None)
        if getattr(args, "reid_max_time_s", None) is not None
        else env_overrides.get("reid_max_time_s", 5.0),
        reid_color_tolerance=getattr(args, "reid_color_tolerance", None)
        if getattr(args, "reid_color_tolerance", None) is not None
        else env_overrides.get("reid_color_tolerance", 0.3),
    )

    # Se solicitado, exporta o modelo para o formato desejado e encerra.
    if cfg.export_format:
        from src.detector import Detector
        det = Detector(
            model_path=cfg.engine_path if cfg.engine_path else cfg.model_path,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            device=cfg.device,
            tracker=cfg.tracker,
            classes=classes_for_mode(cfg.mode),
        )
        out_path = det.export_model(cfg.export_format)
        print(f"[export] modelo exportado em: {out_path}")
        return 0

    try:
        AnalysisRunner(cfg).run()
    except FileNotFoundError as e:
        print(f"[erro] {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"[erro] {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
