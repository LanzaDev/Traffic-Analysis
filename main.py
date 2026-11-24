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
import sys

from src.config import AppConfig, MODES, MODE_LABELS, env_overrides_for_appconfig, load_env_config
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
                   help="Modelo YOLO (Ultralytics). Compatível com yolov8n.pt/yolo11n.pt.")
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
    p.add_argument("--no-show", action="store_true", help="Não exibe janela (modo headless).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limita o nº de frames processados (0 = todos). Útil para testes.")
    p.add_argument("--run-name", default=None, help="Nome base para os arquivos de saída.")
    p.add_argument("--json", action="store_true",
                   help="Imprime o relatório final em JSON no stdout (útil para encanear com outras ferramentas).")
    p.add_argument("--report-format", default=None, choices=["pretty", "compact"],
                   help="Formato da saída JSON no stdout (ignorado se --json não for usado).")
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
            args.speed_config
            if hasattr(args, "speed_config") and args.speed_config is not None
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
            args.speed_config
            if hasattr(args, "speed_config") and args.speed_config is not None
            else env_overrides.get("speed_file", "config/speed.json")
        )

    cfg = AppConfig(
        source=source,
        mode=mode,
        model_path=args.model if args.model is not None else env_overrides.get("model_path", "models/yolo11n.pt"),
        conf=args.conf if args.conf is not None else env_overrides.get("conf", 0.25),
        iou=args.iou if args.iou is not None else env_overrides.get("iou", 0.5),
        imgsz=args.imgsz if args.imgsz is not None else env_overrides.get("imgsz", 640),
        device=args.device if args.device is not None else env_overrides.get("device", "cpu"),
        tracker=args.tracker if args.tracker is not None else env_overrides.get("tracker", "bytetrack"),
        skip_frames=skip_frames,
        zones_file=args.zones if args.zones is not None else env_overrides.get("zones_file", "config/zones.json"),
        speed_file=speed_config,
        output_dir=args.output if args.output is not None else env_overrides.get("output_dir", "output"),
        save_video=save_video,
        show=(not args.no_show) if args.no_show else env_overrides.get("show", True),
        max_frames=args.max_frames if args.max_frames is not None else env_overrides.get("max_frames", 0),
        run_name=args.run_name if args.run_name is not None else env_overrides.get("run_name", ""),
        json_output=args.json or env_overrides.get("json_output", False),
        report_format=args.report_format if args.report_format is not None else env_overrides.get("report_format", "pretty"),
    )

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
