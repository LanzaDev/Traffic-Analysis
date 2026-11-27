#!/usr/bin/env python3
"""
Validador de modelo YOLO11m treinado

Uso:
    python val.py --data seu_dataset/data.yaml --model runs/detect/traffic_yolo11m/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys

from train import train_model, validate_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Validador YOLO11m")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("val", help="Validar modelo treinado")
    p_val.add_argument("--data", required=True, help="Caminho data.yaml")
    p_val.add_argument("--model", required=True, help="Caminho modelo treinado (.pt)")
    p_val.add_argument("--imgsz", type=int, default=640, help="Tamanho imagem")
    p_val.add_argument("--device", default="0", help="Dispositivo")

    args = parser.parse_args()

    if args.command == "val":
        from train import validate_model

        validate_model(
            data=args.data,
            model=args.model,
            imgsz=args.imgsz,
            device=args.device,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())