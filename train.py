#!/usr/bin/env python3
"""
Treinador Simples YOLO11m - Baixa dataset público e treina

Uso:
    python train.py --epochs 50 --imgsz 640 --batch 16 --device 0

Para treinar com seus próprios dados:
    python train.py --data seu_dataset/data.yaml --epochs 100 --device 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm
from ultralytics import YOLO

# Diretórios
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

# Classes do projeto
TRAFFIC_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]

# Dataset público pequeno
DATASET_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"


def download_file(url: str, output_path: Path) -> None:
    """Baixa arquivo com barra de progresso."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name, total=total_size, unit="B", unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extrai arquivo ZIP."""
    import zipfile

    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extraído: {zip_path.name}")


def download_dataset() -> Path:
    """Baixa dataset público pequeno."""
    datasets_dir = DATASETS_DIR / "sample"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    zip_path = datasets_dir / "coco128.zip"

    if not zip_path.exists():
        print("📥 Baixando dataset COCO128 (pequeno, para teste rápido)...")
        print("   Fonte: https://github.com/ultralytics/yolov5")
        download_file(DATASET_URL, zip_path)

    extract_dir = datasets_dir / "coco128"
    if not (extract_dir / "images" / "train").exists():
        print("📦 Extraindo...")
        extract_zip(zip_path, datasets_dir)

    return extract_dir


def create_data_yaml(output_path: Path, train_dir: str | Path, val_dir: str | Path) -> Path:
    """Cria arquivo data.yaml."""
    import yaml

    data = {
        "train": str(Path(train_dir).resolve()),
        "val": str(Path(val_dir).resolve()),
        "nc": len(TRAFFIC_CLASSES),
        "names": TRAFFIC_CLASSES,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✓ data.yaml criado: {output_path}")
    return output_path


def train_model(
    data: str | Path,
    model: str = "yolo11m.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
) -> None:
    """Treina o modelo YOLO11m."""
    print("=" * 60)
    print("🚀 TREINAMENTO YOLO11m PARA VEÍCULOS")
    print("=" * 60)
    print(f"  Modelo: {model}")
    print(f"  Dataset: {data}")
    print(f"  Épocas: {epochs}")
    print(f"  Imgsz: {imgsz}")
    print(f"  Batch: {batch}")
    print(f"  Device: {device}")
    print("=" * 60)

    yolo = YOLO(model)

    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR / "detect"),
        name="traffic_yolo11m",
        pretrained=True,
        verbose=True,
    )

    print("=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print(f"   Melhor modelo: runs/detect/traffic_yolo11m/weights/best.pt")
    print("=" * 60)


def validate_model(data: str | Path, model: str, imgsz: int = 640, device: str = "0") -> None:
    """Valida modelo treinado."""
    print("🔍 Validando modelo...")
    yolo = YOLO(model)
    metrics = yolo.val(data=str(data), imgsz=imgsz, batch=16, device=device)

    print("\n📊 RESULTADOS:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precisão: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Treinador Simples YOLO11m")
    sub = p.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Baixar dataset COCO128")

    # train
    p_train = sub.add_parser("train", help="Treinar modelo")
    p_train.add_argument("--data", required=True, help="Caminho data.yaml")
    p_train.add_argument("--model", default="yolo11m.pt", help="Modelo base")
    p_train.add_argument("--epochs", type=int, default=50, help="Épocas")
    p_train.add_argument("--imgsz", type=int, default=640, help="Tamanho imagem")
    p_train.add_argument("--batch", type=int, default=16, help="Batch size")
    p_train.add_argument("--device", default="0", help="Dispositivo")

    # val
    p_val = sub.add_parser("val", help="Validar modelo")
    p_val.add_argument("--data", required=True, help="Caminho data.yaml")
    p_val.add_argument("--model", required=True, help="Caminho modelo treinado (.pt)")
    p_val.add_argument("--imgsz", type=int, default=640, help="Tamanho imagem")
    p_val.add_argument("--device", default="0", help="Dispositivo")

    # create-data
    p_data = sub.add_parser("create-data", help="Criar data.yaml")
    p_data.add_argument("--output", required=True, help="Caminho saída")
    p_data.add_argument("--train", required=True, help="Diretório treino")
    p_data.add_argument("--val", required=True, help="Diretório validação")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        download_dataset()
    elif args.command == "train":
        train_model(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    elif args.command == "val":
        validate_model(
            data=args.data,
            model=args.model,
            imgsz=args.imgsz,
            device=args.device,
        )
    elif args.command == "create-data":
        create_data_yaml(
            output_path=Path(args.output),
            train_dir=args.train,
            val_dir=args.val,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())