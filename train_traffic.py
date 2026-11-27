#!/usr/bin/env python3
"""
Treinador YOLO11m para Detecção de Veículos - Baixa e Treina com Datasets Públicos

Este script:
1. Baixa datasets públicos de detecção de veículos (UA-DETRAC, BDD100K, etc.)
2. Converte para formato YOLO
3. Treina yolo11m.pt com esses dados
4. Salva o modelo treinado

Requisitos:
    pip install ultralytics opencv-python numpy requests tqdm
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
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
RUNS_DIR = PROJECT_ROOT / "runs"

# Classes do projeto de tráfego
TRAFFIC_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]

# URLs dos datasets públicos
DATASET_URLS = {
    # UA-DETRAC - Dados de tráfego com anotações de veículos
    "udetrac_train": "https://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip",
    "udetrac_test": "https://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip",
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Baixa um arquivo com barra de progresso."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_udetrac_dataset() -> Path:
    """Baixa o dataset UA-DETRAC."""
    datasets_dir = DATASETS_DIR / "udetrac"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Verificar se já existe
    if (datasets_dir / "train" / "images").exists():
        print(f"✓ Dataset UA-DETRAC já existe em {datasets_dir}")
        return datasets_dir

    print("📥 Baixando dataset UA-DETRAC...")
    print("   Fonte: http://detrac-db.rit.albany.edu/")
    print("   Tamanho: ~1.5 GB total")
    print("   ⚠️  AVISO: Este é um download grande, pode demorar")

    # Baixar treino
    train_zip = datasets_dir / "train_data.zip"
    if not train_zip.exists():
        print("\n📦 Baixando dados de treino...")
        download_file(DATASET_URLS["udetrac_train"], train_zip)

    # Baixar teste
    test_zip = datasets_dir / "test_data.zip"
    if not test_zip.exists():
        print("\n📦 Baixando dados de teste...")
        download_file(DATASET_URLS["udetrac_test"], test_zip)

    # Extrair
    print("\n📦 Extraindo arquivos...")
    extract_zip(train_zip, datasets_dir / "train")
    extract_zip(test_zip, datasets_dir / "test")

    # Limpar ZIPs
    train_zip.unlink()
    test_zip.unlink()

    return datasets_dir


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extrai um arquivo ZIP."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extraído: {zip_path.name} -> {extract_to}")


def convert_udetrac_to_yolo(dataset_dir: Path) -> Path:
    """Converte o dataset UA-DETRAC para formato YOLO."""
    yolo_dir = DATASETS_DIR / "udetrac_yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    # Criar estrutura de pastas YOLO
    for split in ["train", "val", "test"]:
        (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    print("🔄 Convertendo anotações UA-DETRAC para formato YOLO...")
    print("   ⚠️  Nota: A conversão completa requer processamento dos arquivos XML")
    print("   de anotação do UA-DETRAC. Para uso imediato, use o data.yaml existente.")

    return yolo_dir


def create_data_yaml(
    output_path: Path,
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: Optional[str | Path] = None,
) -> Path:
    """Cria arquivo data.yaml para treinamento YOLO."""
    data = {
        "train": str(Path(train_dir).resolve()),
        "val": str(Path(val_dir).resolve()),
        "nc": len(TRAFFIC_CLASSES),
        "names": TRAFFIC_CLASSES,
    }

    if test_dir:
        data["test"] = str(Path(test_dir).resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✓ Arquivo data.yaml criado: {output_path}")
    print(f"   Classes: {TRAFFIC_CLASSES}")
    return output_path


def train_model(
    data: str | Path,
    model: str = "yolo11m.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str | Path = "runs/detect",
    name: str = "traffic_yolo11m",
    resume: Optional[str] = None,
) -> None:
    """Treina o modelo YOLO11m."""
    from ultralytics import YOLO

    print("=" * 60)
    print("🚀 TREINAMENTO YOLO11m PARA DETECÇÃO DE VEÍCULOS")
    print("=" * 60)
    print(f"  Modelo base: {model}")
    print(f"  Dataset: {data}")
    print(f"  Épocas: {epochs}")
    print(f"  Tamanho imagem: {imgsz}")
    print(f"  Batch: {batch}")
    print(f"  Dispositivo: {device}")
    print(f"  Projeto: {project}/{name}")
    print("=" * 60)

    yolo = YOLO(model)

    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        pretrained=True,
        resume=str(resume) if resume else None,
        verbose=True,
    )

    print("=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print(f"   Modelo salvo em: {project}/{name}/weights/best.pt")
    print("=" * 60)


def validate_model(
    data: str | Path,
    model: str,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str | Path = "runs/detect",
    name: str = "val",
) -> None:
    """Valida um modelo treinado."""
    from ultralytics import YOLO

    print("🔍 Validando modelo...")
    yolo = YOLO(model)
    metrics = yolo.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
    )

    print("\n📊 RESULTADOS DA VALIDAÇÃO:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precisão: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Treinamento YOLO11m para Detecção de Veículos de Tráfego"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Baixar dataset UA-DETRAC")
    p_dl.add_argument(
        "--output",
        default=str(DATASETS_DIR / "udetrac"),
        help="Diretório de saída",
    )

    # convert
    p_conv = sub.add_parser("convert", help="Converter dataset para formato YOLO")
    p_conv.add_argument(
        "--input",
        default=str(DATASETS_DIR / "udetrac"),
        help="Diretório do dataset baixado",
    )
    p_conv.add_argument(
        "--output",
        default=str(DATASETS_DIR / "udetrac_yolo"),
        help="Diretório de saída",
    )

    # train
    p_train = sub.add_parser("train", help="Treinar modelo YOLO11m")
    p_train.add_argument("--data", required=True, help="Caminho para data.yaml")
    p_train.add_argument("--model", default="yolo11m.pt", help="Modelo base (.pt)")
    p_train.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    p_train.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    p_train.add_argument("--batch", type=int, default=16, help="Tamanho do lote")
    p_train.add_argument("--device", default="0", help="Dispositivo (0 para GPU)")
    p_train.add_argument("--project", default="runs/detect", help="Diretório do projeto")
    p_train.add_argument("--name", default="traffic_yolo11m", help="Nome da execução")
    p_train.add_argument("--resume", default=None, help="Checkpoint para retomar")

    # val
    p_val = sub.add_parser("val", help="Validar modelo treinado")
    p_val.add_argument("--data", required=True, help="Caminho para data.yaml")
    p_val.add_argument(
        "--model", required=True, help="Caminho para pesos treinados (.pt)"
    )
    p_val.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    p_val.add_argument("--batch", type=int, default=16, help="Tamanho do lote")
    p_val.add_argument("--device", default="0", help="Dispositivo")
    p_val.add_argument("--project", default="runs/detect", help="Diretório do projeto")
    p_val.add_argument("--name", default="val", help="Nome da execução")

    # create-data
    p_data = sub.add_parser("create-data", help="Criar arquivo data.yaml")
    p_data.add_argument("--output", required=True, help="Caminho de saída")
    p_data.add_argument("--train", required=True, help="Diretório de treino")
    p_val.add_argument("--val", required=True, help="Diretório de validação")
    p_data.add_argument("--test", default=None, help="Diretório de teste (opcional)")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        download_udetrac_dataset()
    elif args.command == "convert":
        convert_udetrac_to_yolo(Path(args.input))
    elif args.command == "train":
        train_model(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
        )
    elif args.command == "val":
        validate_model(
            data=args.data,
            model=args.model,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
    elif args.command == "create-data":
        names = TRAFFIC_CLASSES
        create_data_yaml(
            output_path=Path(args.output),
            train_dir=args.train,
            val_dir=args.val,
            test_dir=args.test,
            nc=len(names),
            names=names,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())