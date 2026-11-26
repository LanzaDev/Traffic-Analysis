"""Script de treinamento/fine-tuning de modelos YOLO para análise de tráfego.

Permite treinar yolo11m.pt (ou qualquer modelo Ultralytics) com datasets
no formato YAML do Ultralytics, incluindo UA-DETRAC, BDD100K, VisDrone
e imagens próprias de câmeras rodoviárias.

Exemplos de uso:

    # Treinar yolo11m.pt com dataset próprio
    python train.py --data config/data.yaml --model models/yolo11m.pt --epochs 100 --imgsz 640

    # Treinar com validação
    python train.py --data config/data.yaml --epochs 50 --imgsz 640 --val

    # Validar um modelo já treinado
    python train.py --data config/data.yaml --model runs/detect/train/weights/best.pt --val-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

# Classes da concessionária (mapeamento para COCO quando usar modelo padrão)
CONCESSIONARIA_CLASSES = {
    0: "pedestrian",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    70: "articulated_truck",  # ID sintético para modelo customizado
}

CONCESSIONARIA_NAMES = [
    "pedestrian",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "articulated_truck",
]


def build_data_yaml(
    output_path: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: Optional[str | Path] = None,
    nc: int = 6,
    names: Optional[list[str]] = None,
) -> Path:
    """Gera um arquivo data.yaml no formato Ultralytics.

    Parameters
    ----------
    output_path : str | Path
        Caminho onde salvar o data.yaml.
    train_dir : str | Path
        Diretório com imagens de treino.
    val_dir : str | Path
        Diretório com imagens de validação.
    test_dir : str | Path, optional
        Diretório com imagens de teste.
    nc : int
        Número de classes.
    names : list[str], optional
        Lista de nomes das classes.

    Returns
    -------
    Path
        Caminho do arquivo gerado.
    """
    if names is None:
        names = CONCESSIONARIA_NAMES[:nc]

    data = {
        "train": str(Path(train_dir).resolve()),
        "val": str(Path(val_dir).resolve()),
        "nc": nc,
        "names": names,
    }

    if test_dir is not None:
        data["test"] = str(Path(test_dir).resolve())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(f"{k}: {v}" if not isinstance(v, list) else f"{k}: {v}"
                  for k, v in data.items()),
        encoding="utf-8",
    )
    return out


def train(
    data: str | Path,
    model: str | Path = "yolo11m.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str | Path = "runs/detect",
    name: str = "train",
    pretrained: bool = True,
    resume: Optional[str | Path] = None,
) -> None:
    """Treina um modelo YOLO com Ultralytics.

    Parameters
    ----------
    data : str | Path
        Caminho para arquivo data.yaml.
    model : str | Path
        Caminho para pesos pré-treinados (.pt) ou nome do modelo Ultralytics.
    epochs : int
        Número de épocas.
    imgsz : int
        Tamanho da imagem de entrada.
    batch : int
        Tamanho do lote.
    device : str
        Dispositivo (ex: "0" para GPU 0, "cpu" para CPU).
    project : str | Path
        Diretório do projeto para salvar resultados.
    name : str
        Nome da execução.
    pretrained : bool
        Se True, carrega pesos pré-treinados.
    resume : str | Path, optional
        Caminho para checkpoint para retomar treino.
    """
    print(f"[train] Iniciando treinamento com modelo: {model}")
    print(f"[train] Dados: {data}")
    print(f"[train] Épocas: {epochs}, ImgSz: {imgsz}, Batch: {batch}, Device: {device}")

    yolo = YOLO(model)  # carrega modelo pré-treinado ou checkpoint

    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        pretrained=pretrained,
        resume=str(resume) if resume else None,
    )

    print(f"[train] Treinamento concluído. Resultados em: {project}/{name}")


def val(
    data: str | Path,
    model: str | Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str | Path = "runs/detect",
    name: str = "val",
) -> None:
    """Valida um modelo YOLO treinado.

    Parameters
    ----------
    data : str | Path
        Caminho para arquivo data.yaml.
    model : str | Path
        Caminho para pesos treinados (.pt).
    imgsz : int
        Tamanho da imagem.
    batch : int
        Tamanho do lote.
    device : str
        Dispositivo.
    project : str | Path
        Diretório do projeto.
    name : str
        Nome da execução.
    """
    print(f"[val] Validando modelo: {model}")
    print(f"[val] Dados: {data}")

    yolo = YOLO(model)
    metrics = yolo.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
    )

    print(f"[val] Validação concluída. Resultados em: {project}/{name}")
    print(f"[val] mAP50: {metrics.box.map50:.4f}")
    print(f"[val] mAP50-95: {metrics.box.map:.4f}")
    print(f"[val] Precision: {metrics.box.mp:.4f}")
    print(f"[val] Recall: {metrics.box.mr:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Treinamento e validação de modelos YOLO para análise de tráfego."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Treinar modelo YOLO")
    p_train.add_argument("--data", required=True, help="Caminho para data.yaml")
    p_train.add_argument("--model", default="yolo11m.pt", help="Modelo base (.pt ou nome Ultralytics)")
    p_train.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    p_train.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    p_train.add_argument("--batch", type=int, default=16, help="Tamanho do lote")
    p_train.add_argument("--device", default="0", help="Dispositivo (0 para GPU, cpu para CPU)")
    p_train.add_argument("--project", default="runs/detect", help="Diretório do projeto")
    p_train.add_argument("--name", default="train", help="Nome da execução")
    p_train.add_argument("--resume", default=None, help="Checkpoint para retomar treino")

    # val
    p_val = sub.add_parser("val", help="Validar modelo YOLO")
    p_val.add_argument("--data", required=True, help="Caminho para data.yaml")
    p_val.add_argument("--model", required=True, help="Caminho para pesos treinados (.pt)")
    p_val.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    p_val.add_argument("--batch", type=int, default=16, help="Tamanho do lote")
    p_val.add_argument("--device", default="0", help="Dispositivo (0 para GPU, cpu para CPU)")
    p_val.add_argument("--project", default="runs/detect", help="Diretório do projeto")
    p_val.add_argument("--name", default="val", help="Nome da execução")

    # create-data-yaml
    p_data = sub.add_parser("create-data-yaml", help="Gerar arquivo data.yaml")
    p_data.add_argument("--output", required=True, help="Caminho de saída do data.yaml")
    p_data.add_argument("--train", required=True, help="Diretório de treino")
    p_data.add_argument("--val", required=True, help="Diretório de validação")
    p_data.add_argument("--test", default=None, help="Diretório de teste (opcional)")
    p_data.add_argument("--nc", type=int, default=6, help="Número de classes")
    p_data.add_argument("--names", default=None, help="Nomes das classes (JSON ou lista separada por vírgula)")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        train(
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
        val(
            data=args.data,
            model=args.model,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
    elif args.command == "create-data-yaml":
        names: Optional[list[str]] = None
        if args.names is not None:
            try:
                names = json.loads(args.names)
                if not isinstance(names, list):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                names = [n.strip() for n in args.names.split(",") if n.strip()]

        build_data_yaml(
            output_path=args.output,
            train_dir=args.train,
            val_dir=args.val,
            test_dir=args.test,
            nc=args.nc,
            names=names,
        )
        print(f"[create-data-yaml] Arquivo gerado: {args.output}")
    else:
        print(f"Comando desconhecido: {args.command}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())