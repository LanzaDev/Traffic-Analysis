"""Script de validação de modelos YOLO para análise de tráfego.

Wrapper para o subcomando 'val' do train.py.

Exemplo de uso:

    python val.py --data config/data.yaml --model runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import sys

from train import main


def main_val() -> int:
    # Injetar 'val' como primeiro argumento se não estiver presente
    args = sys.argv[1:]
    if not args or args[0] not in ("train", "val", "create-data-yaml"):
        args = ["val"] + args
    return main(args)


if __name__ == "__main__":
    raise SystemExit(main_val())