"""Ferramenta interativa de calibração (ROI e velocidade).

Abre um frame da fonte e permite clicar pontos para definir:
    roi   - zonas poligonais (>= 3 pontos) ou linhas de travessia (2 pontos),
            salvando em config/zones.json.
    speed - 4 pontos de um trecho da pista (ordem: topo-esq, topo-dir,
            base-dir, base-esq) + dimensões reais, salvando em config/speed.json.

Uso:
    python calibrate.py roi   --source assets/traffic.mp4
    python calibrate.py speed --source assets/traffic.mp4 --at 3

Teclas (janela de vídeo):
    ROI  : clique esq = adiciona ponto | u = desfaz | r = limpa atual
           p = fechar polígono | l = fechar linha | s = salvar JSON | q = sair
    SPEED: clique esq = adiciona ponto (até 4) | r = limpa | s = salvar | q = sair
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT, resolve_path

FONT = cv2.FONT_HERSHEY_SIMPLEX


def grab_frame(source: str, at: float) -> np.ndarray:
    src = str(source)
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    elif src.lower().startswith(("rtsp://", "http://", "https://")):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    else:
        p = resolve_path(src)
        cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {source}")
    if at > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Não foi possível ler um frame da fonte.")
    return frame


class Calibrator:
    def __init__(self, mode: str, frame: np.ndarray, zones_path: Path, speed_path: Path):
        self.mode = mode
        self.base = frame
        self.points: list[list[int]] = []
        self.zones_path = zones_path
        self.speed_path = speed_path
        self.zones_data = self._load_zones()
        self.window = f"Calibracao - {mode}"

    def _load_zones(self) -> dict:
        if self.zones_path.exists():
            return json.loads(self.zones_path.read_text(encoding="utf-8"))
        return {"zones": []}

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "speed" and len(self.points) >= 4:
                return
            self.points.append([int(x), int(y)])

    def render(self) -> np.ndarray:
        img = self.base.copy()
        # Zonas já salvas (referência).
        for z in self.zones_data.get("zones", []):
            pts = np.array(z["points"], np.int32)
            color = tuple(z.get("color", [120, 120, 120]))
            if z["type"] == "polygon":
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 1)
            else:
                cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, 1)
            cv2.putText(img, z["name"], tuple(pts[0]), FONT, 0.5, color, 1)

        # Pontos atuais.
        for i, p in enumerate(self.points):
            cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6), FONT, 0.5, (0, 0, 255), 1)
        if len(self.points) >= 2:
            pts = np.array(self.points, np.int32)
            closed = self.mode == "speed" and len(self.points) == 4
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], closed, (0, 255, 255), 2)

        help_text = (
            "ROI: clique=add | u=desfaz | r=limpa | p=poligono | l=linha | s=salvar | q=sair"
            if self.mode == "roi"
            else "SPEED: clique 4 pontos (TL,TR,BR,BL) | r=limpa | s=salvar | q=sair"
        )
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(img, help_text, (8, 19), FONT, 0.5, (240, 240, 240), 1)
        return img

    def finalize_zone(self, kind: str):
        need = 3 if kind == "polygon" else 2
        if len(self.points) < need:
            print(f"[calibrate] precisa de >= {need} pontos para {kind}.")
            return
        pts = self.points[:need] if kind == "line" else self.points[:]
        name = input(f"  Nome da zona ({kind}): ").strip() or f"Zona {len(self.zones_data['zones']) + 1}"
        color = [0, 200, 255] if kind == "polygon" else [0, 255, 128]
        self.zones_data["zones"].append(
            {"name": name, "type": kind, "points": pts, "color": color}
        )
        self.points = []
        print(f"[calibrate] zona '{name}' adicionada (não esqueça de salvar com 's').")

    def save_zones(self):
        self.zones_path.write_text(
            json.dumps(self.zones_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[calibrate] zonas salvas em {self.zones_path}")

    def save_speed(self):
        if len(self.points) != 4:
            print("[calibrate] são necessários exatamente 4 pontos.")
            return
        try:
            w = float(input("  Largura real do trecho (metros) [8]: ").strip() or 8)
            h = float(input("  Comprimento real do trecho (metros) [20]: ").strip() or 20)
            ppm = float(input("  Pixels por metro na vista aérea [15]: ").strip() or 15)
        except ValueError:
            print("[calibrate] valor inválido; abortando.")
            return
        data = {
            "enabled": True,
            "source_points": self.points,
            "real_width_m": w,
            "real_height_m": h,
            "pixels_per_meter": ppm,
            "smoothing_alpha": 0.35,
            "max_speed_kmh": 200.0,
            "min_displacement_m": 0.05,
            "window_s": 0.5,
        }
        self.speed_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[calibrate] calibração de velocidade salva em {self.speed_path}")

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse)
        print("[calibrate] janela aberta. Siga as instruções na barra superior.")
        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.points = []
            elif self.mode == "roi":
                if key == ord("u") and self.points:
                    self.points.pop()
                elif key == ord("p"):
                    self.finalize_zone("polygon")
                elif key == ord("l"):
                    self.finalize_zone("line")
                elif key == ord("s"):
                    self.save_zones()
            elif self.mode == "speed" and key == ord("s"):
                self.save_speed()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Calibração interativa de ROI e velocidade.")
    p.add_argument("mode", choices=["roi", "speed"], help="O que calibrar.")
    p.add_argument("--source", default="assets/traffic.mp4", help="Fonte de vídeo.")
    p.add_argument("--at", type=float, default=3.0, help="Segundo do vídeo para capturar o frame.")
    p.add_argument("--zones", default="config/zones.json")
    p.add_argument("--speed-config", default="config/speed.json")
    args = p.parse_args()

    frame = grab_frame(args.source, args.at)
    zones_path = resolve_path(args.zones)
    speed_path = resolve_path(args.speed_config)
    zones_path.parent.mkdir(parents=True, exist_ok=True)
    Calibrator(args.mode, frame, zones_path, speed_path).run()


if __name__ == "__main__":
    main()
