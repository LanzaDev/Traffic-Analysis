# Rastreamento e Análise de Tráfego 🚗🚶

Projeto modular de **Visão Computacional** para contagem de pessoas/veículos e
estimação de velocidade (km/h) a partir de **vídeos locais** ou **streams de
câmeras CCTV (IP/RTSP)**.

Construído sobre a base do `yolo_project` existente (o vídeo `traffic.mp4`),
evoluindo de scripts simples de detecção para uma arquitetura completa de
**detecção + rastreamento + zonas ROI + contagem + velocidade**.

---

## ✨ Funcionalidades (menu de seleção)

| Modo | Descrição | Classes filtradas |
|------|-----------|-------------------|
| `people`   | 1. Contagem de Pessoas | `person` |
| `vehicles` | 2. Contagem de Veículos | `car`, `truck`, `bus`, `motorcycle` (motorbike) |
| `speed`    | 3. Estimação de Velocidade (km/h) | veículos |
| `full`     | 4. Execução Completa (contagem + velocidade) | pessoas + veículos |

A seleção pode ser feita por **argumentos CLI (`argparse`)** ou por **menu
interativo no terminal** (basta rodar sem `--mode`).

---

## 🧱 Arquitetura

```
traffic_analysis/
├── main.py                 # Entrada: CLI argparse + menu interativo
├── calibrate.py            # Ferramenta interativa de calibração (ROI / velocidade)
├── run.bat                 # Launcher (usa o venv local)
├── requirements.txt
├── config/
│   ├── zones.json          # Zonas ROI (polígonos e linhas)
│   └── speed.json          # Calibração de velocidade (perspectiva)
├── models/
│   └── yolo_traffic_classes.pt # Modelo YOLOv8 treinado para veículos com classes separadas (auto, bus, car, lcv, motorcycle, multiaxle, tractor, truck)
├── assets/
│   ├── traffic.mp4         # Vídeo de teste (copiado do yolo_project)
│   └── preview.jpg         # Frame de referência
├── output/                 # Vídeos anotados + relatórios (JSON/CSV)
└── src/
    ├── config.py           # Dataclasses, modos, classes COCO, loaders JSON
    ├── video_source.py     # Captura arquivo/webcam/RTSP + frame skipping + reconexão
    ├── detector.py         # YOLO (Ultralytics) + ByteTrack/BoT-SORT -> Detection
    ├── zones.py            # Geometria ROI: polígono (pointPolygonTest) e linha (cruzamento)
    ├── counter.py          # Contagem por IDs únicos (sets) por zona + global
    ├── speed.py            # Velocidade: Bird's Eye View (getPerspectiveTransform), m/s×3.6
    ├── visualization.py    # Boxes, IDs, ROI, contadores, rótulos de km/h, HUD
    └── pipeline.py         # Orquestra os 4 modos + gravação + relatório
```

### Como funciona o pipeline
1. `VideoSource` entrega frames (com *frame skipping* e timestamps).
2. `Detector` roda `model.track(..., persist=True, tracker="bytetrack.yaml")`,
   atribuindo **IDs únicos e persistentes** a cada objeto.
3. `ObjectCounter` mantém **sets de IDs por zona** — cada objeto é contado
   apenas na **primeira travessia/presença**, evitando duplicidade entre frames.
4. `SpeedEstimator` projeta o ponto de contato do veículo com o chão numa vista
   métrica ("Bird's Eye View") e calcula `deslocamento / tempo × 3.6` (km/h),
   com suavização EMA e teto de sanidade.
5. `Visualizer` desenha tudo na tela; `pipeline` grava vídeo/relatórios opcionais.

---

## 🚀 Instalação

O projeto já inclui um `venv` pronto. Para recriar do zero:

```bat
cd Desktop\traffic_analysis
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Dependências: `ultralytics`, `opencv-python`, `numpy`, `lapx` (fornece o módulo
> `lap`, necessário pelo ByteTrack). Testado em **Python 3.15**, torch CPU.
> Para GPU, instale o torch CUDA e use `--device 0`.

> **Versionamento:** `venv/`, `output/` (vídeos/relatórios gerados) e vídeos de
> entrada (`assets/*.mp4`) ficam fora do git (`.gitignore`). O modelo
> `models/yolo_traffic.pt` está incluído. Coloque seu vídeo em `assets/traffic.mp4`
> ou use `--source` (veja `assets/README.md`).

---

## 🖥️ Uso

### Menu interativo
```bat
run.bat
```
ou
```bat
venv\Scripts\python.exe main.py
```

### CLI — exemplos por modo
```bat
REM 1. Contagem de pessoas
venv\Scripts\python.exe main.py --mode people --source assets/traffic.mp4

REM 2. Contagem de veículos (gravando vídeo anotado)
venv\Scripts\python.exe main.py --mode vehicles --source assets/traffic.mp4 --save-video

REM 3. Estimação de velocidade
venv\Scripts\python.exe main.py --mode speed --source assets/traffic.mp4

REM 4. Execução completa
venv\Scripts\python.exe main.py --mode full --source assets/traffic.mp4 --save-video

REM Webcam ao vivo com salto de frames (desempenho)
venv\Scripts\python.exe main.py --mode full --source 0 --skip-frames 2

REM Câmera IP / RTSP
venv\Scripts\python.exe main.py --mode vehicles --source "rtsp://user:senha@192.168.1.10:554/stream" --skip-frames 3

REM Teste rápido headless (sem janela), 100 frames
venv\Scripts\python.exe main.py --mode full --no-show --max-frames 100
```

### Principais opções
| Opção | Padrão | Descrição |
|-------|--------|-----------|
| `--source` | `assets/traffic.mp4` | Vídeo, URL RTSP/HTTP ou índice de webcam |
| `--mode` | menu | `people` / `vehicles` / `speed` / `full` |
| `--model` | `models/yolo_traffic.pt` | Modelo YOLO (aceita `yolov8n.pt`, `yolo11n.pt`, etc.) |
| `--tracker` | `bytetrack` | `bytetrack` ou `botsort` |
| `--skip-frames` | `1` | Processa 1 a cada N frames |
| `--conf` / `--iou` | `0.25` / `0.5` | Limiares de detecção / NMS |
| `--imgsz` | `640` | Tamanho de inferência |
| `--device` | `cpu` | `cpu`, `0` (GPU) ou `auto` |
| `--zones` | `config/zones.json` | Arquivo de zonas ROI |
| `--speed-config` | `config/speed.json` | Calibração de velocidade |
| `--save-video` | off | Grava vídeo anotado em `output/` |
| `--no-show` | off | Modo headless (sem janela) |
| `--max-frames` | `0` | Limita frames processados (0 = todos) |

Na janela de vídeo: **`q`** ou **ESC** encerra.

---

## 🐳 Docker

O projeto inclui `Dockerfile` e `docker-compose.yml` para execução isolada.

### Build da imagem
```bat
docker build -t traffic-analysis .
```

### Execução com docker-compose
Coloque seus vídeos em `./videos` e execute:

```bat
docker-compose up --build vehicles --source /videos/traffic.mp4 --save-video
docker-compose run --rm traffic_analysis people --source /videos/camera1.mp4
docker-compose run --rm traffic_analysis speed --source /videos/traffic.mp4 --speed-config /config/speed.json
```

Os relatórios e vídeos anotados ficam em `./output`.

### GPU (NVIDIA)
Descomente a seção `deploy` no `docker-compose.yml` e instale o
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
para usar GPU dentro do container (`TRAFFIC_DEVICE=0`).

---

## 🗺️ Zonas de Interesse (ROI)

`config/zones.json` aceita **polígonos** (presença na área) e **linhas**
(travessia, com sentido IN/OUT):

```json
{
  "zones": [
    { "name": "Area Principal", "type": "polygon",
      "points": [[0,380],[1280,380],[1280,719],[0,719]], "color": [0,200,255] },
    { "name": "Linha de Contagem", "type": "line",
      "points": [[0,560],[1280,560]], "color": [0,255,128] }
  ]
}
```

- **Polígono**: conta o objeto na primeira vez que seu ponto de contato entra na área.
- **Linha**: conta quando o segmento de movimento cruza a linha; o sentido
  (IN/OUT) é determinado pelo produto vetorial.

### Calibração interativa (clique na imagem)
```bat
venv\Scripts\python.exe calibrate.py roi --source assets/traffic.mp4 --at 3
```
Teclas: clique = adiciona ponto · `u` = desfaz · `r` = limpa · `p` = fecha
polígono · `l` = fecha linha · `s` = salvar · `q` = sair.

---

## ⚡ Estimação de Velocidade

`config/speed.json` define um trecho da pista por **4 pontos** (ordem:
topo-esquerda, topo-direita, base-direita, base-esquerda) e suas dimensões reais:

```json
{
  "enabled": true,
  "source_points": [[520,430],[760,430],[900,660],[300,660]],
  "real_width_m": 8.0,
  "real_height_m": 25.0,
  "pixels_per_meter": 15.0,
  "smoothing_alpha": 0.35,
  "max_speed_kmh": 200.0,
  "min_displacement_m": 0.05,
  "window_s": 0.5
}
```

`cv2.getPerspectiveTransform` gera a vista aérea métrica; o deslocamento do
ponto de contato do veículo entre timestamps é convertido **m/s → km/h (×3.6)**.

> ⚠️ **Os valores padrão são ilustrativos.** Para velocidades precisas, meça as
> dimensões reais do trecho e calibre os pontos:
> ```bat
> venv\Scripts\python.exe calibrate.py speed --source assets/traffic.mp4 --at 3
> ```
> Clique nos 4 pontos, informe largura/comprimento reais (m) e pixels por metro.

---

## 📤 Saídas

Em `output/` (com `--save-video` e/ou ao final de cada execução):
- **Vídeo anotado** `.mp4` (boxes, IDs, ROI, contadores, km/h). O FPS do vídeo
  de saída é `FPS_original / skip_frames`, preservando a duração real.
- **Relatório JSON** (`*_report.json`): totais, contagem por classe/zona,
  estatísticas de velocidade, calibração e metadados.
- **CSV de zonas** (`*_zones.csv`): contagem / IN / OUT por zona.

---

## 📝 Notas técnicas

- **YOLOv8 / YOLO11**: a API Ultralytics é a mesma; o projeto aceita qualquer modelo, ex. `--model yolov8n.pt`.
  - `models/yolo11n.pt` — COCO original (detecta person, car, truck, bus, motorcycle, etc.). Útil para contagem de pessoas + veículos.
  - `models/yolo_traffic.pt` — classe genérica `Vehicle` (detecta veículos sem distinguir tipo).
  - `models/yolo_traffic_classes.pt` — **recomendado para tráfego**: YOLOv8 Nano treinado com 8 classes separadas (auto, bus, car, lcv, motorcycle, multiaxle, tractor, truck). Não detecta pessoas.
- **Classes COCO**: `person=0`, `car=2`, `motorcycle=3`, `bus=5`, `truck=7`.
  O alias `motorbike` do enunciado é mapeado para `motorcycle`.
- **Desempenho**: em CPU, `--skip-frames 2..4` e `--imgsz 480` ajudam a manter
  fluidez em CCTV. Streams ao vivo usam buffer mínimo e reconexão automática.
- **Rastreamento**: ByteTrack (padrão) mantém IDs persistentes; BoT-SORT
  (`--tracker botsort`) adiciona compensação de movimento de câmera.
