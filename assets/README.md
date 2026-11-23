# Ativos (assets)

Este diretório guarda mídias de entrada. Por padrão, **vídeos não são versionados**
(veja `.gitignore`) para manter o repositório leve — arquivos de vídeo grandes
deixam o clone lento e podem esbarrar nos limites do GitHub.

## Como usar
- Coloque um vídeo de teste aqui como `traffic.mp4` (padrão do `--source`), **ou**
- Informe qualquer caminho/URL na execução:
  ```bat
  run.bat --mode full --source "C:\caminho\do\seu\video.mp4"
  run.bat --mode vehicles --source "rtsp://usuario:senha@IP:554/stream"
  ```

`preview.jpg` é apenas um frame de referência (1280×720) usado para ilustrar a
calibração de ROI/velocidade.

> O modelo `models/yolo11n.pt` (~5 MB) **está versionado**. Se preferir um repo
> ainda mais enxuto, remova-o do git: o Ultralytics baixa `yolo11n.pt`
> automaticamente na primeira execução.
