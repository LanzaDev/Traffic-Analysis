#!/usr/bin/env python
"""
Gerador de dataset sintético para treino de YOLO - Detecção de Tráfego
Cria imagens sintéticas com retângulos coloridos simulando veículos
e anotações no formato YOLO para treino/teste rápido.
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path

# Diretórios
base_dir = Path("C:/Users/lucas.lanza/Desktop/datasets/traffic_sample")
train_img_dir = base_dir / "images" / "train"
val_img_dir = base_dir / "images" / "val"
train_label_dir = base_dir / "labels" / "train"
val_label_dir = base_dir / "labels" / "val"

# Criar diretórios
for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Classes COCO correspondentes
CLASS_NAMES = {
    0: "person",
    1: "car",
    2: "motorcycle",
    3: "bus",
    4: "truck",
    5: "articulated_truck"
}

# Cores BGR para cada classe
CLASS_COLORS = {
    0: (100, 100, 100),   # pessoa - cinza
    1: (0, 0, 255),       # car - vermelho
    2: (0, 255, 255),     # motorcycle - amarelo
    3: (255, 0, 0),       # bus - azul
    4: (0, 128, 255),     # truck - laranja
    5: (128, 0, 255),     # articulated_truck - roxo
}

def generate_random_image(width=640, height=480):
    """Gera uma imagem sintética com estrada e veículos."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fundo - céu azul claro (topo)
    cv2.rectangle(img, (0, 0), (width, height//3), (200, 220, 240), -1)
    
    # Céu - gradiente para o horizonte
    for y in range(height//3):
        val = int(200 + (y / (height//3)) * 55)
        cv2.line(img, (0, y), (width, y), (val, val+15, val+40), 1)
    
    # Estrada cinza
    road_top = height // 2
    cv2.rectangle(img, (0, road_top), (width, height), (60, 60, 60), -1)
    
    # Linhas da pista
    cv2.line(img, (0, road_top + 50), (width, road_top + 50), (100, 100, 100), 2)
    cv2.line(img, (0, road_top + 100), (width, road_top + 100), (100, 100, 100), 2)
    
    # Linhas divisórias pontilhadas
    for x in range(0, width, 80):
        cv2.rectangle(img, (x, road_top + 125), (x+40, road_top + 135), (255, 255, 255), -1)
    
    return img

def draw_vehicle(img, class_id, x_center, y_center, width_px, height_px):
    """Desenha um veículo na imagem."""
    x1 = max(0, int(x_center - width_px // 2))
    y1 = max(0, int(y_center - height_px // 2))
    x2 = min(img.shape[1] - 1, int(x_center + width_px // 2))
    y2 = min(img.shape[0] - 1, int(y_center + height_px // 2))
    
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    
    # Desenhar retângulo do veículo
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    # Detalhes do veículo (janela)
    cv2.rectangle(img, (x1+5, y1+5), (x2-5, y1+height_px//3), (50, 50, 50), -1)
    
    # Rodas
    cv2.circle(img, (x1+15, y2-10), 8, (20, 20, 20), -1)
    cv2.circle(img, (x2-15, y2-10), 8, (20, 20, 20), -1)
    
    return (x1, y1, x2, y2)

def generate_dataset(num_train=20, num_val=5):
    """Gera dataset sintético completo."""
    
    all_detections = []
    
    for split_name, img_dir, label_dir, num_imgs in [
        ("train", train_img_dir, train_label_dir, num_train),
        ("val", val_img_dir, val_label_dir, num_val)
    ]:
        print(f"\nGerando {num_imgs} imagens para {split_name}...")
        
        for i in range(num_imgs):
            img = generate_random_image()
            img_name = f"{split_name}_{i:03d}.jpg"
            img_path = img_dir / img_name
            
            # Gerar 1-4 veículos por imagem
            num_objects = random.randint(1, 4)
            frame_detections = []
            
            for _ in range(num_objects):
                class_id = random.choice([1, 2, 3, 4, 5])  # Não incluir person (0) para simplificar
                
                # Posição aleatória na estrada (metade inferior da imagem)
                x_center = random.randint(50, img.shape[1] - 50)
                y_center = random.randint(img.shape[0] // 2 + 50, img.shape[0] - 60)
                
                # Tamanho baseado na classe
                if class_id == 1:  # car
                    w, h = random.randint(40, 80), random.randint(25, 45)
                elif class_id == 2:  # motorcycle
                    w, h = random.randint(20, 40), random.randint(15, 30)
                elif class_id == 3:  # bus
                    w, h = random.randint(90, 130), random.randint(40, 60)
                elif class_id == 4:  # truck
                    w, h = random.randint(70, 100), random.randint(35, 55)
                else:  # articulated_truck
                    w, h = random.randint(100, 140), random.randint(40, 60)
                
                # Desenhar veículo
                bbox = draw_vehicle(img, class_id, x_center, y_center, w, h)
                
                # Converter para formato YOLO (normalizado 0-1)
                x_center_norm = x_center / img.shape[1]
                y_center_norm = y_center / img.shape[0]
                w_norm = w / img.shape[1]
                h_norm = h / img.shape[0]
                
                frame_detections.append((class_id, x_center_norm, y_center_norm, w_norm, h_norm))
            
            # Salvar imagem
            cv2.imwrite(str(img_path), img)
            
            # Salvar anotações
            label_path = label_dir / f"{split_name}_{i:03d}.txt"
            with open(label_path, 'w') as f:
                for cls_id, x_c, y_c, w, h in frame_detections:
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            print(f"  Gerado: {img_name} ({len(frame_detections)} objetos)")
    
    print("\n✅ Dataset gerado com sucesso!")

if __name__ == "__main__":
    print("🚗 Gerando dataset sintético para treino de YOLO...")
    generate_dataset(num_train=30, num_val=10)
    print("\n📁 Estrutura criada:")
    print("   traffic_sample/")
    print("   ├── images/train/ (30 imagens)")
    print("   ├── images/val/ (10 imagens)")
    print("   ├── labels/train/ (30 arquivos .txt)")
    print("   └── labels/val/ (10 arquivos .txt)")