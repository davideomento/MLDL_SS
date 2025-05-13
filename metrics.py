import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from models.deeplabv2.deeplabv2 import get_deeplab_v2
import os
from models.bisenet.build_bisenet import BiSeNet

# ================================
# Ambiente (Colab)
# ================================

print("📍 Ambiente: Colab")
pretrain_model_path = '/content/MLDL_SS/deeplabv2_weights.pth'


model_deeplab = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path=pretrain_model_path)
model_bisenet = BiSeNet(num_classes=19, context_path='resnet18')

# =====================
# Utils - mIoU
# =====================
'''
def calculate_iou(predicted, target, num_classes, ignore_index=255):
    # Maschera per escludere i pixel da ignorare
    mask = target != ignore_index
    
    ious = []
    for i in range(num_classes):
        # Consideriamo solo i pixel che non sono da ignorare
        intersection = ((predicted == i) & (target == i) & mask).sum().item()
        union = ((predicted == i) | (target == i) & mask).sum().item()
        
        if union == 0:
            iou = float('nan')  # Se non ci sono pixel di quella classe, mettiamo NaN
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

'''
def calculate_iou(predicted, target, num_classes, ignore_index=255):
    mask = target != ignore_index
    predicted = predicted[mask]
    target = target[mask]

    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)

    for i in range(num_classes):
        inter = ((predicted == i) & (target == i)).sum().item()
        uni = ((predicted == i) | (target == i)).sum().item()
        intersection[i] += inter
        union[i] += uni

    return intersection, union


def benchmark_model(model: torch.nn.Module,
                    image_size: tuple = (3, 512, 1024),
                    iterations: int = 200,
                    device: str = 'cuda') -> pd.DataFrame:
    """
    Esegue il benchmark del modello: ritorna un DataFrame con latenza e FPS per immagine.
    """
    model = model.to(device).eval()
    dummy_input = torch.randn((1, *image_size), device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    records = []
    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            latency = end - start
            fps = 1.0 / latency
            records.append({'iteration': i + 1, 'latency_s': latency, 'fps': fps})

    return pd.DataFrame.from_records(records)
