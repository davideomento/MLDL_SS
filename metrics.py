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
import wandb

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

#Function to save the metrics on WandB           
def save_metrics_on_wandb(epoch, metrics_train, metrics_val):

    to_serialize = {
        "epoch": epoch,
        "train_mIoU": metrics_train['mean_iou'],
        "train_loss": metrics_train['mean_loss'],
        "val_mIoU": metrics_val['mean_iou'],
        "val_mIoU_per_class": metrics_val['iou_per_class'],
        "val_loss": metrics_val['mean_loss']
    }

    print(metrics_train['iou_per_class'])

    for index, iou in enumerate(metrics_train['iou_per_class']):
        to_serialize[f"class_{index}_train"] = iou

    for index, iou in enumerate(metrics_val['iou_per_class']):
        to_serialize[f"class_{index}_val"] = iou

    # Log delle metriche di training e validazione su WandB
    if epoch != 50:
        wandb.log(to_serialize)

    # Salvataggio delle metriche finali al 50esimo epoch
    if epoch == 50:
        wandb.log({
            "train_mIoU_final": metrics_train['mean_iou'],
            "train_loss_final": metrics_train['mean_loss'],
            "train_latency": metrics_train['mean_latency'],
            "train_fps": metrics_train['mean_fps'],
            "train_flops": metrics_train['num_flops'],
            "train_params": metrics_train['trainable_params'],
            "val_mIoU_final": metrics_val['mean_iou'],
            "val_loss_final": metrics_val['mean_loss'],
            "val_latency": metrics_val['mean_latency'],
            "val_fps": metrics_val['mean_fps'],
            "val_flops": metrics_val['num_flops'],
            "val_params": metrics_val['trainable_params']
        })