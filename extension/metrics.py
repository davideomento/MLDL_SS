import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import wandb
import fvcore.nn.flop_count as fc
from fvcore.nn import FlopCountAnalysis 

# ================================
# Ambiente (Colab)
# ================================

is_colab = 'COLAB_GPU' in os.environ
is_kaggle = os.path.exists('/kaggle')

if is_colab:
    pretrain_model_path = '/content/MLDL_SS/deeplabv2_weights.pth'
elif is_kaggle:
    pretrain_model_path = '/kaggle/input/deeplab-resnet-pretrained-imagenet/deeplab_resnet_pretrained_imagenet (1).pth'  # <-- verifica che il file sia lì
else:
    print("📍 Ambiente: Locale")
    base_drive_path = './'
    working_dir = './'


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
                    device: str = 'cuda') -> dict:
    """
    Esegue il benchmark del modello: ritorna un dizionario con latenza, FPS, FLOPs e num_parametri.
    """
    model = model.to(device).eval()
    dummy_input = torch.randn((1, *image_size), device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
        if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
            torch.cuda.synchronize()

    # Benchmark latenza e FPS
    records = []
    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
                torch.cuda.synchronize()
            end = time.time()

            latency = end - start
            fps = 1.0 / latency
            records.append({'latency_s': latency, 'fps': fps})

    df_bench = pd.DataFrame.from_records(records)
    mean_latency = df_bench['latency_s'].mean()
    mean_fps = df_bench['fps'].mean()

    # FLOPs e parametri con fvcore
    # Passa l'input senza batch dimension per FlopCountAnalysis
    flop_analyzer = FlopCountAnalysis(model, dummy_input)
    total_flops = flop_analyzer.total()  # numero di FLOPs
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'mean_latency': mean_latency,
        'mean_fps': mean_fps,
        'num_flops': total_flops,
        'trainable_params': num_params
    }

#Function to save the metrics on WandB
def save_metrics_on_wandb(epoch, metrics_train, metrics_val):

    to_serialize = {
        "epoch": epoch,
        "train_loss": metrics_train,
        "val_mIoU": metrics_val['miou'],
        "val_loss": metrics_val['loss'],
        "val_accuracy":metrics_val['accuracy']
    }
    #"val_IoU_per_class": metrics_val['iou_per_class']
    for index, iou in enumerate(metrics_val['iou_per_class']):
        to_serialize[f"class_{index}_val"] = iou

    # Log delle metriche di training e validazione su WandB
    if epoch != 50:
        wandb.log(to_serialize)

    # Salvataggio delle metriche finali al 50esimo epoch
    if epoch == 50:
        wandb.log({
            "train_loss_final": metrics_train,
            "val_IoU_final": metrics_val['miou'],
            "val_loss_final": metrics_val['loss'],
            "val_latency": metrics_val['mean_latency'],
            "val_fps": metrics_val['mean_fps'],
            "val_flops": metrics_val['num_flops'],
            "val_params": metrics_val['trainable_params']
        })