import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from models.deeplabv2.deeplabv2 import get_deeplab_v2
import os
from models.bisenet.build_bisenet import get_bisenet

# ================================
# Ambiente (Colab, Kaggle, Locale)
# ================================
is_colab = 'COLAB_GPU' in os.environ
is_kaggle = os.path.exists('/kaggle')

if is_colab:
    print("📍 Ambiente: Colab")
    pretrain_model_path = '/content/MLDL_SS/deeplabv2_weights.pth'

elif is_kaggle:
    print("📍 Ambiente: Kaggle")
    pretrain_model_path = '/kaggle/input/deeplab_resnet_pretrained_imagenet.pth'
else:
    print("📍 Ambiente: Locale")
    pretrain_model_path = './deeplabv2_weights.pth'


model_deeplab = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path=pretrain_model_path)

# =====================
# Utils - mIoU
# =====================

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

#SISTEMA CALCOLO IoU

#Misura la velocità del modello in termini di latenza e FPS (Frame Per Second)  
def benchmark_model(model: torch.nn.Module,
                    image_size: tuple = (3, 512, 1024),
                    iterations: int = 200,
                    device: str = 'cuda') -> pd.DataFrame:
    """
    Esegue il benchmark del modello: ritorna un DataFrame con latenza e FPS per immagine.
    """
    model = model.to(device).eval()
    dummy_input = torch.randn((1, *image_size), device=device) #creo un input fittizio per il modello della dimensione specificata

    # Warm-up
    with torch.no_grad():
        for _ in range(50): #fa 50 test a vuoto per scaldare la GPU, serve per misurazioni più realistiche e stabili
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
            end = time.time() #calcola il tempo di esecuzione del modello

            latency = end - start
            fps = 1.0 / latency
            records.append({'iteration': i + 1, 'latency_s': latency, 'fps': fps})

    return pd.DataFrame.from_records(records)