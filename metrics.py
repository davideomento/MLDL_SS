import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from models.deeplabv2.deeplabv2 import get_deeplab_v2


model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path='/content/MLDL_SS/deeplabv2_weights.pth')


# =====================
# Utils - mIoU
# =====================

def calculate_iou(preds, labels, num_classes=19):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        label_inds = labels == cls

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))  # classe non presente
        else:
            ious.append(intersection / union)

    return ious


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
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    # Benchmark
    records = []
    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end = time.time()

            latency = end - start
            fps = 1.0 / latency
            records.append({'iteration': i + 1, 'latency_s': latency, 'fps': fps})

    return pd.DataFrame.from_records(records)