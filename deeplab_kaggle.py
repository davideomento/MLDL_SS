import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset

from datasets.cityscapes import CityScapes
from utils import poly_lr_scheduler
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from metrics import benchmark_model, calculate_iou, save_metrics_on_wandb

# =====================
# Set Seed for Reproducibility
# =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ================================
# Ambiente: Kaggle
# ================================

print("\ud83d\udccd Ambiente: Kaggle")

# Percorso di lavoro per salvare output (modelli, metriche, immagini)
base_path = '/kaggle/working'

# Dataset path
data_dir = '/kaggle/input/cityscapes/Cityscapes/Cityspaces'

# Pretrained model path
pretrain_model_path = '/kaggle/input/deeplab-resnet-pretrained-imagenet/deeplab_resnet_pretrained_imagenet (1).pth'
save_dir = os.path.join(base_path, 'checkpoints_deeplabv2')
os.makedirs(save_dir, exist_ok=True)  # <-- CREA LA CARTELLA SE NON ESISTE
checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pt")

# =====================
# Label Transforms
# =====================
class LabelTransform():
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return torch.as_tensor(mask, dtype=torch.long)   

img_transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def mask_transform(mask):
    return F.resize(mask, (512, 1024), interpolation=F.InterpolationMode.NEAREST)

def get_transforms():
    return {'train': (img_transform, mask_transform), 'val': (img_transform, mask_transform)}

# =====================
# Dataset & Dataloader
# =====================
transforms_dict = get_transforms()
label_transform = LabelTransform()
train_dataset = CityScapes(data_dir, 'train', transforms_dict['train'], label_transform)
val_dataset = CityScapes(data_dir, 'val', transforms_dict['val'], label_transform)

train_subset = Subset(train_dataset, np.random.permutation(len(train_dataset))[:len(train_dataset)])
val_subset = Subset(val_dataset, np.random.permutation(len(val_dataset))[:len(val_dataset)])

train_dataloader = DataLoader(train_subset, batch_size=2, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_subset, batch_size=2, shuffle=False, num_workers=2)

# =====================
# Model, Loss, Optimizer
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path=pretrain_model_path).to(device)

class_weights = torch.tensor([
    2.6, 6.9, 3.5, 3.6, 3.6, 3.8, 3.4, 3.5, 5.1, 4.7,
    6.2, 5.2, 4.9, 3.6, 4.3, 5.6, 6.5, 7.0, 6.6
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
optimizer = optim.SGD(model.optim_parameters(lr=1e-3), momentum=0.9, weight_decay=0.0005)
num_epochs = 50
max_iter = num_epochs * len(train_dataloader)

# =====================
# Train / Validate Functions (rimangono invariati)
# =====================
# [OMESSO PER BREVITA' - Il contenuto delle funzioni `train`, `validate`, `decode_segmap` va mantenuto come nel tuo script]

# =====================
# Main Function
# =====================
def main():
    var_model = "Deeplabv2"
    init_lr = 1e-3
    best_miou = 0
    start_epoch = 1
    project_name = f"{var_model}_official"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Ripreso da epoca {checkpoint['epoch']} con mIoU: {best_miou:.4f}")

    wandb.init(
        project=project_name,
        entity="mldl-semseg-politecnico-di-torino",
        name=f"run_{var_model}",
        resume="allow"
    )
    print("🛰️ Wandb inizializzato")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train(epoch, model, train_dataloader, criterion, optimizer, init_lr)
        val_metrics = validate(model, val_dataloader, criterion, epoch=epoch)
        save_metrics_on_wandb(epoch, train_loss, val_metrics)

        checkpoint_data = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_miou': val_metrics['miou'],
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint salvato a {checkpoint_path}")

    validate(model, val_dataloader, criterion)

if __name__ == "__main__":
    main()