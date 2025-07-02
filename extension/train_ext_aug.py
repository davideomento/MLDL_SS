import os
import random
from re import A
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F  
import wandb
import albumentations as A
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from stdc_model import STDC_Seg
from albumentations.pytorch import ToTensorV2

from monai.losses import DiceLoss
from cityscapes_aug import CityScapes_aug
from metrics import benchmark_model, calculate_iou, save_metrics_on_wandb, ClassImportanceWeights
from utils import poly_lr_scheduler

class CombinedLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)

    def forward(self, input, target):
        ce_loss = self.ce(input, target)

        # Aggiungi la dimensione canale al target per DiceLoss
        target_dice = target.unsqueeze(1)  # shape [B, 1, H, W]

        # Crea una maschera per ignorare i pixel con valore 255 nel target originale
        mask = (target != 255).unsqueeze(1)  # shape [B, 1, H, W]

        # Applica la maschera su input e target_dice
        input_masked = input * mask  # maschera i pixel da ignorare
        target_masked = target_dice * mask

        dice_loss = self.dice(input_masked, target_masked)

        return ce_loss + dice_loss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print("📍 Ambiente: Colab (Drive)")
base_path = '/content/drive/MyDrive/Project_MLDL'
data_dir = '/content/MLDL_SS/Cityscapes/Cityspaces'
save_dir = os.path.join(base_path, 'checkpoints_STDC2_prova_tati')
os.makedirs(save_dir, exist_ok=True)

# ✅ CORRETTO: restituisce target con shape (H, W), no canale in più
class LabelTransform():
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        return torch.as_tensor(mask, dtype=torch.long)

def get_transforms():
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(512, 1024), scale=(0.5, 1.0), ratio=(1.5, 2.5), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=512, width=1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return {'train': train_transform, 'val': val_transform}

transforms_dict = get_transforms()
label_transform = LabelTransform()

train_dataset = CityScapes_aug(root_dir=data_dir, split='train', transform=transforms_dict['train'], target_transform=label_transform)
val_dataset = CityScapes_aug(root_dir=data_dir, split='val', transform=transforms_dict['val'], target_transform=label_transform)


# Calcolo la lunghezza del 10%
train_len_10 = int(0.1 * len(train_dataset))
val_len_10 = int(0.1 * len(val_dataset))

# Prendo indici casuali senza ripetizioni
train_indices_10 = np.random.choice(len(train_dataset), train_len_10, replace=False)
val_indices_10 = np.random.choice(len(val_dataset), val_len_10, replace=False)

# Creo i subset
train_subset_10 = Subset(train_dataset, train_indices_10)
val_subset_10 = Subset(val_dataset, val_indices_10)

# Creo i dataloader con i subset
train_dataloader = DataLoader(train_subset_10, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
val_dataloader = DataLoader(val_subset_10, batch_size=4, shuffle=False, num_workers=2, drop_last=True)


'''
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STDC_Seg(num_classes=19, backbone='STDC2', use_detail=True).to(device)

class_weights = torch.tensor([
    2.6, 6.9, 3.5, 3.6, 3.6, 3.8, 3.4, 3.5, 5.1, 4.7,
    6.2, 5.2, 4.9, 3.6, 4.3, 5.6, 6.5, 7.0, 6.6
], dtype=torch.float).to(device)

criterion = CombinedLoss(weight=class_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

num_epochs = 50
max_iter = num_epochs

def train(epoch, model, train_loader, criterion, optimizer, init_lr, λ=1.0):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    current_iter = epoch
    poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter)
    seg_loss_fn = criterion
    detail_criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # ✅ CORRETTO: rimuove dimensione extra

        optimizer.zero_grad()
        output = model(inputs)

        if isinstance(output, tuple):
            seg_out, detail_map = output
            seg_out = F.interpolate(seg_out, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            loss_seg = seg_loss_fn(seg_out, targets)

            detail_target = (targets > 0).float().unsqueeze(1)
            detail_map = F.interpolate(detail_map, size=detail_target.shape[-2:], mode='bilinear', align_corners=False)
            loss_detail = detail_criterion(detail_map, detail_target)

            loss = loss_seg + λ * loss_detail
        else:
            output = F.interpolate(output, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            loss = seg_loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    mean_loss = running_loss / len(train_loader)
    lr = optimizer.param_groups[0]['lr']

    wandb.log({
        "epoch": epoch,
        "loss": mean_loss,
        "lr": lr
    }, step=epoch)

    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss,
    }, model_save_path)

    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)

    return mean_loss

CITYSCAPES_COLORS = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
    (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
    (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
    (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
    (  0, 80,100), (  0,  0,230), (119, 11, 32)
]

def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == label_id] = color
    return color_mask

def validate(model, val_loader, criterion, epoch, num_classes=19):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    loss_values = []
    accuracy_values = []
    total_intersection = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            inter, uni = calculate_iou(predicted, targets, num_classes, ignore_index=255)
            total_intersection += inter
            total_union += uni

            loss_values.append(loss.item())
            accuracy_values.append((predicted == targets).sum().item() / targets.numel())

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    iou_per_class = total_intersection / total_union
    miou = torch.nanmean(iou_per_class).item()
    weight_tensor = torch.tensor([class_weights.get(i, 0.0) for i in range(len(iou_per_class))], device=iou_per_class.device)
    valid_mask = ~torch.isnan(iou_per_class)
    weighted_iou = torch.nansum(iou_per_class * weight_tensor) / torch.sum(weight_tensor[valid_mask])
    wmiou = weighted_iou.item()

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}% | mIoU: {miou:.4f}')

    bench_results = benchmark_model(model) if epoch == 50 else {k: None for k in ['mean_latency','mean_fps','num_flops','trainable_params']}

    return {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'miou': miou,
        'iou_per_class': iou_per_class,
        'loss_values': loss_values,
        'accuracy_values': accuracy_values,
        **bench_results
    }

def main():
    checkpoint_path = os.path.join(save_dir, 'checkpoints.pth')
    var_model = "STDC2"
    best_miou = 0
    start_epoch = 1
    init_lr = 2.5e-2
    project_name = f"{var_model}_provatati"

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
        id=f"{var_model}_run",
        resume="allow"
    )

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

    validate(model, val_dataloader, criterion, epoch)

if __name__ == "__main__":
    main()
