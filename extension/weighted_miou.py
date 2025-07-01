import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import os

from cityscapes_aug import CityScapes_aug  # il tuo dataset
from stdc_model import STDC_Seg           # il tuo modello
from metrics import calculate_iou         # la funzione per IoU
from metrics import ClassImportanceWeights  # la classe per pesi di importanza

# Parametri
checkpoint_folder = '/content/drive/MyDrive/Project_MLDL/checkpoints_STDC2_paperlike'  # cartella checkpoint
checkpoint_path = f'{checkpoint_folder}/checkpoints.pth'
excel_path = '/content/drive/MyDrive/Project_MLDL/weighted_miou_results.xlsx'  # percorso file Excel su Drive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 19

# Commento da input
comment = input("Inserisci un commento per questa run: ")

# Carica dataset validation
val_dataset = CityScapes_aug(
    root_dir='/content/MLDL_SS/Cityscapes/Cityspaces',
    split='val',
    transform=None,
    target_transform=None
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Carica modello
model = STDC_Seg(num_classes=num_classes, backbone='STDC2', use_detail=True)
model.to(device)

# Carica checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model_state'])
model.eval()

total_intersection = torch.zeros(num_classes, dtype=torch.float64, device=device)
total_union = torch.zeros(num_classes, dtype=torch.float64, device=device)

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        preds = torch.argmax(outputs, dim=1)

        inter, uni = calculate_iou(preds, targets, num_classes, ignore_index=255)
        total_intersection += inter.to(device)
        total_union += uni.to(device)

# Calcola IoU per classe
ious = total_intersection / total_union

# Prendi i pesi e calcola weighted mIoU
class_weights_obj = ClassImportanceWeights()
class_weights = class_weights_obj.get_weights().to(device)

weighted_miou = (ious * class_weights).sum() / class_weights.sum()

miou_mean = ious.mean().item()
weighted_miou_val = weighted_miou.item()

print(f"mIoU standard: {miou_mean:.4f}")
print(f"Weighted mIoU: {weighted_miou_val:.4f}")

# Salvataggio su Excel (aggiorna o crea file)
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    df = pd.DataFrame(columns=['CheckpointFolder', 'Comment', 'mIoU', 'Weighted_mIoU'])

new_row = {
    'CheckpointFolder': checkpoint_folder,
    'Comment': comment,
    'mIoU': miou_mean,
    'Weighted_mIoU': weighted_miou_val
}

df = df.append(new_row, ignore_index=True)
df.to_excel(excel_path, index=False)
print(f"Risultati salvati in {excel_path}")
