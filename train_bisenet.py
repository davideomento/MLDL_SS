import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

from monai.losses import DiceLoss
from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import get_bisenet
from metrics import benchmark_model, calculate_iou

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================
# Set Seed
# =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =====================
# Paths
# =====================
print("📍 Ambiente: Colab (Drive)")
base_path = '/content/drive/MyDrive/Project_MLDL'
data_dir = '/content/MLDL_SS/Cityscapes/Cityspaces'
save_dir = os.path.join(base_path, 'checkpoints_tati2')
os.makedirs(save_dir, exist_ok=True)


# =====================
# Label Transform
# =====================
class LabelTransform():
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return torch.as_tensor(mask, dtype=torch.long)          

###############

def get_transforms():
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(512, 1024), scale=(0.5, 1.0), ratio=(1.75, 2.25)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(512, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return {
        'train': train_transform,
        'val': val_transform
    }



# =====================
# Dataset & Dataloader
# =====================
transforms_dict = get_transforms()
label_transform = LabelTransform()

train_dataset = CityScapes(
    root_dir=data_dir,
    split='train',
    transform=transforms_dict['train'],
    target_transform=label_transform
)

val_dataset = CityScapes(
    root_dir=data_dir,
    split='val',
    transform=transforms_dict['val'],
    target_transform=label_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# =====================
# Model Setup
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18_weights = resnet18.state_dict()

model = get_bisenet(
    num_classes=19,
    pretrain=True,
    pretrained_weights=resnet18_weights
).to(device)


class_weights = torch.tensor([
    2.6, 6.9, 3.5, 3.6, 3.6, 3.8, 3.4, 3.5, 5.1, 4.7,
    6.2, 5.2, 4.9, 3.6, 4.3, 5.6, 6.5, 7.0, 6.6
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# =====================
# Poly LR Scheduler (Per Iter)
# =====================
num_epochs = 50
max_iter = len(train_dataloader) * num_epochs

def poly_lr_scheduler(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - curr_iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# =====================
# Training & Validation
# =====================
def train(epoch, model, train_loader, criterion, optimizer, init_lr):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    global_iter = (epoch - 1) * len(train_loader)

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        iter_count = global_iter + batch_idx
        scheduler.step()
        #poly_lr_scheduler(optimizer, init_lr, iter_count, max_iter)

        optimizer.zero_grad()
        outputs = model(inputs)

        if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
            main_out, aux1_out, aux2_out = outputs
            loss = (
                criterion(main_out, targets)
                + 0.4 * criterion(aux1_out, targets)
                + 0.4 * criterion(aux2_out, targets)
            )
        else:
            total_loss = criterion(outputs, targets) + dice_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

 
CITYSCAPES_COLORS = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
    (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
    (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
    (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
    (  0, 80,100), (  0,  0,230), (119, 11, 32)
]
     
def decode_segmap(mask):
    """Converte una mappa con classi 0-18 in immagine RGB"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == label_id] = color
    return color_mask



def validate(model, val_loader, criterion, num_classes=19, epoch=0):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    total_ious = []
    loss_values = []
    accuracy_values = []

    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                main_out, aux1_out, aux2_out = outputs
                loss = (
                    criterion(main_out, targets)
                    + 0.4 * criterion(aux1_out, targets)
                    + 0.4 * criterion(aux2_out, targets)
                )
                outputs = main_out  # solo per predizione
            else:
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            ious = calculate_iou(predicted, targets, num_classes, ignore_index=255)
            total_ious.append(ious)

            # Salvataggio della loss e accuratezza per epoca
            loss_values.append(loss.item())
            accuracy_values.append((predicted == targets).sum().item() / targets.numel())

    # Calcolo delle metriche per epoca
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    iou_per_class = torch.tensor(total_ious).nanmean(dim=0)
    miou = iou_per_class.nanmean().item()

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}% | mIoU: {miou:.4f}')
    
    return {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'miou': miou,
        'iou_per_class': iou_per_class,
        'loss_values': loss_values,
        'accuracy_values': accuracy_values
    }



# Modificare la funzione main per raccogliere e salvare i dati
def main():
    best_model_path = os.path.join(save_dir, 'best_model_bisenet.pth')
    checkpoint_path = os.path.join(save_dir, 'checkpoint_bisenet.pth')

    save_every = 1
    best_miou = 0
    start_epoch = 1
    init_lr = 0.025

    # Dati per il salvataggio delle metriche
    metrics_data = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'miou': [],
        'iou_per_class': []
    }

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Ripreso da epoca {checkpoint['epoch']} con mIoU: {best_miou:.4f}")

    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss = train(epoch, model, train_dataloader, criterion, optimizer, init_lr)
        
        # Validation and Metrics
        val_metrics = validate(model, val_dataloader, criterion, epoch=epoch)
        
        # Registriamo i dati per il salvataggio
        metrics_data['epoch'].append(epoch)
        metrics_data['train_loss'].append(train_loss)
        metrics_data['val_loss'].append(val_metrics['loss'])
        metrics_data['val_accuracy'].append(val_metrics['accuracy'])
        metrics_data['miou'].append(val_metrics['miou'])
        metrics_data['iou_per_class'].append(val_metrics['iou_per_class'].cpu().numpy())

        # Salvataggio del modello migliore
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model salvato con mIoU: {val_metrics['miou']:.4f}")

        if epoch % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_miou': best_miou
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint salvato all’epoca {epoch}")
            
            # Salvataggio delle metriche su CSV
            df = pd.DataFrame(metrics_data)
            csv_path = os.path.join(save_dir, f'metrics_epoch_{epoch}.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 Metriche salvate in {csv_path}")

    # Al termine dell'addestramento, carica il miglior modello e valida di nuovo
    model.load_state_dict(torch.load(best_model_path))
    validate(model, val_dataloader, criterion)

    # Esegui il grafico delle metriche salvate
    plot_metrics(metrics_data)

def plot_metrics(metrics_data):
    # Funzione per plottare le metriche nel tempo
    df = pd.DataFrame(metrics_data)

    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['miou'], label='mIoU')
    plt.title('Mean IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot della IoU per classe (opzionale)
    iou_per_class = np.array(df['iou_per_class'].tolist())
    for i in range(iou_per_class.shape[1]):
        plt.plot(df['epoch'], iou_per_class[:, i], label=f'Class {i}')
    plt.title('IoU per Class over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
