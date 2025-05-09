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

from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import get_bisenet
from tqdm import tqdm
import random
import numpy as np
import os 
import torchvision.models as models

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18_weights = resnet18.state_dict()

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

# ================================
# Ambiente (Colab, Kaggle, Locale)
# ================================

is_colab = 'COLAB_GPU' in os.environ

if is_colab:
    print("📍 Ambiente: Colab")
    base_path = '/content/drive/MyDrive'
    data_dir = '/content/Cityscapes/Cityspaces'
    pretrain_model_path = '/content/MLDL_SS/deeplabv2_weights.pth'
else:
    print("📍 Ambiente: Locale")
    base_path = './'
    data_dir = './Cityscapes/Cityspaces'
    pretrain_model_path = './deeplabv2_weights.pth'

save_dir = os.path.join(base_path, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)

# =====================
# Transforms
# =====================
class LabelTransform():
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        # Resize
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        # Convert to tensor and long
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).long()
        return mask_tensor


def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((512, 1024)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.CenterCrop((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

# =====================
# Dataset & Dataloader
# =====================
transforms_dict = get_transforms()

train_dataset = CityScapes(
    root_dir=data_dir,
    split='train',
    transform=transforms_dict['train'],
    target_transform=LabelTransform()
)

val_dataset = CityScapes(
    root_dir=data_dir,
    split='val',
    transform=transforms_dict['val'],
    target_transform=LabelTransform()
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

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=0.0001)

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
        poly_lr_scheduler(optimizer, init_lr, iter_count, max_iter)

        optimizer.zero_grad()
        outputs = model(inputs)

        # DeepLabV2 returns a tuple (output, aux), use outputs[0] if that's the case
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
   
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

def validate(model, val_loader, criterion, num_classes=19):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    total_ious = []

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

            if batch_idx == 0:
                pred_vis = predicted[0].cpu().numpy()
                gt_vis = targets[0].cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                axes[0].imshow(gt_vis, cmap='tab20')
                axes[0].set_title("Ground Truth")
                axes[0].axis('off')
                
                axes[1].imshow(pred_vis, cmap='tab20')
                axes[1].set_title("Predizione")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/gt_vs_pred_epoch_{epoch}.png")
                plt.close()


    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    iou_per_class = torch.tensor(total_ious).nanmean(dim=0)
    miou = iou_per_class.nanmean().item()

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}% | mIoU: {miou:.4f}')
    return val_accuracy, miou

# =====================
# Main Loop
# =====================
def main():
    best_model_path = os.path.join(save_dir, 'best_model_bisenet.pth')
    checkpoint_path = os.path.join(save_dir, 'checkpoint_bisenet.pth')

    save_every = 1
    best_miou = 0
    start_epoch = 1
    init_lr = 0.025

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Ripreso da epoca {checkpoint['epoch']} con mIoU: {best_miou:.4f}")

    for epoch in range(start_epoch, num_epochs + 1):
        train(epoch, model, train_dataloader, criterion, optimizer, init_lr)
        val_accuracy, miou = validate(model, val_dataloader, criterion, epoch=epoch)

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model salvato con mIoU: {miou:.4f}")

        if epoch % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_miou': best_miou
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint salvato all’epoca {epoch}")

        if epoch % 10 == 0:
            model.eval()
            df = benchmark_model(model, image_size=(3, 512, 1024), iterations=100, device=device)
            csv_path = os.path.join(save_dir, f'benchmark_epoch_{epoch}.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 Benchmark salvato: {csv_path}")

    model.load_state_dict(torch.load(best_model_path))
    validate(model, val_dataloader, criterion)

    # Plot di benchmark (se presente)
    if 'df' in locals():
        plt.figure(figsize=(10, 4))
        plt.plot(df['iteration'], df['latency_s'], label='Latency (s)')
        plt.title('Latency per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Latency (s)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(df['iteration'], df['fps'], label='FPS', color='green')
        plt.title('FPS per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('FPS')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
