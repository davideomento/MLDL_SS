import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm

from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from metrics import benchmark_model, calculate_iou

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
# Ambiente: Colab o Locale
# ================================
is_colab = 'COLAB_GPU' in os.environ

if is_colab:
    print("📍 Ambiente: Colab")
    base_path = '/content/drive/MyDrive/Project_MLDL'
    data_dir = '/content/MLDL_SS/Cityscapes/Cityspaces'
    pretrain_model_path = '/content/MLDL_SS/deeplabv2_weights.pth'
else:
    print("📍 Ambiente: Locale")
    base_path = './'
    data_dir = './Cityscapes/Cityspaces'
    pretrain_model_path = './deeplabv2_weights.pth'

save_dir = os.path.join(base_path, 'checkpoints_tati')
os.makedirs(save_dir, exist_ok=True)

# =====================
# Transforms
# =====================
class LabelTransform():
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).long()
        return mask_tensor

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((720, 1280)),
            # transforms.RandomHorizontalFlip(), # questa non va tolta?
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

transforms_dict = get_transforms()

# =====================
# Dataset & Dataloader
# =====================
train_dataset = GTA5(
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

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

# =====================
# Model, Loss, Optimizer
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_deeplab_v2(
    num_classes=19,
    pretrain=True,
    pretrain_model_path=pretrain_model_path
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.SGD(model.optim_parameters(lr=0.001), momentum=0.9, weight_decay=0.0005)

# =====================
# Poly Learning Rate Scheduler
# =====================
class PolyLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = self.last_epoch
        return [base_lr * (1 - current_iter / self.max_iter) ** self.power for base_lr in self.base_lrs]

# =====================
# Train / Validate
# =====================
def train(epoch, model, train_loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        scheduler.step()  # Step the scheduler to update learning rate

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
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            ious = calculate_iou(predicted, targets, num_classes, ignore_index=255)
            total_ious.append(ious)

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    iou_per_class = torch.tensor(total_ious).nanmean(dim=0)
    miou = iou_per_class.nanmean().item()

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}% | mIoU: {miou:.4f}')
    return val_accuracy, miou

# =====================
# Main Training Loop
# =====================
def main():
    best_model_path = os.path.join(save_dir, 'best_model_deeplab.pth')
    checkpoint_path = os.path.join(save_dir, 'checkpoint_deeplab.pth')

    num_epochs = 50
    save_every = 1
    best_miou = 0
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Ripreso da epoca {checkpoint['epoch']} con mIoU: {best_miou:.4f}")

    # Initialize the poly scheduler
    max_iter = len(train_dataloader) * num_epochs
    scheduler = PolyLR(optimizer, max_iter=max_iter, power=0.9)

    for epoch in range(start_epoch, num_epochs + 1):
        train(epoch, model, train_dataloader, criterion, optimizer, scheduler)
        val_accuracy, miou = validate(model, val_dataloader, criterion)

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
            print(f"📂 Checkpoint salvato all'epoca {epoch}")

        if epoch % 10 == 0:
            model.eval()
            df = benchmark_model(model, image_size=(3, 512, 1024), iterations=100, device=device)
            csv_path = os.path.join(save_dir, f'benchmark_epoch_{epoch}.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 Benchmark salvato: {csv_path}")

    model.load_state_dict(torch.load(best_model_path))
    validate(model, val_dataloader, criterion)

    # Plot Latency and FPS
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
