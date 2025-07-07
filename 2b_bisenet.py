import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
# Custom modules (already present in the project)
from datasets.dataset_cityscapes.cityscapes import CityScapes
from models.bisenet.build_bisenet import BiSeNet
from models.bisenet.build_contextpath import build_contextpath
from utils.metrics import benchmark_model, calculate_iou, save_metrics_on_wandb
from utils.utils import poly_lr_scheduler, set_seed, decode_segmap

# ================================
# Set reproducibility
# ================================
set_seed(42)

# =====================
# Constants
# =====================
NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Paths
# =====================
print("📍 Environment: Colab (Drive)")
base_path = '/content/drive/MyDrive/Project_MLDL'
data_dir = '/content/MLDL_SS/Cityscapes/Cityspaces'
save_dir = os.path.join(base_path, 'checkpoints_bisenet_official')
os.makedirs(save_dir, exist_ok=True)

# =====================
# Transformations
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
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def mask_transform(mask):
    return F.resize(mask, (512, 1024), interpolation=F.InterpolationMode.NEAREST)

def get_transforms():
    return {
        'train': (img_transform, mask_transform),
        'val': (img_transform, mask_transform)
    }

# =====================
# Data
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
# Model
# =====================
context_path = build_contextpath(name='resnet18')
model = BiSeNet(num_classes=NUM_CLASSES, context_path='resnet18').to(DEVICE)

class_weights = torch.tensor([
    2.6, 6.9, 3.5, 3.6, 3.6, 3.8, 3.4, 3.5, 5.1, 4.7,
    6.2, 5.2, 4.9, 3.6, 4.3, 5.6, 6.5, 7.0, 6.6
], dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-2, weight_decay=1e-4, momentum=0.9)
num_epochs = 50

# =====================
# Training function
# =====================
def train(epoch, model, train_loader, criterion, optimizer, init_lr):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    poly_lr_scheduler(optimizer, init_lr, epoch, num_epochs)

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
            main_out, aux1_out, aux2_out = outputs
            loss = (criterion(main_out, targets) + 
                    criterion(aux1_out, targets) + 
                    criterion(aux2_out, targets))
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    return running_loss / len(train_loader)

# =====================
# Validation function
# =====================
def validate(model, val_loader, criterion, epoch, num_classes=19):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    total_intersection = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            inter, uni = calculate_iou(predicted, targets, num_classes, ignore_index=255)
            total_intersection += inter
            total_union += uni

            if batch_idx == 0:
                img_tensor = inputs[0].cpu()
                gt_vis = targets[0].cpu().numpy()
                pred_vis = predicted[0].cpu().numpy()

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img_dn = img_tensor * std + mean
                img_np = img_dn.permute(1,2,0).numpy()

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(img_np)
                axes[0].set_title("Input Image")
                axes[1].imshow(decode_segmap(gt_vis))
                axes[1].set_title("GT")
                axes[2].imshow(decode_segmap(pred_vis))
                axes[2].set_title("Prediction")
                for ax in axes: ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"validation_epoch_{epoch}.png")
                plt.close()
                wandb.log({"validation_image": wandb.Image(f"validation_epoch_{epoch}.png")}, step=epoch)

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    iou_per_class = total_intersection / total_union
    miou = torch.nanmean(iou_per_class).item()

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}% | mIoU: {miou:.4f}')

    bench_results = benchmark_model(model) if epoch == num_epochs else {
        'mean_latency': None,
        'mean_fps': None,
        'num_flops': None,
        'trainable_params': None
    }

    return {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'miou': miou,
        'iou_per_class': iou_per_class,
        **bench_results
    }

# =====================
# Main function
# =====================
def main():
    checkpoint_path = os.path.join(save_dir, 'checkpoint_bisenet_official.pth')
    model_name = "bisenet"
    best_miou = 0
    start_epoch = 1
    init_lr = 2.5e-2

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Resumed from epoch {checkpoint['epoch']} with mIoU: {best_miou:.4f}")

    # Initialize wandb
    wandb.init(
        project=f"{model_name}_official",
        entity="mldl-semseg-politecnico-di-torino",
        name=f"run_{model_name}",
        id=f"{model_name}_run",
        resume="allow"
    )
    print("🛰️ Wandb initialized")

    # Training and validation loop
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train(epoch, model, train_dataloader, criterion, optimizer, init_lr)
        val_metrics = validate(model, val_dataloader, criterion, epoch=epoch)
        save_metrics_on_wandb(epoch, train_loss, val_metrics)

        checkpoint_data = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_miou': val_metrics['miou']
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
