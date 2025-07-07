import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project-specific modules
from datasets.dataset_cityscapes.cityscapes import CityScapes
from utils.utils import (
    poly_lr_scheduler,
    set_seed,
    get_transforms,
    decode_segmap
)
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from utils.metrics import benchmark_model, calculate_iou, save_metrics_on_wandb

# ================================
# Set Reproducibility
# ================================
set_seed(42)

# ================================
# Environment: Colab or Local
# ================================
print("📍 Environment: Colab")
base_path = '/content/drive/MyDrive/Project_MLDL'
data_dir = '/content/MLDL_SS/Cityscapes/Cityspaces'
pretrain_model_path = '/content/MLDL_SS/deeplabv2_weights.pth'
save_dir = os.path.join(base_path, 'checkpoints_deeplabv2')
os.makedirs(save_dir, exist_ok=True)

# ================================
# Label transformation class (resizing only)
# ================================
class LabelTransform():
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return torch.as_tensor(mask, dtype=torch.long)

# ================================
# Dataset and Dataloaders
# ================================
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

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

# ================================
# Model, Loss and Optimizer Setup
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_deeplab_v2(
    num_classes=19,
    pretrain=True,
    pretrain_model_path=pretrain_model_path
).to(device)

# Class weights for unbalanced datasets
class_weights = torch.tensor([
    2.6, 6.9, 3.5, 3.6, 3.6, 3.8, 3.4, 3.5, 5.1, 4.7,
    6.2, 5.2, 4.9, 3.6, 4.3, 5.6, 6.5, 7.0, 6.6
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
optimizer = optim.SGD(model.optim_parameters(lr=1e-3), momentum=0.9, weight_decay=0.0005)

num_epochs = 50
max_iter = num_epochs * len(train_dataloader)

# ================================
# Training Function
# ================================
def train(epoch, model, train_loader, criterion, optimizer, init_lr):
    model.train()
    running_loss = 0.0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (inputs, targets) in loop:
        current_iter = epoch * len(train_loader) + batch_idx
        poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    mean_loss = running_loss / len(train_loader)
    lr = optimizer.param_groups[0]['lr']

    # Save model checkpoint
    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss,
    }, model_save_path)

    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(model_save_path)
    wandb.log({"epoch": epoch, "loss": mean_loss, "lr": lr}, step=epoch)
    wandb.log_artifact(artifact)

    return mean_loss

# ================================
# Validation Function
# ================================
def validate(model, val_loader, criterion, num_classes=19, epoch=0):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    loss_values, accuracy_values = [], []
    total_intersection = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

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

            if batch_idx == 0:
                # Visualize first batch
                img_tensor = inputs[0].cpu()
                gt_vis = targets[0].cpu().numpy()
                pred_vis = predicted[0].cpu().numpy()

                # De-normalize image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img_dn = img_tensor * std + mean
                img_np = img_dn.permute(1, 2, 0).numpy()

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(img_np)
                axes[0].set_title("Input Image")
                axes[0].axis('off')
                axes[1].imshow(decode_segmap(gt_vis))
                axes[1].set_title("GT (Colored)")
                axes[1].axis('off')
                axes[2].imshow(decode_segmap(pred_vis))
                axes[2].set_title("Prediction")
                axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(f"validation_epoch_{epoch}.png")
                plt.close()
                wandb.log({"validation_image": wandb.Image(fig)}, step=epoch)

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    iou_per_class = total_intersection / total_union
    miou = torch.nanmean(iou_per_class).item()

    if epoch == 50:
        bench_results = benchmark_model(model)
    else:
        bench_results = {k: None for k in ['mean_latency','mean_fps','num_flops','trainable_params']}

    return {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'miou': miou,
        'iou_per_class': iou_per_class,
        'loss_values': loss_values,
        'accuracy_values': accuracy_values,
        **bench_results
    }

# ================================
# Main Training Loop
# ================================
def main():
    checkpoint_path = os.path.join(save_dir, 'checkpoint_deeplabv2.pth')
    var_model = "Deeplabv2" 
    init_lr = 1e-3
    best_miou = 0
    start_epoch = 1
    project_name = f"{var_model}_official"

    # Resume training from local checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Resumed from epoch {checkpoint['epoch']} with mIoU: {best_miou:.4f}")
    
    wandb.init(
        project=project_name,
        entity="mldl-semseg-politecnico-di-torino",
        name=f"run_{var_model}",
        id='we5mwjaw',
        resume="allow"
    )
    print("🛰️ Wandb initialized")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train(epoch, model, train_dataloader, criterion, optimizer, init_lr)
        val_metrics = validate(model, val_dataloader, criterion, epoch=epoch)
        save_metrics_on_wandb(epoch, train_loss, val_metrics)

        # Save checkpoint locally
        checkpoint_data = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_miou': val_metrics['miou'],
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved at {checkpoint_path}")

    # Final evaluation
    validate(model, val_dataloader, criterion)

if __name__ == "__main__":
    main()
