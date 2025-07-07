import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F  
import wandb
import albumentations as A
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.STDC.stdc_model import *
from albumentations.pytorch import ToTensorV2
from datasets.dataset_cityscapes.cityscapes_aug import CityScapes_aug
from utils.metrics import benchmark_model, calculate_iou, save_metrics_on_wandb
from utils.utils import decode_segmap, load_pretrained_backbone, poly_lr_scheduler, set_seed, get_detail_target, DetailLoss

# =====================
# Set Seed
# =====================
set_seed(42)

# ================================
# Environment: Colab or Local
# ================================
print("📍 Ambiente: Colab (Drive)")
base_path = '/content/drive/MyDrive/Project_MLDL'
data_dir = '/content/MLDL_SS/Cityscapes/Cityspaces'
save_dir = os.path.join(base_path, 'checkpoints_STDC2_provatati')
os.makedirs(save_dir, exist_ok=True)


# =====================
# Label Transform
# =====================
class LabelTransform():
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        return torch.as_tensor(mask, dtype=torch.long)          


def get_transforms():
    train_transform = A.Compose([
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5),
        #A.RandomScale(scale_limit=(0.125, 1.5), p=0.5),  # Slight scaling
        #A.RandomCrop(height=512, width=1024, p=0.5),  # Random crop to maintain size
        A.HorizontalFlip(p=0.5),  # Flip horizontally
        A.Resize(height=512, width=1024),

        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        #A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Low intensity
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),  # Subtle color variation
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    

    val_transform = A.Compose([
        A.Resize(height=512, width=1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return {
        'train': train_transform,
        'val': val_transform
    }



# ================================
# Dataset and Dataloaders
# ================================
transforms_dict = get_transforms()
label_transform = LabelTransform()

train_dataset = CityScapes_aug(
    root_dir=data_dir,
    split='train',
    transform=transforms_dict['train'],
    target_transform=label_transform
)

val_dataset = CityScapes_aug(
    root_dir=data_dir,
    split='val',
    transform=transforms_dict['val'],
    target_transform=label_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# ================================
# Model, Loss and Optimizer Setup
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STDC_Seg(num_classes=19, backbone='STDC2', use_detail=True).to(device)




class_weights = torch.tensor([
    2.6, 6.9, 3.5, 3.6, 3.6, 3.8, 3.4, 3.5, 5.1, 4.7,
    6.2, 5.2, 4.9, 3.6, 4.3, 5.6, 6.5, 7.0, 6.6
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

num_epochs = 50
max_iter = num_epochs


# ================================
# Training Function
# ================================
def train(epoch, model, train_loader, criterion, optimizer, init_lr, λ=1.0):
    model.train()

    running_loss = 0.0
    batch_idx = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    current_iter = epoch
    poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter)
    seg_loss_fn = criterion
    detail_criterion = DetailLoss()

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        optimizer.zero_grad()
        output = model(inputs)

        if isinstance(output, tuple):  # con detail head
            seg_out, detail_map = output
            seg_out = torch.nn.functional.interpolate(seg_out, size=targets.shape[1:], mode='bilinear', align_corners=False)
            loss_seg = seg_loss_fn(seg_out, targets)

            detail_target = get_detail_target(targets)
            detail_map = F.interpolate(detail_map, size=detail_target.shape[-2:], mode='bilinear', align_corners=False)
            loss_detail = detail_criterion(detail_map, detail_target)
            
            loss = loss_seg + λ * loss_detail
        else:
            output = torch.nn.functional.interpolate(output, size=targets.shape[1:], mode='bilinear', align_corners=False)
            loss = seg_loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    # ⬇️ Salvataggio modello e logging wandb dopo il training dell'epoca
    mean_loss = running_loss / len(train_loader)
    lr = optimizer.param_groups[0]['lr']  # Prende il learning rate corrente

    print("Saving the model")
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

    print(f"Model saved for epoch {epoch}")

    return mean_loss



# ================================
# Validation Function
# ================================
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
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # usa solo l'output semantico
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

            if batch_idx == 0:
                img_tensor = inputs[0].cpu()
                gt_vis = targets[0].cpu().numpy()
                pred_vis = predicted[0].cpu().numpy()

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img_dn = img_tensor * std + mean
                img_np = img_dn.permute(1,2,0).numpy()

                label_path = val_dataset.label_paths[batch_idx]
                color_path = label_path.replace('_gtFine_labelTrainIds.png', '_gtFine_color.png')
                color_img = Image.open(color_path)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(img_np)
                axes[0].set_title("Input Image")
                axes[0].axis('off')

                axes[1].imshow(decode_segmap(gt_vis))  # usa colormap ufficiale
                axes[1].set_title("GT (Colored)")
                axes[1].axis('off')

                axes[2].imshow(decode_segmap(pred_vis))
                axes[2].set_title("Prediction")
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(f"validation_epoch_{epoch}.png")
                plt.close()
                wandb.log({"validation_image": wandb.Image(fig)}, step=epoch)
                tqdm.write(f"Validation image saved for epoch {epoch}")


    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    iou_per_class = total_intersection / total_union
    miou = torch.nanmean(iou_per_class).item()

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}% | mIoU: {miou:.4f}')

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
    checkpoint_path = os.path.join(save_dir, 'checkpoints.pth')
    pretrained_backbone_path = '/content/drive/MyDrive/checkpoints/STDC2-Seg/model_maxmIOU50.pth'  # metti qui il path corretto

    var_model = "STDC2"
    best_miou = 0
    start_epoch = 1
    init_lr = 2.5e-2
    project_name = f"{var_model}_pretrained_weight_provatati"

    load_pretrained_backbone(model, pretrained_backbone_path,device)
    # Debug: verifica se il backbone è stato inizializzato
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Totale parametri: {total_params/1e6:.2f}M | Trainabili: {trainable_params/1e6:.2f}M")


    # 🔹 Ripristina da checkpoint locale se esiste
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Ripreso da epoca {checkpoint['epoch']} con mIoU: {best_miou:.4f}")

    # 🔹 Inizializza wandb una sola volta
    wandb.init(
        project=project_name,
        entity="mldl-semseg-politecnico-di-torino",
        name=f"run_{var_model}",
        id=f"{var_model}_run",
        resume="allow"
    )
    print("🛰️ Wandb inizializzato")

    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss = train(epoch, model, train_dataloader, criterion, optimizer, init_lr)
        # Validation and Metrics
        val_metrics = validate(model, val_dataloader, criterion, epoch=epoch)
        save_metrics_on_wandb(epoch, train_loss, val_metrics)

        # 🔹 Salva il checkpoint localmente
        checkpoint_data = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'best_miou': val_metrics['miou'],
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint salvato a {checkpoint_path}")
    
    # Validazione finale
    validate(model, val_dataloader, criterion, epoch)


if __name__ == "__main__":
    main()