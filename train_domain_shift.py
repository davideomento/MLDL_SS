import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.gta5 import *
import torch.nn.functional as nnF
from models.discriminator import FCDiscriminator


#from monai.losses import DiceLoss
from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import BiSeNet
from models.bisenet.build_contextpath import build_contextpath
from metrics import benchmark_model, calculate_iou, save_metrics_on_wandb
from utils import poly_lr_scheduler


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
data_dir_train = '/content/MLDL_SS/GTA5'
data_dir_val = '/content/MLDL_SS/Cityscapes/Cityspaces'    
save_dir = os.path.join(base_path, 'checkpoints_provadversarial')
os.makedirs(save_dir, exist_ok=True)


# =====================
# Label Transform
# =====================
class LabelTransform:
    def __init__(self, size, id_conversion=True):
        self.size = size
        self.id_conversion = id_conversion

    def __call__(self, mask):
        # mask: torch.Tensor (H, W) o PIL.Image se ancora non convertita
        if not isinstance(mask, torch.Tensor):
            mask = to_tensor_no_normalization(mask)

        if self.id_conversion:
            mask = transform_gta_to_cityscapes_label(mask)  # solo per GTA5

        mask = mask.unsqueeze(0).unsqueeze(0).float()  # shape (1,1,H,W)
        mask = nnF.interpolate(mask, size=self.size, mode='nearest')
        mask = mask.squeeze().long()
        return mask


###############

# Trasformazione per l'immagine
img_transform_gta = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

img_transform_cs = transforms.Compose([
    transforms.Resize((512, 1024)),  # Resize fisso
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def get_transforms():
    return {
        'train': (img_transform_gta, lambda mask: mask),  # Dummy mask_transform, serve per compatibilità
        'val': (img_transform_cs, lambda mask: mask)
    }

    

# =====================
# Dataset & Dataloader
# =====================
transforms_dict = get_transforms()
label_transform_train = LabelTransform(size=(720, 1280), id_conversion=True)  # GTA5
label_transform_val   = LabelTransform(size=(512, 1024), id_conversion=False)  # Cityscapes

train_dataset = GTA5(
    root_dir=data_dir_train,
    transform=transforms_dict['train'] ,
    target_transform=label_transform_train
)


val_dataset = CityScapes(
    root_dir=data_dir_val,
    split='val',
    transform=transforms_dict['val'],
    target_transform=label_transform_val
)

# Dataloader per il dominio sorgente (GTA5)
source_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

# Dataloader per il dominio target (Cityscapes, ma senza label supervisionate)
target_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2)


# output stride e numero classi corrispondono all'output di BiSeNet
num_classes = 19
discriminator = FCDiscriminator(input_channels=num_classes).cuda()

# =====================
# Model Setup
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Costruisci context path e BiSeNet in modo modulare
context_path = build_contextpath(
    name='resnet18',
)

model = BiSeNet(num_classes=19, context_path='resnet18').cuda()

class_weights = torch.tensor([
    2.6, 6.9, 3.5, 3.6, 3.6, 3.8, 3.4, 3.5, 5.1, 4.7,
    6.2, 5.2, 4.9, 3.6, 4.3, 5.6, 6.5, 7.0, 6.6
], dtype=torch.float).to(device)


optimizer_seg = torch.optim.SGD(model.parameters(), lr=2.5e-4)
optimizer_disc = torch.optim.SGD(discriminator.parameters(), lr=1e-4)

criterion_seg = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
criterion_adv = nn.BCEWithLogitsLoss()


num_epochs = 50
max_iter = num_epochs

# =====================
# Training & Validation
# =====================
def train(epoch, model, source_dataloader, target_dataloader, criterion_seg, criterion_adv, optimizer_seg, optimizer_disc, lr_seg, lr_disc):
    model.train()
    discriminator.train()

    running_loss = 0.0
    loop = tqdm(zip(source_dataloader, target_dataloader), total=min(len(source_dataloader), len(target_dataloader)), desc=f"Epoch {epoch}")

    poly_lr_scheduler(optimizer_seg, lr_seg, epoch, max_iter)
    poly_lr_scheduler(optimizer_disc, lr_disc, epoch, max_iter)

    for batch_idx, ((inputs_s, targets_s), (inputs_t, _)) in enumerate(loop):
        inputs_s, targets_s = inputs_s.to(device), targets_s.to(device)
        inputs_t = inputs_t.to(device)

        # Segmentation loss
        optimizer_seg.zero_grad()
        outputs_s = model(inputs_s)
        alpha = 1
        if isinstance(outputs_s, (tuple, list)) and len(outputs_s) == 3:
            main_out, aux1_out, aux2_out = outputs_s
            loss_seg = (
                criterion_seg(main_out, targets_s)
                + alpha * criterion_seg(aux1_out, targets_s)
                + alpha * criterion_seg(aux2_out, targets_s)
            )
        else:
            loss_seg = criterion_seg(outputs_s, targets_s)

        loss_seg.backward()
        optimizer_seg.step()

        # Adversarial loss
        optimizer_disc.zero_grad()
        outputs_s_detached = outputs_s[0].detach()  # Disattiva il gradiente per l'output della rete principale
        outputs_t = model(inputs_t)
        outputs_t_detached = outputs_t[0].detach()  # Disattiva il gradiente per l'output della rete principale

        pred_s = discriminator(outputs_s_detached)
        pred_t = discriminator(outputs_t_detached)
        loss_disc = criterion_adv(pred_s, torch.ones_like(pred_s)) + criterion_adv(pred_t, torch.zeros_like(pred_t))
        loss_disc.backward()
        optimizer_disc.step()

        optimizer_seg.zero_grad()
        outputs_t = model(inputs_t)
        pred_t = discriminator(outputs_t[0])
        loss_adv = criterion_adv(pred_t, torch.ones_like(pred_t))  # il segmentatore vuole che il discriminatore "pensi" target come source
        loss_adv.backward()
        optimizer_seg.step()        

        running_loss += loss_seg.item() + loss_adv.item() + loss_disc.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    # ⬇️ Salvataggio modello e logging wandb dopo il training dell'epoca
    mean_loss = running_loss / len(loop)
    lr_seg = optimizer_seg.param_groups[0]['lr']  # Prende il learning rate corrente
    lr_disc = optimizer_disc.param_groups[0]['lr']  # Prende il learning rate corrente
    print("Saving the model")
    wandb.log({
        "epoch": epoch,
        "loss": mean_loss,
        "lr_seg": lr_seg,
        "lr_disc": lr_disc
    })

    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_seg_state_dict': optimizer_seg.state_dict(),
        'optimizer_disc_state_dict': optimizer_disc.state_dict(),
        'loss': mean_loss,
    }, model_save_path)

    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)

    print(f"Model saved for epoch {epoch}")

    return mean_loss



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



def validate(model, target_loader, criterion_seg, epoch, num_classes=19):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    loss_values = []
    accuracy_values = []
    total_intersection = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)


    with torch.no_grad():
        loop = tqdm(enumerate(target_loader), total=len(target_loader), desc="Validating")
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion_seg(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            inter, uni = calculate_iou(predicted, targets, num_classes, ignore_index=255)
            total_intersection += inter
            total_union += uni


            # Salvataggio della loss e accuratezza per epoca
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

                # ===> Carica immagine _color dal filesystem
                # 1. Prendi il percorso della label
                label_path = val_dataset.label_paths[batch_idx]
                # 2. Costruisci path della versione _color
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
                fname = f"{save_dir}/img_gt_pred_gtcolor_epoch_{epoch}.png"
                plt.savefig(fname)
                plt.close()


    # Calcolo delle metriche per epoca
    val_loss /= len(target_loader)
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



# Modificare la funzione main per raccogliere e salvare i dati
def main():
    checkpoint_path = os.path.join(save_dir, 'checkpoint_bisenet_adversarial.pth')
    var_model = "bisenet_adversarial" 
    best_miou = 0
    start_epoch = 1
    lr_seg = 2.5e-2
    lr_disc = 1e-4
    project_name = f"{var_model}_prova"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer_seg.load_state_dict(checkpoint['optimizer_state_seg'])
        optimizer_disc.load_state_dict(checkpoint['optimizer_state_disc'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch'] + 1
        print(f"✔ Ripreso da epoca {checkpoint['epoch']} con mIoU: {best_miou:.4f}")

    # 🔹 Inizializza wandb una sola volta
    wandb.init(
        project=project_name,
        entity="mldl-semseg-politecnico-di-torino",
        name=f"run_{var_model}",
        #inserire id
        resume="allow"
    )
    print("🛰️ Wandb inizializzato")

    
    for epoch in range(start_epoch, num_epochs + 1):

        # Training
        train_loss = train(epoch, model, source_dataloader, target_dataloader, criterion_seg, criterion_adv, optimizer_seg, optimizer_disc, lr_seg, lr_disc)
        
        # Validation and Metrics
        val_metrics = validate(model, target_dataloader, criterion_seg, epoch=epoch)
        save_metrics_on_wandb(epoch, train_loss, val_metrics)

        # 🔹 Salva il checkpoint localmente
        checkpoint_data = {
            'model_state': model.state_dict(),
            'optimizer_state_seg': optimizer_seg.state_dict(),
            'optimizer_state_disc': optimizer_disc.state_dict(),
            'epoch': epoch,
            'best_miou': val_metrics['miou'],
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint salvato a {checkpoint_path}")
    
    # Validazione finale
    validate(model, target_dataloader, criterion_seg, epoch=num_epochs)



if __name__ == "__main__":
    main()
