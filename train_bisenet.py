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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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
data_dir = '/content/MLDL_SS/Cityscapes/Cityspaces'
save_dir = os.path.join(base_path, 'checkpoints_tati')
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

# Trasformazione per l'immagine
img_transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # Resize fisso
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# Trasformazione per la mask (solo resize, no toTensor, no normalize)
def mask_transform(mask):
    return F.resize(mask, (512, 1024), interpolation=F.InterpolationMode.NEAREST)

def get_transforms():
    return {
        'train': (img_transform, mask_transform),
        'val': (img_transform, mask_transform)
    }
'''
def get_transforms():
    train_transform = A.Compose([
        A.OneOf([
            A.Resize(height=int(512 * s), width=int(1024 * s))
            for s in [0.75, 1.0, 1.5, 1.75, 2.0]
        ], p=1.0),
        
        A.PadIfNeeded(min_height=512, min_width=1024, border_mode=0),  # padding se resize più piccola
        A.RandomCrop(height=512, width=1024),  # crop fisso
        A.HorizontalFlip(p=0.5),
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
    }'''






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

dataset_train_size = len(train_dataset)
subset_train_size = int(0.01 * dataset_train_size)
random_indices = np.random.permutation(dataset_train_size)[:subset_train_size]
train_subset = Subset(train_dataset, random_indices)

dataset_val_size = len(val_dataset)
subset_val_size = int(0.1 * dataset_val_size)
random_indices = np.random.permutation(dataset_val_size)[:subset_val_size]
val_subset = Subset(val_dataset, random_indices)

train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2)

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

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
#dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-2, weight_decay=1e-4, momentum=0.9)

num_epochs = 50
max_iter = num_epochs* len(train_dataloader)

# =====================
# Training & Validation
# =====================
def train(epoch, model, train_loader, criterion, optimizer, init_lr):
    model.train()
    running_loss = 0.0
    batch_idx = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    current_iter = epoch * len(train_loader) + batch_idx
    poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter)

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        alpha = 1
        if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
            main_out, aux1_out, aux2_out = outputs
            loss = (
                criterion(main_out, targets)
                + alpha * criterion(aux1_out, targets)
                + alpha * criterion(aux2_out, targets)
            )
        else:
            loss = criterion(outputs, targets)

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
    })

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
            loss = criterion(outputs, targets)

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
                plt.savefig(f"validation_epoch_{epoch}.png")
                plt.close()
                wandb.log({"validation_image": wandb.Image(fig)})
                tqdm.write(f"Validation image saved for epoch {epoch}")


    # Calcolo delle metriche per epoca
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



# Modificare la funzione main per raccogliere e salvare i dati
def main():
    checkpoint_path = os.path.join(save_dir, 'checkpoint_bisenet.pth')
    var_model = "bisenet" 
    best_miou = 0
    start_epoch = 1
    init_lr = 2.5e-2
    project_name = f"{var_model}provabisenet"

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
    validate(model, val_dataloader, criterion)

'''def plot_metrics(metrics_data):
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

'''

if __name__ == "__main__":
    main()
