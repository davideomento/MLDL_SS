import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch import nn
from datasets.cityscapes import CityScapes
from metrics import benchmark_model, calculate_iou
from models.bisenet.build_bisenet import 
from tqdm import tqdm
import random
import numpy as np

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
        # Set ignore index for 255
        mask_tensor[mask_tensor == 255] = 0
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

root_cityscapes = './Cityscapes/Cityspaces'
transforms_dict = get_transforms()

train_dataset = CityScapes(
    root_dir=root_cityscapes,
    split='train',
    transform=transforms_dict['train'],
    target_transform=LabelTransform()
)

val_dataset = CityScapes(
    root_dir=root_cityscapes,
    split='val',
    transform=transforms_dict['val'],
    target_transform=LabelTransform()
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=6)

# =====================
# Model, Loss, Optimizer
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_deeplab_v2(
    num_classes=19,
    pretrain=True,
    pretrain_model_path='/content/MLDL_SS/deeplabv2_weights.pth'
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# =====================
# Train / Validate
# =====================

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)

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
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

            ious = calculate_iou(predicted, targets, num_classes)
            total_ious.append(ious)

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total

    iou_per_class = torch.tensor(total_ious).nanmean(dim=0)
    miou = iou_per_class.nanmean().item()

    print(f'Validation Loss: {val_loss:.6f} | Acc: {val_accuracy:.2f}% | mIoU: {miou:.4f}')
    return val_accuracy, miou

# =====================
# Main Training Function
# =====================

def main():
    num_epochs = 10
    best_miou = 0

    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_dataloader, criterion, optimizer)
        val_accuracy, miou = validate(model, val_dataloader, criterion)

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved with mIoU: {miou:.4f}')

    # Final Evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    validate(model, val_dataloader, criterion)

    # Benchmarking
    model.eval()
    model.to(device)
    df = benchmark_model(model, image_size=(3, 512, 1024), iterations=200, device=device)

    # Save Benchmark Results
    csv_path = 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✔ Risultati salvati in: {csv_path}")

    # Latency Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df['iteration'], df['latency_s'], label='Latency (s)')
    plt.title('Latency per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Latency (s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # FPS Plot
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