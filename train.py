# TODO: Define here your training and validation loops.

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.datasets import Citysfrom torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from PIL import Imagecapes
import shutil





# =====================
# Transforms
# =====================
#transform = transforms.Compose([
#    transforms.Resize((512, 1024)),
#   transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406],  #corrette perchè pre-addestriamo su Imagenet
#                        std=[0.229, 0.224, 0.225])
#])

class LabelTransform:
    def __init__(self, size=(512, 1024)):
        self.size = size

    def __call__(self, mask):
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return F.pil_to_tensor(mask).squeeze(0).long()


# lab 1 transformations to vislize the images
def transform_1():
    # Define transformations for the dataset
    transform = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        'val' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
    }
    return transform

# lab 2 transformation to contruct the NN
def transform_2():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

# =====================
# Dataset & Dataloader
# =====================
root_cityscapes = '/content/drive/MyDrive/Cityscapes'

train_dataset = Cityscapes(
    root=root_cityscapes,
    split='train',
    mode='fine',
    target_type='semantic',
    transform=transform,
    target_transform=LabelTransform()
)

val_dataset = Cityscapes(
    root=root_cityscapes,
    split='val',
    mode='fine',
    target_type='semantic', # restituisce le maschere di segmentazione semantica
    transform=transform,
    target_transform=LabelTransform() # serve per ridimensionare e convertire le maschere in tensori long, come richiesto dalla CrossEntropyLoss
    #Nota: solo per ricordarci che possiamo trasformare ancora i dati per trainare meglio, ad es. horizontal flip, rotation, ecc. (Lab4)
)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =====================
# Model, Loss, Optimizer
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_deeplab_v2(
    num_classes=19,
    pretrain=True,
    pretrain_model_path='/content/drive/MyDrive/DeepLab_resnet_pretrained_imagenet.pth'
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# =====================
# Training Loop (bozza)
# =====================
num_epochs = 10
best_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)['out']  # 'out' è il campo per le previsioni pixel-wise
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")




#Training loop
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

for epoch in range(1, num_epochs+1): 
    train(epoch, model, train_loader, criterion, optimizer)
    val_accuracy = validate(model, val_loader, criterion)

    # Save the model if the validation accuracy is improved
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Model saved with accuracy: {best_acc:.2f}%')