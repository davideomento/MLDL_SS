# TODO: Define here your training and validation loops.
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch import nn
from download_dataset import *
from datasets.cityscapes import *
from models.deeplabv2.deeplabv2 import get_deeplab_v2





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
def transform():
    # Define transformations for the dataset
    transform = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop((512, 1024)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        'val' : transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.CenterCrop((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
    }
    return transform



# =====================
# Dataset & Dataloader
# =====================
root_cityscapes = './Cityscapes'

train_dataset = CityScapes(
    root=root_cityscapes,
    split='train',
    mode='fine',
    target_type='semantic',
    transform=transform,
    target_transform=LabelTransform()
)

val_dataset = CityScapes(
    root=root_cityscapes,
    split='val',
    mode='fine',
    target_type='semantic',
    transform=transform,
    target_transform=LabelTransform()
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
    pretrain_model_path='deeplabv2_weights.pth'
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# =====================
# Training Loop (bozza)
# =====================
num_epochs = 10
best_acc = 0

#Training
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


# Validation
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