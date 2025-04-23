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


# =====================
# Dataset Definition
# =====================
class CityScapes(Dataset):
    def _init_(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)

        self.image_paths = []
        self.label_paths = []

        for city in os.listdir(self.image_dir):
            city_path = os.path.join(self.image_dir, city)
            for file_name in os.listdir(city_path):
                if file_name.endswith('_leftImg8bit.png'):
                    self.image_paths.append(os.path.join(city_path, file_name))
                    label_file = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    self.label_paths.append(os.path.join(self.label_dir, city, label_file))

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label