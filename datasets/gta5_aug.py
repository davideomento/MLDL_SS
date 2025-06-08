import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numpy as np

class GTA5_aug(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform  # ora dovrebbe trasformare immagine+maschera insieme
        self.target_transform = target_transform

        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')

        self.image_paths = []
        self.label_paths = []

        for file_name in os.listdir(self.image_dir):
            if file_name.endswith('.png'):
                self.image_paths.append(os.path.join(self.image_dir, file_name))
                self.label_paths.append(os.path.join(self.label_dir, file_name))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.label_paths[idx]))

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']


        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask
