import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class CityScapes(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split  # 'train', 'val', 'test'
        self.transform = transform
        self.target_transform = target_transform

        # Imposta i percorsi per le immagini e le maschere in base alla variabile `split`
        image_dir = os.path.join(root_dir, 'leftImg8bit', split)
        mask_dir = os.path.join(root_dir, 'gtFine', split)

        # Carica le immagini e le maschere
        self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"]

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask.long()

    def __len__(self):
        return len(self.images)
