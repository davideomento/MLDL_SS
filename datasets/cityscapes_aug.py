import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CityScapes_aug(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform  # ora dovrebbe trasformare immagine+maschera insieme
        self.target_transform = target_transform

        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)

        self.image_paths = []
        self.label_paths = []

        for city in sorted(os.listdir(self.image_dir)):
            city_img_dir = os.path.join(self.image_dir, city)
            city_label_dir = os.path.join(self.label_dir, city)
            if not os.path.isdir(city_img_dir):
                continue
            for file_name in sorted(os.listdir(city_img_dir)):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(city_img_dir, file_name)
                    label_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    lbl_path = os.path.join(city_label_dir, label_name)
                    if os.path.isfile(lbl_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(lbl_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.label_paths[idx]).convert("L"))

        if self.transform:
            # Albumentations expects dict: {"image":..., "mask":...}
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Convert mask a torch tensor long (per CrossEntropyLoss)
        mask = torch.as_tensor(mask, dtype=torch.long)

        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask
