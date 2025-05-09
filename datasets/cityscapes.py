import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)

        self.image_paths = []
        self.label_paths = []

        for city in os.listdir(self.image_dir):
            city_path = os.path.join(self.image_dir, city)
            for file_name in os.listdir(city_path):
                if file_name.endswith('_leftImg8bit.png'):
                    self.image_paths.append(os.path.join(city_path, file_name))
                    label_file = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    self.label_paths.append(os.path.join(self.label_dir, city, label_file))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask



