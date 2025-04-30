import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, check_first_n=3):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.check_first_n = check_first_n  # Numero di immagini da controllare

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

        self.images_checked = 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        if self.target_transform:
            label = self.target_transform(label)

        # Controlla per le prime N immagini
        if self.images_checked < self.check_first_n:
            self.check_labels(label)
            self.images_checked += 1

        if self.transform:
            img = self.transform(img)


        return img, label

    def check_labels(self, label):
        unique_labels = torch.unique(torch.tensor(np.array(label)))
        print(f"Unique labels: {unique_labels}")
        if torch.any(unique_labels >= 19) or torch.any(unique_labels < 0):
            print(f"Warning: Invalid labels found!")
        else:
            print("All labels are valid.")


