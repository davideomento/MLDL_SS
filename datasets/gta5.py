# TODO: implement here your custom dataset class for GTA5

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GTA5(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)

        self.image_paths = []
        self.label_paths = []

        # Loop over images in the images directory and match the label
        for file_name in os.listdir(self.image_dir):
            if file_name.endswith('.png'):  # Assuming PNG format for images
                self.image_paths.append(os.path.join(self.image_dir, file_name))
                label_file = file_name.replace('.png', '_label.png')  # Adjust based on label format
                self.label_paths.append(os.path.join(self.label_dir, label_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label
