import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class GTA5_aug(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform  # Transformations that apply jointly to image and mask (e.g., Albumentations)
        self.target_transform = target_transform  # Optional transformations only for the mask

        # Define paths to images and labels folders
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')

        self.image_paths = []
        self.label_paths = []

        # Collect paired image and label file paths
        for file_name in os.listdir(self.image_dir):
            if file_name.endswith('.png'):
                self.image_paths.append(os.path.join(self.image_dir, file_name))
                self.label_paths.append(os.path.join(self.label_dir, file_name))

    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask as numpy arrays
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.label_paths[idx]))  # Label mask, usually single channel

        # If transform is provided, apply it jointly on image and mask (expects dict input)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Apply any mask-only transformations if provided
        if self.target_transform:
            mask = self.target_transform(mask)

        # Return the processed image and mask
        return img, mask
