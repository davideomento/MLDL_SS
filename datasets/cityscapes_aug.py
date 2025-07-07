import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CityScapes_aug(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        Custom dataset for the Cityscapes dataset with Albumentations support.
        
        Args:
            root_dir (str): Root directory containing the dataset.
            split (str): Split type ('train', 'val', or 'test').
            transform (callable, optional): Transformation applied to both image and mask using Albumentations.
            target_transform (callable, optional): Optional transformation only for the mask.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform  # Should apply jointly to image and mask
        self.target_transform = target_transform

        # Set image and label directories based on the split
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)

        # Lists to store full paths of images and labels
        self.image_paths = []
        self.label_paths = []

        # Iterate through all cities (subfolders)
        for city in sorted(os.listdir(self.image_dir)):
            city_img_dir = os.path.join(self.image_dir, city)
            city_label_dir = os.path.join(self.label_dir, city)
            if not os.path.isdir(city_img_dir):
                continue

            # Collect all matching image-label pairs
            for file_name in sorted(os.listdir(city_img_dir)):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(city_img_dir, file_name)
                    # Construct corresponding label filename
                    label_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    lbl_path = os.path.join(city_label_dir, label_name)

                    if os.path.isfile(lbl_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(lbl_path)

    def __len__(self):
        """
        Returns:
            int: Total number of image-mask pairs
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and returns a single image and its segmentation mask.

        Args:
            idx (int): Index of the item

        Returns:
            tuple: (transformed image, mask tensor)
        """
        # Load image and mask using PIL and convert to numpy arrays
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.label_paths[idx]).convert("L"))  # grayscale mask

        if self.transform:
            # Albumentations expects a dict with "image" and "mask"
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Convert mask to torch tensor of type long (required for CrossEntropyLoss)
        mask = torch.as_tensor(mask, dtype=torch.long)

        if self.target_transform:
            # Apply optional mask-only transformation
            mask = self.target_transform(mask)

        return img, mask
