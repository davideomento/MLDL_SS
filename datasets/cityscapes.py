import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform  # Expected to be a tuple: (image_transform, mask_transform)
        self.target_transform = target_transform  # Optional transform applied only to mask

        # Define paths to images and labels folders based on dataset root and split
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)

        self.image_paths = []
        self.label_paths = []

        # Loop over city folders inside image directory
        for city in sorted(os.listdir(self.image_dir)):
            city_img_dir = os.path.join(self.image_dir, city)
            city_label_dir = os.path.join(self.label_dir, city)
            if not os.path.isdir(city_img_dir):
                continue  # Skip non-directory entries

            # Loop over all image files in city directory
            for file_name in sorted(os.listdir(city_img_dir)):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(city_img_dir, file_name)
                    # Construct corresponding ground truth label filename
                    label_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    lbl_path = os.path.join(city_label_dir, label_name)
                    # Append paths if label file exists
                    if os.path.isfile(lbl_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(lbl_path)

    def __len__(self):
        # Return total number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask with PIL
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.label_paths[idx]).convert("L")

        # Expect transform to be a tuple: (img_transform, mask_transform)
        img_transform, mask_transform = self.transform

        # Apply transformations separately to image and mask
        img = img_transform(img)
        mask = mask_transform(mask)

        # Convert mask to torch tensor of type long for loss functions like CrossEntropyLoss
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Optionally apply further transforms only to mask (e.g. encoding)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Return transformed image and mask tensor
        return img, mask
