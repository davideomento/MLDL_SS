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
    
    # Mappa le etichette GTA5 secondo la tabella ufficiale di Cityscapes
    def map_gta5_labels(label):
        """
        label: tensor numpy 2D con classi GTA5 (fino a 33)
        ritorna: label con solo 19 classi Cityscapes (0-18) + 255 (ignore)
        """
        # Mappa GTA5 → Cityscapes trainId
        gta5_to_cityscapes = {
            7: 0,   # road
            8: 1,   # sidewalk
            11: 2,  # building
            12: 3,  # wall
            13: 4,  # fence
            17: 5,  # pole
            19: 6,  # traffic light
            20: 7,  # traffic sign
            21: 8,  # vegetation
            22: 9,  # terrain
            23:10,  # sky
            24:11,  # person
            25:12,  # rider
            26:13,  # car
            27:14,  # truck
            28:15,  # bus
            31:16,  # train
            32:17,  # motorcycle
            33:18,  # bicycle
        }

        label_copy = 255 * np.ones_like(label, dtype=np.uint8)  # default: ignore
        for gta_id, city_id in gta5_to_cityscapes.items():
            label_copy[label == gta_id] = city_id

        return label_copy

