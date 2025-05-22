import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets.gta5_labels import GTA5Labels_TaskCV2017
from torchvision.transforms import functional as F


def build_gta5_to_cityscapes_mapping():
    gta_labels = GTA5Labels_TaskCV2017()
    mapping = {}
    for new_index, label in enumerate(gta_labels.list_):
        mapping[label.ID] = new_index
    return mapping


class GTA5(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.id_mapping = build_gta5_to_cityscapes_mapping()

        self.image_dir = os.path.join(root_dir, 'images') #NOTA BENE SENZA SPLIT perchè gta non ha la suddivisione
        self.label_dir = os.path.join(root_dir, 'labels')

        self.image_paths = []
        self.label_paths = []

        for file_name in os.listdir(self.image_dir):
            if file_name.endswith('.png'):
                self.image_paths.append(os.path.join(self.image_dir, file_name))
                label_file = file_name
                self.label_paths.append(os.path.join(self.label_dir, label_file))

    def __len__(self):
        return len(self.image_paths)

    def _map_labels(self, label_tensor):
        mapped = torch.full_like(label_tensor, 255)
        for gta_id, cityscapes_id in self.id_mapping.items():
            mapped[label_tensor == gta_id] = cityscapes_id
        return mapped
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        if self.transform:
            img = self.transform(img)

        #label = F.resize(label, (720, 1280), interpolation=F.InterpolationMode.NEAREST)
        label_tensor = F.pil_to_tensor(label).squeeze(0).long()
        label_tensor = self._map_labels(label_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
            
        return img, label_tensor