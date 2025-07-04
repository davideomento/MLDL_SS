import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets.gta5_labels import GTA5Labels_TaskCV2017
from torchvision.transforms import functional as F

def to_tensor_no_normalization(pic):
    """
    Converte un'immagine PIL (o ndarray) in un tensore torch senza normalizzazione.
    Utile per trasformare maschere di segmentazione con etichette discrete.
    """
    if isinstance(pic, torch.Tensor):
        return pic.long()
    if isinstance(pic, Image.Image):
        return F.pil_to_tensor(pic).squeeze(0).long()
    raise TypeError(f"Input non supportato per to_tensor_no_normalization: {type(pic)}")


def transform_gta_to_cityscapes_label(mask):
    """
    Mappa gli ID delle classi GTA5 a quelli di Cityscapes.
    Le etichette non mappate sono settate a 255 (ignore index).
    """
    id_to_trainid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
        22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
        31: 16, 32: 17, 33: 18
    }
    mapped = torch.full_like(mask, 255)
    for gta_id, train_id in id_to_trainid.items():
        mapped[mask == gta_id] = train_id
    return mapped


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
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        # Se self.transform è una tupla (img_transform, mask_transform)
        if self.transform:
            img_transform, _ = self.transform
            img = img_transform(img)

        label_tensor = F.pil_to_tensor(label).squeeze(0).long()

        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return img, label_tensor
