import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets.gta5_labels import GTA5Labels_TaskCV2017
from torchvision.transforms import functional as F
import albumentations as A

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

    def _map_labels(self, label_tensor):
        mapped = torch.full_like(label_tensor, 255)
        for gta_id, cityscapes_id in self.id_mapping.items():
            mapped[label_tensor == gta_id] = cityscapes_id
        return mapped
    
    def __getitem__(self, idx):
        # 1. Caricamento
        img_path = self.image_paths[idx]
        mask_path = self.label_paths[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 2. Augmentazione e resize dell’immagine + maschera insieme
        if self.transform:
            img_transform, _ = self.transform

            # Se è un Compose di Albumentations
            if isinstance(img_transform, A.core.composition.Compose):
                # Albumentations lavora su ndarray
                data = img_transform(
                    image=np.array(img),
                    mask=np.array(mask)
                )
                img = data['image']
                mask = data['mask']
            else:
                # torchvision transforms: solo sull’immagine
                img = img_transform(img)

        # 3. Mappatura ID GTA → Cityscapes e resize maschera
        # Trasforma la maschera in tensor di interi
        mask_tensor = mask.long()
        mask_tensor = self._map_labels(mask_tensor)

        # LabelTransform fa ID conversion (se richiesto) + resize tensoriale
        if self.target_transform:
            mask_tensor = self.target_transform(mask_tensor)

        return img, mask_tensor
