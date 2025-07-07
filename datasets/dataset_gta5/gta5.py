import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from project.datasets.gta.gta5_labels import GTA5Labels_TaskCV2017
from torchvision.transforms import functional as F


def to_tensor_no_normalization(pic):
    """
    Convert a PIL Image or ndarray to a torch tensor without normalization.
    Useful for segmentation masks with discrete labels, 
    since we want to keep label values as integers without scaling.
    """
    if isinstance(pic, torch.Tensor):
        return pic.long()  # If already tensor, convert dtype to long (int64)
    if isinstance(pic, Image.Image):
        # Convert PIL Image to tensor and remove channel dim (mask is single channel)
        return F.pil_to_tensor(pic).squeeze(0).long()
    raise TypeError(f"Unsupported input type for to_tensor_no_normalization: {type(pic)}")


def transform_gta_to_cityscapes_label(mask):
    """
    Maps GTA5 class IDs in the mask to Cityscapes train IDs.
    Any GTA class ID not mapped is set to 255 (ignore index for loss).
    Args:
        mask (Tensor): tensor with GTA class IDs.
    Returns:
        Tensor: mapped mask with Cityscapes class IDs or 255 for ignore.
    """
    # Mapping GTA IDs to Cityscapes train IDs (example mapping)
    id_to_trainid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
        22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
        31: 16, 32: 17, 33: 18
    }
    # Create a tensor filled with 255 (ignore index)
    mapped = torch.full_like(mask, 255)
    # Replace the values according to the mapping
    for gta_id, train_id in id_to_trainid.items():
        mapped[mask == gta_id] = train_id
    return mapped


def build_gta5_to_cityscapes_mapping():
    """
    Builds a mapping dictionary from GTA5 label IDs to sequential indices.
    Uses the GTA5Labels_TaskCV2017 class to get all labels.
    Returns:
        dict: mapping from GTA5 label ID to index in list.
    """
    gta_labels = GTA5Labels_TaskCV2017()
    mapping = {}
    for new_index, label in enumerate(gta_labels.list_):
        mapping[label.ID] = new_index
    return mapping


class GTA5(Dataset):
    """
    Dataset class for GTA5 dataset.
    Assumes folder structure:
    root_dir/
        images/
            *.png
        labels/
            *.png  (same filenames as images)
    No train/val/test split in GTA5 by default.
    """
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        # Build the label ID mapping dictionary
        self.id_mapping = build_gta5_to_cityscapes_mapping()

        self.image_dir = os.path.join(root_dir, 'images')  # GTA5 has no train/val split
        self.label_dir = os.path.join(root_dir, 'labels')

        self.image_paths = []
        self.label_paths = []

        # Collect all image and label file paths
        for file_name in os.listdir(self.image_dir):
            if file_name.endswith('.png'):
                self.image_paths.append(os.path.join(self.image_dir, file_name))
                label_file = file_name  # labels have the same filename
                self.label_paths.append(os.path.join(self.label_dir, label_file))

    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and label at index idx
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx])

        # If transform is provided as (img_transform, mask_transform), apply only img_transform here
        if self.transform:
            img_transform, _ = self.transform
            img = img_transform(img)

        # Convert label PIL image to tensor of long integers (class IDs)
        label_tensor = F.pil_to_tensor(label).squeeze(0).long()

        # Apply target_transform if provided (e.g. mapping GTA to Cityscapes IDs)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        # Return image tensor and label tensor
        return img, label_tensor
