import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image

# ================================
# Polynomial Learning Rate Scheduler
# ================================
def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, lr_decay_iter=1, power=0.9):
    """
    Adjusts learning rate using polynomial decay schedule.

    Args:
        optimizer: optimizer whose learning rate will be adjusted.
        init_lr: initial base learning rate.
        iter: current iteration number.
        max_iter: maximum number of iterations.
        lr_decay_iter: interval of iterations to decay learning rate (default 1).
        power: polynomial power (controls decay curve shape).

    Returns:
        lr: updated learning rate for current iteration.
    """
    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr

# ================================
# Fast Confusion Matrix Histogram Computation
# ================================
def fast_hist(a, b, n):
    """
    Computes confusion matrix histogram between ground truth labels (a) and predictions (b).

    Args:
        a (ndarray): ground truth labels.
        b (ndarray): predicted labels.
        n (int): number of classes.

    Returns:
        hist (ndarray): n x n confusion matrix, where hist[i, j] counts
                        pixels with true label i predicted as label j.
    """
    k = (a >= 0) & (a < n)  # valid labels mask
    # Using bincount on combined indices to build confusion matrix
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# ================================
# Compute Per-Class IoU from Confusion Matrix
# ================================
def per_class_iou(hist):
    """
    Calculates the Intersection over Union (IoU) for each class based on confusion matrix.

    Args:
        hist (ndarray): confusion matrix (n x n).

    Returns:
        iou (ndarray): IoU for each class.
    """
    epsilon = 1e-5  # small value to avoid division by zero
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

# ================================
# Set Random Seeds for Reproducibility
# ================================
def set_seed(seed=42):
    """
    Set random seeds across various libraries to ensure reproducible experiments.

    Args:
        seed (int): seed value (default 42).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures deterministic behavior for CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================
# Mask Resize Transformation (Nearest Interpolation)
# ================================
def mask_transform(mask):
    """
    Resizes segmentation mask to fixed size with nearest neighbor interpolation
    to preserve discrete label values.

    Args:
        mask (PIL.Image or Tensor): input segmentation mask.

    Returns:
        resized mask.
    """
    return F.resize(mask, (512, 1024), interpolation=F.InterpolationMode.NEAREST)

# ================================
# Compose Image and Mask Transforms for Training and Validation
# ================================
def get_transforms():
    """
    Returns composed transformations for images and masks for training and validation.

    Image transforms include resizing, converting to tensor, and normalization.
    Mask transforms include resizing with nearest neighbor interpolation.

    Returns:
        dict with 'train' and 'val' keys, each containing (img_transform, mask_transform).
    """
    img_transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return {
        'train': (img_transform, mask_transform),
        'val': (img_transform, mask_transform)
    }

# ================================
# Cityscapes Color Palette for Visualization
# ================================
CITYSCAPES_COLORS = [
    (128, 64,128), (244, 35,232), (70, 70, 70), (102,102,156),
    (190,153,153), (153,153,153), (250,170,30), (220,220,0),
    (107,142,35), (152,251,152), (70,130,180), (220,20,60),
    (255,0,0), (0,0,142), (0,0,70), (0,60,100),
    (0,80,100), (0,0,230), (119,11,32)
]

# ================================
# Decode Segmentation Mask to RGB Image
# ================================
def decode_segmap(mask):
    """
    Converts a 2D segmentation mask to a color image using predefined palette.

    Args:
        mask (ndarray): 2D array of class IDs (H x W).

    Returns:
        color_mask (ndarray): RGB image (H x W x 3) where each class label
                              is replaced with its color.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == label_id] = color
    return color_mask

# ================================
# Binary Cross Entropy with Logits Ignoring Specific Index
# ================================
def bce_with_logits_ignore(pred, target, ignore_index=255):
    """
    Computes binary cross-entropy loss with logits, ignoring pixels with ignore_index label.

    Args:
        pred (Tensor): predictions (logits).
        target (Tensor): ground truth binary mask.
        ignore_index (int): label index to ignore.

    Returns:
        loss (Tensor): averaged loss over non-ignored pixels.
    """
    mask = (target != ignore_index)  # mask for valid pixels
    loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
    return loss[mask].mean()

# ================================
# Linearly Adjust Adversarial Lambda Parameter Over Epochs
# ================================
def adjust_lambda_adv(current_epoch, max_epoch=50, max_lambda=0.1, start_lambda=0.01):
    """
    Linearly increases adversarial loss weighting factor lambda during training.

    Args:
        current_epoch (int): current training epoch.
        max_epoch (int): total number of epochs.
        max_lambda (float): max lambda value.
        start_lambda (float): starting lambda value.

    Returns:
        adjusted lambda value for current epoch.
    """
    progress = current_epoch / max_epoch
    return start_lambda + (max_lambda - start_lambda) * progress


def load_pretrained_backbone(model, pretrained_path, device):
    print(f"📅 Caricamento pesi pretrained da {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)

    # Estrai il dict dei pesi dal checkpoint
    if isinstance(checkpoint, dict):
        print("🗝️ Chiavi nel checkpoint:", list(checkpoint.keys()))
        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
    else:
        pretrained_dict = checkpoint

    model_dict = model.state_dict()
    print(f"🔍 Numero layer modello: {len(model_dict)}")
    print(f"🔍 Numero pesi checkpoint: {len(pretrained_dict)}")

    # Normalizza le chiavi del checkpoint rimuovendo prefissi tipo 'module.' o 'cp.' se presenti
    def clean_key(k):
        if k.startswith('module.'):
            return k[len('module.'):]
        elif k.startswith('cp.'):
            return k[len('cp.'):]
        else:
            return k

    cleaned_pretrained_dict = {clean_key(k): v for k, v in pretrained_dict.items()}

    new_pretrained_dict = {}
    missing_keys = []
    # Identifica le chiavi mancanti nel checkpoint
    for k in model_dict.keys():
        if k in cleaned_pretrained_dict:
            if model_dict[k].shape == cleaned_pretrained_dict[k].shape:
                new_pretrained_dict[k] = cleaned_pretrained_dict[k]
            else:
                print(f"⚠️ Shape mismatch per '{k}': modello {model_dict[k].shape}, checkpoint {cleaned_pretrained_dict[k].shape}")
                missing_keys.append(k)
        else:
            missing_keys.append(k)

    # Identifica le chiavi nel checkpoint non usate dal modello (unexpected_keys)
    unexpected_keys = [k for k in cleaned_pretrained_dict.keys() if k not in model_dict]

    print(f"✅ Trovati {len(new_pretrained_dict)} pesi corrispondenti.")
    print(f"⚠️ Chiavi modello mancanti nel checkpoint ({len(missing_keys)}): {missing_keys[:10]}")
    print(f"⚠️ Chiavi checkpoint non usate nel modello ({len(unexpected_keys)}): {unexpected_keys[:10]}")

    if not new_pretrained_dict:
        raise ValueError("❌ Nessun peso corrispondente trovato nel file pretrained.")

    model.load_state_dict(new_pretrained_dict, strict=False)
    print(f"✅ Pesi caricati nel modello (partial load).")
