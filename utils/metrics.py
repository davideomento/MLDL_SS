import torch
import time
import pandas as pd
import wandb
from fvcore.nn import FlopCountAnalysis

# ================================
# Utility Function - IoU Calculation
# ================================
def calculate_iou(predicted, target, num_classes, ignore_index=255):
    """
    Compute Intersection over Union (IoU) metrics for semantic segmentation.
    
    Args:
        predicted (Tensor): predicted labels (H x W)
        target (Tensor): ground truth labels (H x W)
        num_classes (int): number of semantic classes
        ignore_index (int): label value to ignore (e.g., 255)
    
    Returns:
        intersection (Tensor): tensor with intersection counts per class
        union (Tensor): tensor with union counts per class
    """
    # Create mask to ignore pixels with ignore_index
    mask = target != ignore_index
    predicted = predicted[mask]
    target = target[mask]

    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)

    # Calculate intersection and union for each class
    for i in range(num_classes):
        inter = ((predicted == i) & (target == i)).sum().item()
        uni = ((predicted == i) | (target == i)).sum().item()
        intersection[i] += inter
        union[i] += uni

    return intersection, union

# ================================
# Benchmarking Function
# ================================
def benchmark_model(model: torch.nn.Module,
                    image_size: tuple = (3, 512, 1024),
                    iterations: int = 200,
                    device: str = 'cuda') -> dict:
    """
    Benchmarks the model performance using synthetic input data.
    
    Measures:
    - Mean inference latency (seconds)
    - Mean FPS (frames per second)
    - Total FLOPs (floating point operations)
    - Number of trainable parameters
    
    Args:
        model (torch.nn.Module): model to benchmark
        image_size (tuple): input tensor shape (channels, height, width)
        iterations (int): number of iterations for timing
        device (str): device to run benchmark on ('cuda' or 'cpu')
    
    Returns:
        dict: dictionary with benchmark statistics
    """
    model = model.to(device).eval()
    dummy_input = torch.randn((1, *image_size), device=device)  # Create random input tensor

    # Warm-up runs to stabilize GPU/CPU performance
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
        if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
            torch.cuda.synchronize()

    # Measure latency and FPS over specified iterations
    records = []
    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
                torch.cuda.synchronize()
            end = time.time()

            latency = end - start  # Time taken for one forward pass
            fps = 1.0 / latency    # Frames processed per second
            records.append({'latency_s': latency, 'fps': fps})

    # Convert timing results to DataFrame and compute mean metrics
    df_bench = pd.DataFrame.from_records(records)
    mean_latency = df_bench['latency_s'].mean()
    mean_fps = df_bench['fps'].mean()

    # Compute FLOPs using fvcore utility
    flop_analyzer = FlopCountAnalysis(model, dummy_input)
    total_flops = flop_analyzer.total()

    # Count total number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Return all collected benchmark metrics as dictionary
    return {
        'mean_latency': mean_latency,
        'mean_fps': mean_fps,
        'num_flops': total_flops,
        'trainable_params': num_params
    }

# ================================
# Function to Save Metrics to Weights & Biases
# ================================
def save_metrics_on_wandb(epoch, metrics_train, metrics_val):
    """
    Logs training and validation metrics to Weights & Biases (wandb).
    
    Args:
        epoch (int): current training epoch
        metrics_train (float): training loss or other training metric
        metrics_val (dict): dictionary with validation metrics including:
            - 'miou': mean IoU over classes
            - 'loss': validation loss
            - 'accuracy': pixel accuracy
            - 'iou_per_class': list or array with IoU for each class
            - 'mean_latency', 'mean_fps', 'num_flops', 'trainable_params' (optional)
    
    Logs metrics per epoch and final metrics at the last epoch.
    """
    # Prepare dictionary of metrics to log for current epoch
    to_serialize = {
        "epoch": epoch,
        "train_loss": metrics_train,
        "val_mIoU": metrics_val['miou'],
        "val_loss": metrics_val['loss'],
        "val_accuracy": metrics_val['accuracy']
    }

    # Add per-class IoU to the log dictionary
    for index, iou in enumerate(metrics_val['iou_per_class']):
        to_serialize[f"class_{index}_val"] = iou

    # Log metrics for current epoch to wandb
    wandb.log(to_serialize)

    # If this is the last epoch, log additional final metrics such as latency and flops
    if epoch == 50:
        wandb.log({
            "train_loss_final": metrics_train,
            "val_IoU_final": metrics_val['miou'],
            "val_loss_final": metrics_val['loss'],
            "val_latency": metrics_val.get('mean_latency', None),
            "val_fps": metrics_val.get('mean_fps', None),
            "val_flops": metrics_val.get('num_flops', None),
            "val_params": metrics_val.get('trainable_params', None)
        })
