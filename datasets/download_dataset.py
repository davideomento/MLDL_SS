import os
from zipfile import ZipFile
from tqdm import tqdm
import shutil

# ================================
# Detect environment (Google Colab or Local)
# ================================
is_colab = 'COLAB_GPU' in os.environ  # Check if running in Colab by environment variable

if is_colab:
    print("📍 Environment: Colab")
    # Set base path in Google Drive and working directory
    base_drive_path = '/content/drive/MyDrive/Project_MLDL'  # <-- Customize this path if needed
    working_dir = './'  # Current directory in Colab environment
    weights_path = os.path.join(base_drive_path, 'deeplab_resnet_pretrained_imagenet.pth')
else:
    print("📍 Environment: Local")
    # Local setup: use current directory as base and working directory
    base_drive_path = './'
    working_dir = './'

# ================================
# Cityscapes Dataset Extraction
# ================================

cityscapes_zip = os.path.join(base_drive_path, 'Cityscapes.zip')  # Path to Cityscapes zip archive
cityscapes_folder = os.path.join(working_dir, 'Cityscapes')      # Path where Cityscapes should be extracted

if not os.path.exists(cityscapes_folder):
    # If Cityscapes folder does not exist, try to extract from zip
    if os.path.exists(cityscapes_zip):
        print("📂 Extracting Cityscapes dataset...")
        with ZipFile(cityscapes_zip, 'r') as zip_ref:
            members = zip_ref.namelist()  # List of all files in the zip
            # Extract all files with progress bar using tqdm
            for member in tqdm(members, desc="Extracting Cityscapes", unit="file"):
                zip_ref.extract(member, working_dir)
        print('✅ Cityscapes dataset is ready!')
    else:
        print(f'❌ Cityscapes ZIP file not found at: {cityscapes_zip}')
else:
    print('✔ Cityscapes dataset already present, skipping extraction.')

# ================================
# DeepLabv2 Pretrained Weights Setup
# ================================

weights_path = os.path.join(base_drive_path, 'deeplab_resnet_pretrained_imagenet.pth')  # Source weights path
local_weights = os.path.join(working_dir, 'deeplabv2_weights.pth')                     # Destination path for local copy

if not os.path.exists(local_weights):
    # If local copy does not exist, try to copy from source location
    if os.path.exists(weights_path):
        print("💾 Copying pretrained DeepLabv2 weights locally...")
        shutil.copy(weights_path, local_weights)
        print('✅ DeepLabv2 weights copied successfully.')
    else:
        print(f'❌ DeepLabv2 pretrained weights file not found at: {weights_path}')
else:
    print('✔ DeepLabv2 weights already present locally.')

# ================================
# GTA5 Dataset Extraction
# ================================

gta5_zip = os.path.join(base_drive_path, 'GTA5.zip')  # Path to GTA5 zip archive
gta5_folder = os.path.join(working_dir, 'GTA5')      # Path where GTA5 dataset should be extracted

if not os.path.exists(gta5_folder):
    # If GTA5 folder does not exist, try to extract from zip
    if os.path.exists(gta5_zip):
        print("📂 Extracting GTA5 dataset...")
        with ZipFile(gta5_zip, 'r') as zip_ref:
            members = zip_ref.namelist()
            # Extract all files with progress bar
            for member in tqdm(members, desc="Extracting GTA5", unit="file"):
                zip_ref.extract(member, working_dir)
        print('✅ GTA5 dataset is ready!')
    else:
        print(f'❌ GTA5 ZIP file not found at: {gta5_zip}')
else:
    print('✔ GTA5 dataset already present, skipping extraction.')
