import os
from zipfile import ZipFile
from tqdm import tqdm
import shutil

# ================================
# Rileva ambiente
# ================================
is_colab = 'COLAB_GPU' in os.environ
is_kaggle = os.path.exists('/kaggle')

if is_colab:
    print("📍 Ambiente: Colab")
    base_drive_path = '/content/drive/MyDrive'  # ← Personalizza se serve
    working_dir = './'
elif is_kaggle:
    print("📍 Ambiente: Kaggle")
    base_drive_path = '/kaggle/input'  # I dataset sono già in '/kaggle/input'
    working_dir = '/kaggle/working'
else:
    print("📍 Ambiente: Locale")
    base_drive_path = './'
    working_dir = './'

# ================================
# Cityscapes
# ================================

cityscapes_zip = os.path.join(base_drive_path, 'Cityscapes.zip')
cityscapes_folder = os.path.join(working_dir, 'Cityscapes')

if not os.path.exists(cityscapes_folder):
    if os.path.exists(cityscapes_zip):
        print("📂 Estraendo Cityscapes...")
        with ZipFile(cityscapes_zip, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc="Estrazione Cityscapes", unit="file"):
                zip_ref.extract(member, working_dir)
        print('✅ Cityscapes pronto!')
    else:
        print(f'❌ File Cityscapes ZIP non trovato: {cityscapes_zip}')
else:
    print('✔ Cityscapes già presente.')

# ================================
# DeepLabv2 Weights
# ================================

weights_path = os.path.join(base_drive_path, 'deeplab_resnet_pretrained_imagenet.pth')
local_weights = os.path.join(working_dir, 'deeplabv2_weights.pth')

if not os.path.exists(local_weights):
    if os.path.exists(weights_path):
        print("💾 Copiando pesi pre-addestrati DeepLabv2...")
        shutil.copy(weights_path, local_weights)
        print('✅ Pesi DeepLab copiati localmente.')
    else:
        print(f'❌ File dei pesi DeepLabv2 non trovato: {weights_path}')
else:
    print('✔ Pesi DeepLabv2 già presenti.')

# ================================
# GTA5
# ================================

gta5_zip = os.path.join(base_drive_path, 'GTA5.zip')
gta5_folder = os.path.join(working_dir, 'GTA5')

if not os.path.exists(gta5_folder):
    if os.path.exists(gta5_zip):
        print("📂 Estraendo GTA5...")
        with ZipFile(gta5_zip, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc="Estrazione GTA5", unit="file"):
                zip_ref.extract(member, working_dir)
        print('✅ GTA5 pronto!')
    else:
        print(f'❌ File GTA5 ZIP non trovato: {gta5_zip}')
else:
    print('✔ GTA5 già presente, nessuna estrazione necessaria.')
