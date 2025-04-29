import os
import gdown
from zipfile import ZipFile

# ================================
# Cityscapes
# ================================

cityscapes_zip = 'cityscapes.zip'
cityscapes_folder = './Cityscapes'

if not os.path.exists(cityscapes_folder):
    if not os.path.exists(cityscapes_zip):
        print("📦 Scaricando Cityscapes...")
        gdown.download('https://drive.google.com/uc?id=1WheH0FcXKYMIiqOQuF_QZf3JADM1zGEj', cityscapes_zip, quiet=False)
    print("📂 Estraendo Cityscapes...")
    with ZipFile(cityscapes_zip, 'r') as zip_ref:
        zip_ref.extractall(cityscapes_folder)
    print('✅ Cityscapes pronto!')
else:
    print('✔ Cityscapes già presente, nessun download necessario.')

# ================================
# DeepLabv2 Weights
# ================================

weights_path = 'data/deeplabv2_weights.pth'
if not os.path.exists(weights_path):
    print("💾 Scaricando pesi pre-addestrati DeepLabv2...")
    gdown.download(
        'https://drive.google.com/uc?id=1fcwW74wfXLoBZhJTMHlWZ_Y2q1fFDWT9',
        weights_path,
        quiet=False
    )
    print('✅ Pesi DeepLab scaricati.')
else:
    print('✔ Pesi DeepLabv2 già presenti.')

# ================================
# GTA5 Dataset (se hai il link)
# ================================

gta5_zip = 'gta5.zip'
gta5_folder = './GTA5'

if not os.path.exists(gta5_folder):
    if not os.path.exists(gta5_zip):
        print("📦 Scaricando GTA5...")
        gdown.download('https://drive.google.com/uc?id=1NGSaDgt0JiUr8NrMsAZ-Iuw4zc-w2p9t&export=download', gta5_zip, quiet=False)
    print("📂 Estraendo GTA5...")
    with ZipFile(gta5_zip, 'r') as zip_ref:
        zip_ref.extractall(gta5_folder)
    print('✅ GTA5 pronto!')
else:
    print('✔ GTA5 già presente, nessun download necessario.')