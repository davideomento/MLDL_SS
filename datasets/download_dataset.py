import os
from zipfile import ZipFile

# ================================
# Cityscapes
# ================================

cityscapes_zip = '/content/drive/MyDrive/Cityscapes.zip'  # <-- Modifica questo path
cityscapes_folder = './Cityscapes'

if not os.path.exists(cityscapes_folder):
    if os.path.exists(cityscapes_zip):
        print("📂 Estraendo Cityscapes...")
        with ZipFile(cityscapes_zip, 'r') as zip_ref:
            zip_ref.extractall('./')
        print('✅ Cityscapes pronto!')
    else:
        print('❌ File Cityscapes ZIP non trovato al path specificato.')
else:
    print('✔ Cityscapes già presente, nessun estrazione necessaria.')

# ================================
# DeepLabv2 Weights
# ================================

weights_path = '/content/drive/MyDrive/deeplab_resnet_pretrained_imagenet.pth'  # <-- Modifica questo path
local_weights = 'deeplabv2_weights.pth'

if not os.path.exists(local_weights):
    if os.path.exists(weights_path):
        print("💾 Copiando pesi pre-addestrati DeepLabv2...")
        os.system(f'cp "{weights_path}" "{local_weights}"')
        print('✅ Pesi DeepLab copiati localmente.')
    else:
        print('❌ File dei pesi DeepLabv2 non trovato al path specificato.')
else:
    print('✔ Pesi DeepLabv2 già presenti.')

# ================================
# GTA5 Dataset
# ================================

gta5_zip = '/content/drive/MyDrive/GTA5.zip'  # <-- Modifica questo path
gta5_folder = './GTA5'

if not os.path.exists(gta5_folder):
    if os.path.exists(gta5_zip):
        print("📂 Estraendo GTA5...")
        with ZipFile(gta5_zip, 'r') as zip_ref:
            zip_ref.extractall('./')
        print('✅ GTA5 pronto!')
    else:
        print('❌ File GTA5 ZIP non trovato al path specificato.')
else:
    print('✔ GTA5 già presente, nessuna estrazione necessaria.')
