import gdown
from zipfile import ZipFile

# Cityscapes dataset download link
url = 'https://drive.google.com/uc?id=1WheH0FcXKYMIiqOQuF_QZf3JADM1zGEj'
output = 'cityscapes.zip'

# Scarica lo ZIP
gdown.download(url, output, quiet=False)

# Estrai il contenuto
with ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('./Cityscapes')

print('Cityscapes download and extraction complete!')

# Pretrained weights download link

gdown.download(
    'https://drive.google.com/uc?id=1fcwW74wfXLoBZhJTMHlWZ_Y2q1fFDWT9',
    'deeplabv2_weights.pth',
    quiet=False
)

# GTA5 dataset download link

url = ''
output = ''

# Scarica lo ZIP
gdown.download(url, output, quiet=False)

# Estrai il contenuto
with ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('./')

print('GTA5 download and extraction complete!')
