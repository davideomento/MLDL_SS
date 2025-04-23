import requests
from zipfile import ZipFile
from io import BytesIO

#DA CAPIRE SE METTERE SU DRIVE O BASTA SOLO from torchvision.datasets import Cityscapes
# Define the path to the dataset
dataset_path = 'https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing' 

# Send a GET request to the URL
response = requests.get(dataset_path)
# Check if the request was successful
if response.status_code == 200:
    # Open the downloaded bytes and extract them
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall('./dataset') 
    print('Download and extraction complete!')