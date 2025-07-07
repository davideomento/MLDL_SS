# Real-Time Semantic Segmentation with Domain Adaptation

## Overview

This project tackles the **domain shift problem** in real-time semantic segmentation by exploring and comparing various domain adaptation strategies.

Initially, two segmentation networks are trained and evaluated on the Cityscapes dataset to establish upper-bound baselines for fully supervised performance:
- **DeepLabV2** — classic, high-accuracy segmentation network.
- **BiSeNet** — lightweight, real-time segmentation network.

## Domain Shift Analysis

We analyze the domain shift by training BiSeNet on the synthetic **GTA5** dataset and evaluating it on the real-world **Cityscapes** dataset, revealing a significant performance drop caused by the domain gap.

To mitigate this issue, we explore two main strategies:
- Basic data augmentation techniques as a lightweight domain adaptation method.
- Adversarial domain adaptation techniques for further performance gains.

## Real-Time Performance with STDC Network

To further improve real-time segmentation, data augmentations are combined with the **Short-Term Dense Concatenate (STDC)** network, which is trained and evaluated on Cityscapes.

Several architectural variations of STDC are explored, with the best configurations improving accuracy while maintaining low latency suitable for real-time applications.

## Results Summary

| Model     | mIoU (%) | Latency (seconds) | Notes                    |
|-----------|----------|-------------------|--------------------------|
| DeepLabV2 | 61.85    | Higher latency    | Classic segmentation     |
| BiSeNet   | 52.75    | ~0.03             | Real-time capable        |
| STDC1     | 56.39    | ~0.0323           | Faster real-time model   |
| STDC2     | 57.81    | ~0.0371           | Improved accuracy        |

## Usage

Before running training or evaluation scripts, ensure the required datasets and pretrained weights are downloaded and paths are correctly set.

1. Download pretrained weights and datasets from the links below:

- **DeepLabV2 pretrained weights:**  
  https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing

- **Cityscapes dataset:**  
  https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing

- **GTA5 dataset:**  
  https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing

- **STDC pretrained weights:**  
  https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1?usp=sharing

2. Run `download_dataset.py` inside the `datasets/` folder to download the datasets.


3. Run the scripts as needed:

```bash
├── 2a_deeplabv2.py                # DeepLabV2 trained and tested on Cityscapes
├── 2b_bisenet.py                  # BiSeNet trained and tested on Cityscapes
├── 3a_domain_shift.py             # BiSeNet trained on GTA5 and tested on Cityscapes
├── 3b_domain_shift_data_aug.py    # BiSeNet trained on augmented GTA5 and tested on Cityscapes
├── 4_domain_shift_adversarial.py  # BiSeNet trained on GTA5 with adversarial adaptation and tested on Cityscapes
├── 5_extension_stdc.py            # STDC1 and STDC2 trained and tested on Cityscapes



# Folder organization

MLDL_SemanticSegmentation/
│
├── datasets/                
│   ├── datasets_cityscapes/  
│       ├── __init__.py         
│       ├── cityscapes_aug.py
│       └── cityscapes.py
│   ├── datasets_gta5/
│       ├── __init__.py    
│       ├── gta5_aug.py
│       ├── gta5_labels.py
│       └── gta5.py
│   ├── __init__.py
│   └── download_dataset.py
│
├── models/
│   ├── bisenet/  
│       ├── __init__.py
│       ├── build_bisenet.py
│       └── build_contextpath.py
│   ├── deeplabv2/  
│       ├── __init__.py
│       └── deeplabv2.py
│   ├── STDC/  
│       ├── __init__.py
│       ├── stdc_model.py
│       └── stdcnet.py
│   └── __init__.py
│
├── utils/                   
│   ├── __init__.py
│   ├── discrimantor.py
│   ├── metrics.py
│   └── utils.py
│
├── 2a_deeplabv2.py
├── 2b_bisenet.py
├── 3a_domain_shift.py
├── 3b_domain_shift_data_aug.py
├── 4_domain_shift_adversarial.py
├── 5_extension_stdc.py
│
├── .gitignore                # File e cartelle da ignorare da Git
└── README.md                 # Documentazione del progetto
