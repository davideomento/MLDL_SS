# MLDL_SS
Semantic segmentation project 2025!

MLML_SemanticSegmentation/
│
├── extension/                # STDC1 and STDC2 implementation withe changes
│   ├── datasets/           # Componenti riutilizzabili
│       ├── cityscapes_aug.py
│       └── cityscapes.py
│   ├── models/
│       ├── stdc_model.py
│       └── stdcnet.py
│   ├── metrics.py
│   ├── train_stdc.py
│   └── utils.py
├── project/
│   ├── datasets/  
│       ├── cityscapes_aug.py
│       ├── cityscapes.py
│       ├── download_dataset.py  
│       ├── gta5_aug.py
│       ├── gta5_labels.py
│       └── gta5.py
│
├── public/                   # File pubblici (HTML, immagini, ecc.)
│   └── index.html
│
├── tests/                    # Test automatizzati
│   └── App.test.js
│
├── .gitignore                # File e cartelle da ignorare da Git
├── package.json              # Configurazione del progetto Node.js
├── README.md                 # Documentazione del progetto
└── LICENSE                   # Licenza del progetto
