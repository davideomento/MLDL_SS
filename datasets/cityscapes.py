def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    mask_path = self.label_paths[idx]

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)

    if self.transform:
        image = self.transform(image)  # Solo immagine

    if self.target_transform:
        mask = self.target_transform(mask)  # Solo maschera

    return image, mask
