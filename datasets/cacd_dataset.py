import os
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torch

class CACDDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.data = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        print(f"[INFO] {len(self.data)} imágenes válidas en: {root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['folder'], row['name'])

        image = Image.open(img_path).convert('RGB')
        age = float(row['age'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)