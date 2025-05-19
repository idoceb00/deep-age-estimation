import os
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class UTKFaceDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

        # Delete entries with invalid or empty age
        self.df = self.df[self.df['age'].notna() & (self.df['age'] > 0)].reset_index(drop=True)
        print(f"[INFO] {len(self.df)} imágenes válidas cargadas desde {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['files'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        age = torch.tensor(float(row['age']), dtype=torch.float32)
        return image, age