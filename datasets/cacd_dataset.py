import os
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES =True
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class CACDDataset(Dataset):
    def __init__(self, csv_path, root_dir, split='train', transform=None, seed=42):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

        print(f"[INFO] Imágenes válidas encontradas: {len(self.df)}")

        # División reproducible 80/20
        train_df, val_df = train_test_split(self.df, test_size=0.2, random_state=seed, shuffle=True)

        if split == 'train':
            self.data = train_df.reset_index(drop=True)
        elif split == 'val':
            self.data = val_df.reset_index(drop=True)
        else:
            raise ValueError("Split must be 'train' or 'val'")

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
