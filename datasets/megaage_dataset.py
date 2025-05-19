# datasets/megaage_dataset.py

import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MegaAgeDataset(Dataset):
    def __init__(self, name_file, age_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(name_file, 'r') as f_names, open(age_file, 'r') as f_ages:
            name_lines = f_names.readlines()
            age_lines = f_ages.readlines()

            for name, age in zip(name_lines, age_lines):
                fname = name.strip()
                try:
                    age = int(age.strip())
                except ValueError:
                    continue

                img_path = os.path.join(self.root_dir, fname)
                if os.path.exists(img_path):
                    self.samples.append((fname, age))

        print(f"[INFO] {len(self.samples)} imágenes cargadas desde {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, age = self.samples[idx]
        img_path = os.path.join(self.root_dir, fname)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)