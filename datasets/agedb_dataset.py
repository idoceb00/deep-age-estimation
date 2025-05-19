import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AgeDBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                age = self._extract_age(fname)
                if age is not None:
                    self.samples.append((fname, age))

        print(f"[INFO] {len(self.samples)} imágenes válidas en: {root_dir}")

    def _extract_age(self, filename):
        """
        Extracts the age from the file name.
        Example: 0_MariaCallas_35_f.jpg → age = 35
        """
        try:
            parts = filename.split("_")
            return int(parts[2])
        except (IndexError, ValueError):
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, age = self.samples[idx]
        img_path = os.path.join(self.root_dir, fname)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)