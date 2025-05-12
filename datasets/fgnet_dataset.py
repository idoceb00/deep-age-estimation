import os
from PIL import Image
from torch.utils.data import Dataset
import re

class FGNETDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.ages = []

        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                match = re.match(r'^\d{3}[A-Z](\d{2})', filename)
                if match:
                    try:
                        age = int(match.group(1))
                        self.image_paths.append(os.path.join(root_dir, filename))
                        self.ages.append(age)
                    except ValueError:
                        print(f"[WARN] Edad inválida en: {filename}")
                else:
                    print(f"[SKIP] No se reconoce el patrón de edad en: {filename}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        age = self.ages[idx]
        if self.transform:
            image = self.transform(image)
        return image, age