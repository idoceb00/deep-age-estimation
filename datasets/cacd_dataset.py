import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob

class CACDDataset(Dataset):
    def __init__(self, csv_path, images_dir, image_size=224, custom_transform=None):
        self.data = pd.read_csv(csv_path)
        self.images_dir=images_dir
        self.transform = custom_transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["name"]
        age = float(self.data.iloc[idx]["age"])

        pattern = os.path.join(self.images_dir, "**", img_name)
        matches = glob.glob(pattern, recursive=True)

        if not matches:
            raise FileNotFoundError(f"No se encontro la imagen: {img_name}")

        img_path = matches[0]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, age