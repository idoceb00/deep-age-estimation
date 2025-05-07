import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FGNETDataset(Dataset):
    def __init__(self, images_dir="data/FGNET/images", image_size=224, custom_transform=None):
        self.images_dir = images_dir
        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png')) and 'A' in f
        ]

        self.transform = custom_transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        try:
            age_str = img_name.split('A')[1].split('.')[0]
            age = float(age_str)
        except (IndexError, ValueError):
            raise ValueError(f"No se pudo extraer la edad del archivo: {img_name}")

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, age
