# main.py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.cacd_dataset import CACDDataset

def show_sample_images(dataset, num_images=5):
    for i in range(num_images):
        img, age = dataset[i]
        img = img.permute(1, 2, 0)  # cambiar de [C,H,W] a [H,W,C] para matplotlib
        img = (img * 0.5) + 0.5     # desnormaliza si usaste Normalize(mean=0.5, std=0.5)
        plt.imshow(img.numpy())
        plt.title(f"Edad: {age}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    dataset = CACDDataset(
        csv_path="data/cacd/CACD_features_sex.csv",
        images_dir="data/cacd/cacd_split/cacd_split",
        image_size=224
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    print(f"Dataset cargado con {len(dataset)} imágenes.")

    # Muestra 5 imágenes con sus edades
    show_sample_images(dataset, num_images=5)

    from datasets.fgnet_dataset import FGNETDataset

    fgnet = FGNETDataset("data/FGNET/images")
    print(f"FG-NET cargado con {len(fgnet)} imágenes.")

    show_sample_images(fgnet, num_images=5)