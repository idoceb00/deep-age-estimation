import torch
from utils.device import get_device
from datasets.mixed_loader import get_mixed_dataloaders
from models.deepagenet import DeepAgeNet
from scripts.train import train_model
from scripts.evaluate import evaluate
from datasets.loaders import *
if __name__ == "__main__":
    # Configuración de paths
    cacd_dir = "data/cacd/cacd_split/cacd_split"
    cacd_csv = "data/cacd/CACD_filtered.csv"
    fgnet_dir = "data/FGNET/images"
    morph_dir="data/MORPH/images"
    agedb_dir="data/AgeDB"
    imdb_dir="data/IMDB-WIKI/imdb-clean-1024/imdb-clean-1024"
    imdb_csv="data/IMDB-WIKI/imdb_train_new_1024.csv"
    utkface_dir="data/UTKFace/UTKFace/UTKFace/UTKFace"
    utkface_csv="data/UTKFace/ageutk_full.csv"
    megaage_dir="data/MegaAge/megaage_asian/megaage_asian/test"
    megaage_txt_train = "data/MegaAge/train_age2.txt"
    megaage_txt_test="data/MegaAge/test_age2.txt"
    model_path = "checkpoints/best_model.pth"

    # Cargar DataLoaders mixtos con split train/val/test
    train_loader, val_loader, test_loader = get_morph_dataloaders(
        root_dir=morph_dir,
        used_portion=1.0,
        batch_size=32,

    )

    # Inicializar modelo y dispositivo
    device = get_device()
    model = DeepAgeNet(pretrained=True).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=25,
        unfreeze_interval=5,
        save_path=model_path
    )

    # Evaluar el mejor modelo en el conjunto de test
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    evaluate(model, test_loader, device)