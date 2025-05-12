import torch
from utils.device import get_device
from datasets.mixed_loader import get_mixed_dataloaders
from models.deepagenet import DeepAgeNet
from scripts.train import train_model
from scripts.evaluate import evaluate

if __name__ == "__main__":
    # Configuración de paths
    cacd_dir = "data/cacd/cacd_split/cacd_split"
    cacd_csv = "data/cacd/CACD_filtered.csv"
    fgnet_dir = "data/fgnet/images"
    model_path = "checkpoints/best_model.pth"

    # Cargar DataLoaders mixtos con split train/val/test
    train_loader, val_loader, test_loader = get_mixed_dataloaders(
        cacd_dir=cacd_dir,
        cacd_csv=cacd_csv,
        fgnet_dir=fgnet_dir,
        train_total=2000,
        test_total=1000,
        val_ratio=0.2,
        batch_size=32,
        seed=42
    )

    # Inicializar modelo y dispositivo
    device = get_device()
    model = DeepAgeNet(pretrained=True).to(device)

    # Entrenar modelo con validación
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=25,
        save_path=model_path
    )

    # Evaluar el mejor modelo en el conjunto de test
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    evaluate(model, test_loader, device)