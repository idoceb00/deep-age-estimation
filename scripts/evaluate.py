import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from datasets.fgnet_dataset import FGNETDataset
from models.deepagenet import DeepAgeNet
from utils.device import get_device
import numpy as np

def evaluate_model_on_fgnet(weights_path: str, fgnet_dir: str, batch_size: int = 32):
    device = get_device()
    print(f"[INFO] Evaluando modelo en {device} usando FG-NET...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FGNETDataset(root_dir=fgnet_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = DeepAgeNet(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.float32).to(device)
            outputs = model(inputs).squeeze(1)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        print("\n[PRIMERAS 10 PREDICCIONES]")
        for i in range(10):
            print(f"Real: {targets[i]}, Predicho: {preds[i]:.2f}, Error: {abs(targets[i] - preds[i]):.2f}")

    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    age_group_acc = compute_age_group_accuracy(preds, targets, tolerance=5)
    plot_residual_error(preds, targets)

    print("\n[RESULTADOS EVALUACIÓN FG-NET]")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.4f}")
    print(f"Accuracy @+-5 años: {age_group_acc * 100:.2f}%")

    return mae, mse, r2

def plot_residual_error(preds, targets, output_path="residual_fgnet.png"):
    residuals = np.array(preds) - np.array(targets)

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color='steelblue', edgecolor='black')
    plt.title("Residual Error Distribution (Predicted - Real Age")
    plt.xlabel("Error (years)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Residual plot guardado en {output_path}")

def compute_age_group_accuracy(preds, targets, tolerance=5):
    preds = np.array(preds)
    targets = np.array(targets)
    correct = np.sum(np.abs(preds - targets) <= tolerance)
    return correct / len(targets)