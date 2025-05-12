import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def compute_age_group_accuracy(preds, targets, tolerance=5):
    preds = np.array(preds)
    targets = np.array(targets)
    correct = np.sum(np.abs(preds - targets) <= tolerance)
    return correct / len(targets)

def plot_residual_error(preds, targets, output_path="residual_fgnet.png"):
    residuals = np.array(preds) - np.array(targets)

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color='steelblue', edgecolor='black')
    plt.title("Residual Error Distribution (Predicted - Real Age)")
    plt.xlabel("Error (years)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Residual plot guardado en {output_path}")

def plot_absolute_error(preds, targets, output_path="abs_error_fgnet.png"):
    errors = np.abs(np.array(preds) - np.array(targets))
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, color='steelblue', edgecolor='black')
    plt.title("Absolute Error Distribution")
    plt.xlabel("Absolute Error (years)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.float32).to(device)
            outputs = model(inputs).squeeze(1)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    acc = compute_age_group_accuracy(preds, targets, tolerance=5)

    print("\n[RESULTADOS TEST FINAL]")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R²:  {r2:.4f}")
    print(f"Accuracy @±5 años: {acc * 100:.2f}%")

    plot_residual_error(preds, targets)
    plot_absolute_error(preds, targets)
    return mae, mse, r2, acc