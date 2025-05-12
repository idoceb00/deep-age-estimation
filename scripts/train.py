import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_absolute_error

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

def unfreeze_backbone(model, block_name):
    block = getattr(model.backbone, block_name, None)
    if block is not None:
        for param in block.parameters():
            param.requires_grad = True
        print(f"[INFO] Unfrozen block: {block_name}")

def train_model(model, train_loader, val_loader, device, epochs=25, save_path="checkpoints/best_model.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    freeze_backbone(model)
    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_mae = float('inf')
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": []
    }

    unfreeze_blocks = ["layer4", "layer3", "layer2", "layer1", "conv1"]
    blocks_to_unfreeze = []
    unfreeze_interval = 5
    early_stopping_patience = 5
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):

        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.type(torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        train_mae = mean_absolute_error(all_targets, all_preds)
        avg_train_loss = running_loss / len(train_loader)

        # Validación
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.type(torch.float32).to(device)

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_mae = mean_absolute_error(val_targets, val_preds)
        avg_val_loss = val_loss / len(val_loader)

        # Scheduler
        scheduler.step(avg_val_loss)

        # Logging
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(f"[{epoch:02d}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
              f"| Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f}")

        # Guardar mejor modelo
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Guardado nuevo mejor modelo en {save_path} (Val MAE: {val_mae:.2f})")
        else:
            no_improve_epochs += 1
            print(f"[INFO] Sin mejora en Val MAE por {no_improve_epochs} epochs")

        if no_improve_epochs >= early_stopping_patience:
            print(f"[EARLY STOPPING] Deteniendo entrenamiento en epoch {epoch} por falta de mejora.")
            break

        if epoch % unfreeze_interval == 0 and len(blocks_to_unfreeze) < len(unfreeze_blocks):
            block_name = unfreeze_blocks[len(blocks_to_unfreeze)]
            unfreeze_backbone(model, block_name)
            blocks_to_unfreeze.append(block_name)


    # Plot
    plot_training_curves(history)

def plot_training_curves(history, output_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss por Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_mae"], label="Train MAE")
    plt.plot(epochs, history["val_mae"], label="Val MAE")
    plt.title("MAE por Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Curvas de entrenamiento guardadas en {output_path}")