import os
import yaml
import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

from datasets.cacd_dataset import CACDDataset
from models.deepagenet import DeepAgeNet

from utils.device import get_device

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["lr"] = float(config["lr"])
    config["weight_decay"] = float(config["weight_decay"])
    return config

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

def unfreeze_backbone(model, stages=1):
    layers_to_unfreeze = ['layer4', 'layer3', 'layer2', 'layer1'][:stages]
    for name, child in model.backbone.named_children():
        if name in layers_to_unfreeze:
            for param in child.parameters():
                param.requires_grad = True

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_los = 0.0
    for images, ages in loader:
        images, ages = images.to(device), ages.to(device)

        outputs = model(images).view(-1)
        loss = criterion(outputs, ages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_los += loss.item() * images.size(0)

    return running_los / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    mae_total = 0.0

    with torch.no_grad():
        for images, ages in loader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, ages)
            running_loss += loss.item() * images.size(0)
            mae_total += torch.abs(outputs - ages).sum().item()

    return running_loss / len(loader.dataset), mae_total / len(loader.dataset)

def train():
    config = load_config()
    print("weight_decay:", config["weight_decay"], type(config["weight_decay"]))
    device = get_device()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"]),
    ])

    train_set = CACDDataset(config["csv_path"], config["root_dir"], split='train', transform=transform)
    val_set = CACDDataset(config["csv_path"], config["root_dir"], split='val', transform=transform)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])

    model = DeepAgeNet(pretrained=True).to(device)
    freeze_backbone(model)

    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    best_val_mae = float("inf")
    log = []

    for epoch in range(1, config["epochs"] + 1):
        if epoch == config["unfreeze_epoch"]:
            unfreeze_backbone(model, stages=config["unfreeze_stages"])

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        scheduler.step(val_mae)

        print(f"Epoch {epoch}: Train Loss = {train_loss: .4f}, Val Los = {val_loss: .4f}, Val MAE = {val_mae: .2f}")

        log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print("Best model saved")

    pd.DataFrame(log).to_csv("metrics/train_log.csv", index=False)

