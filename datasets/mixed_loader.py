import os
import torch
import random
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from torchvision import transforms

from datasets.cacd_dataset  import CACDDataset
from datasets.fgnet_dataset import FGNETDataset

def get_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_mixed_dataloaders(cacd_dir, cacd_csv, fgnet_dir,
                          train_total=3000, test_total=1000,
                          fgnet_ratio=0.35, val_ratio=0.2,
                          batch_size=32, seed=42):

    random.seed(seed)

    # Proporciones
    train_cacd_count = int(train_total * 0.8)
    train_fgnet_count = train_total - train_cacd_count

    test_fgnet_count = int(test_total * 0.8)
    test_cacd_count = test_total - test_fgnet_count

    transform = get_transforms()

    # Datasets completos
    cacd_dataset = CACDDataset(csv_path=cacd_csv, root_dir=cacd_dir, transform=transform)
    fgnet_dataset = FGNETDataset(root_dir=fgnet_dir, transform=transform)

    # Índices aleatorios
    cacd_indices = list(range(len(cacd_dataset)))
    fgnet_indices = list(range(len(fgnet_dataset)))
    random.shuffle(cacd_indices)
    random.shuffle(fgnet_indices)

    # Subsets
    train_cacd = Subset(cacd_dataset, cacd_indices[:train_cacd_count])
    test_cacd = Subset(cacd_dataset, cacd_indices[train_cacd_count:train_cacd_count+test_cacd_count])

    train_fgnet = Subset(fgnet_dataset, fgnet_indices[:train_fgnet_count])
    test_fgnet = Subset(fgnet_dataset, fgnet_indices[train_fgnet_count:train_fgnet_count+test_fgnet_count])

    full_train_dataset = ConcatDataset([train_cacd, train_fgnet])

    val_size = int(val_ratio * train_total)
    train_size = train_total - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = ConcatDataset([test_fgnet, test_cacd])
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader