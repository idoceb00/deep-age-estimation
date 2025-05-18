import os
from datasets.cacd_dataset import CACDDataset
from datasets.fgnet_dataset import FGNETDataset
from datasets.agedb_dataset import AgeDBDataset
from datasets.morph_dataset import MORPHDataset
import random
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset
import torch
from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_cacd_dataloaders(
        cacd_dir,
        cacd_csv,
        train_total=18000,
        test_total=2000,
        val_ratio=0.2,
        batch_size=32,
        seed=42,
        num_workers=4
):
    random.seed(seed)

    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    df = pd.read_csv(cacd_csv)
    indices = list(df.index)
    random.shuffle(indices)
    selected_indices = indices[:train_total + test_total]

    train_indices = selected_indices[:train_total]
    test_indices = selected_indices[train_total:]

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    # Split train en train y val
    val_size = int(val_ratio * len(train_df))
    val_df = train_df[:val_size].reset_index(drop=True)
    train_df = train_df[val_size:].reset_index(drop=True)

    # Crear datasets
    train_dataset = CACDDataset(df=train_df, root_dir=cacd_dir, transform=train_transforms)
    val_dataset = CACDDataset(df=val_df, root_dir=cacd_dir, transform=test_transforms)
    test_dataset = CACDDataset(df=test_df, root_dir=cacd_dir, transform=test_transforms)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_fgnet_dataloaders(
        fgnet_dir,
        train_total=900,
        test_total=100,
        val_ratio=0.2,
        batch_size=32,
        seed=42,
        num_workers=4
):
    random.seed(seed)

    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    full_dataset = FGNETDataset(root_dir=fgnet_dir, transform=train_transforms)

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    selected_indices = indices[:train_total + test_total]

    train_indices = selected_indices[:train_total]
    test_indices = selected_indices[train_total:]

    # Subset del dataset original
    train_dataset_full = Subset(full_dataset, train_indices)
    test_dataset = Subset(FGNETDataset(root_dir=fgnet_dir, transform=test_transforms), test_indices)

    # Split train en train + val
    val_size = int(val_ratio * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size

    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_morph_dataloaders(
        morph_dir="data/MORPH",
        batch_size=32,
        num_workers=4
):
    train_dir = os.path.join(morph_dir, "Train")
    val_dir   = os.path.join(morph_dir, "Validation")
    test_dir  = os.path.join(morph_dir, "Test")

    train_transform = get_train_transforms()
    test_transform  = get_test_transforms()

    train_dataset = MORPHDataset(root_dir=train_dir, transform=train_transform)
    val_dataset   = MORPHDataset(root_dir=val_dir, transform=test_transform)
    test_dataset  = MORPHDataset(root_dir=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_agedb_dataloaders(
        agedb_dir="data/AgeDB",
        train_total=1000,
        test_total=2000,
        val_ratio=0.2,
        batch_size=32,
        seed=42,
        num_workers=4
):
    random.seed(seed)

    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    full_dataset = AgeDBDataset(root_dir=agedb_dir, transform=train_transforms)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    selected_indices = indices[:train_total + test_total]

    train_indices = selected_indices[:train_total]
    test_indices = selected_indices[train_total:]

    train_dataset_full = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(
        AgeDBDataset(root_dir=agedb_dir, transform=test_transforms),
        test_indices
    )

    val_size = int(val_ratio * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

from datasets.imdb_dataset import IMDBDataset

def get_imdb_dataloaders(
        root_dir="data/IMDB-WIKI/imdb-clean-1024/imdb-clean-1024",
        csv_train="data/IMDB-WIKI/imdb_train_new_1024.csv",
        csv_val="data/IMDB-WIKI/imdb_valid_new_1024.csv",
        csv_test="data/IMDB-WIKI/imdb_test_new_1024.csv",
        batch_size=32,
        num_workers=4
):
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    train_dataset = IMDBDataset(csv_train, root_dir, transform=train_transform)
    val_dataset   = IMDBDataset(csv_val, root_dir, transform=test_transform)
    test_dataset  = IMDBDataset(csv_test, root_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

from datasets.utkface_dataset import UTKFaceDataset

def get_utkface_dataloaders(
        root_dir="data/UTKFace/UTKFace/UTKFace",
        csv_path="data/UTKFace/utkface.csv",
        train_total=18000,
        test_total=2000,
        val_ratio=0.2,
        batch_size=32,
        seed=42,
        num_workers=4
):
    import random
    random.seed(seed)

    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    df = pd.read_csv(csv_path)
    indices = list(df.index)
    random.shuffle(indices)

    selected_indices = indices[:train_total + test_total]
    train_indices = selected_indices[:train_total]
    test_indices = selected_indices[train_total:]

    # Dataset completo para acceso por índice
    full_train_dataset = UTKFaceDataset(csv_path, root_dir, transform=train_transforms)
    full_test_dataset  = UTKFaceDataset(csv_path, root_dir, transform=test_transforms)

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    test_dataset  = torch.utils.data.Subset(full_test_dataset, test_indices)

    # División train/val
    val_size = int(val_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    # Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

from datasets.megaage_dataset import MegaAgeDataset
from torch.utils.data import DataLoader
import random

def get_megaage_dataloaders(
        name_train="data/MegaAge/megaage_asian/megaage_asian/list/train_name.txt",
        age_train="data/MegaAge/megaage_asian/megaage_asian/list/train_age.txt",
        name_test="data/MegaAge/megaage_asian/megaage_asian/list/test_name.txt",
        age_test="data/MegaAge/megaage_asian/megaage_asian/list/test_age.txt",
        root_train="data/MegaAge/megaage_asian/megaage_asian/train",
        root_test="data/MegaAge/megaage_asian/megaage_asian/test",
        val_ratio=0.2,
        batch_size=32,
        seed=42,
        num_workers=4
):
    from torch.utils.data import random_split

    random.seed(seed)

    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    full_train_dataset = MegaAgeDataset(name_train, age_train, root_train, transform=train_transform)
    test_dataset       = MegaAgeDataset(name_test, age_test, root_test, transform=test_transform)

    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader