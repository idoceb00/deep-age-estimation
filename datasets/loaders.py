import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from datasets.cacd_dataset import CACDDataset
from datasets.fgnet_dataset import FGNETDataset
from datasets.agedb_dataset import AgeDBDataset
from datasets.morph_dataset import MORPHDataset
from datasets.imdb_dataset import IMDBDataset
from datasets.utkface_dataset import UTKFaceDataset
from datasets.megaage_dataset import MegaAgeDataset

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

# Unified pattern for all datasets
def split_dataset(full_dataset, used_portion, seed):
    """
    Randomly selects a subset of the dataset and splits it into train/val/test
    The split follows a fixed 70/20/10 ratio

    """
    total_len = int(len(full_dataset) * used_portion)
    subset_indices = random.sample(range(len(full_dataset)), total_len)
    subset = Subset(full_dataset, subset_indices)

    train_size = int(0.7 * total_len)
    val_size = int(0.2 * total_len)
    test_size = total_len - train_size - val_size

    return random_split(subset, [train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(seed))

def get_dataloaders(dataset_cls, dataset_args, used_portion=1.0, batch_size=32, seed=42, num_workers=4):
    """
    Loads a dataset, samples a portion, splits into train/val/test, and returns
    the corresponding DataLoaders.

    """
    random.seed(seed)
    dataset = dataset_cls(**dataset_args, transform=get_train_transforms())
    train_set, val_set, test_set = split_dataset(dataset, used_portion, seed)
    print(f"[INFO] Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Cambiar transform para val/test
    val_set.dataset.transform = get_test_transforms()
    test_set.dataset.transform = get_test_transforms()

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

def get_cacd_dataloaders(root_dir, csv_path, **kwargs):
    return get_dataloaders(CACDDataset, {"root_dir": root_dir, "df": pd.read_csv(csv_path)}, **kwargs)

def get_fgnet_dataloaders(root_dir, **kwargs):
    return get_dataloaders(FGNETDataset, {"root_dir": root_dir}, **kwargs)

def get_agedb_dataloaders(root_dir, **kwargs):
    return get_dataloaders(AgeDBDataset, {"root_dir": root_dir}, **kwargs)

def get_morph_dataloaders(root_dir, **kwargs):
    combined_path = os.path.join(root_dir, "Train")
    return get_dataloaders(MORPHDataset, {"root_dir": combined_path}, **kwargs)

def get_imdb_dataloaders(csv_path, root_dir, **kwargs):
    return get_dataloaders(IMDBDataset, {"csv_path": csv_path, "root_dir": root_dir}, **kwargs)

def get_utkface_dataloaders(csv_path, root_dir, **kwargs):
    return get_dataloaders(UTKFaceDataset, {"csv_path": csv_path, "root_dir": root_dir}, **kwargs)

def get_megaage_dataloaders(name_file, age_file, root_dir, **kwargs):
    return get_dataloaders(MegaAgeDataset, {"name_file": name_file, "age_file": age_file, "root_dir": root_dir}, **kwargs)
