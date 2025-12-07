"""
Dataset loading utilities
"""

import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

class EmotionDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")

        # Extract emotion from RAVDESS filename in a robust, platform-independent way
        name = Path(img_path).name
        # name examples: '03-01-01-01-01-01-01_rgb.png' or '03-01-01-01-01-01-01_aug_rgb.png'
        base = name.split('_')[0]
        parts = base.split('-')
        emo_id = parts[2] if len(parts) > 2 else None
        if emo_id is None or emo_id not in EMOTIONS:
            # fallback: assign unknown as index 0
            label = 0
        else:
            label = list(EMOTIONS.keys()).index(emo_id)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def load_dataset(feature_dir):
    """Return a sorted list of feature file paths in feature_dir.

    This function mirrors the simple dataset discovery used elsewhere in the
    project. It looks for mel PNGs or numpy feature files.
    """
    feature_dir = os.path.abspath(feature_dir)
    if not os.path.isdir(feature_dir):
        return []

    # Accept several feature file naming conventions introduced by preprocessing:
    # - legacy mel images: *_mel.png
    # - new RGB stacked images: *_rgb.png
    # - numpy stacks: *_features.npy
    # - any single-channel numpy (.npy)
    files = sorted([
        os.path.join(feature_dir, f)
        for f in os.listdir(feature_dir)
        if f.endswith(("_mel.png", "_rgb.png", "_mel.npy", "_features.npy", ".npy"))
    ])
    return files

def get_dataloaders(feature_dir, batch_size=32, seed=42, arch=None):
    # Load all mel PNGs
    all_files = load_dataset(feature_dir)

    # Choose transforms depending on architecture (ImageNet backbones expect 224x224 + normalization)
    if arch and arch.lower() in ("mobilenet", "efficientnet"):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        default_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        # Default transform: resize to a reasonable size and convert to tensor
        default_transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])

    # default split proportions: train 70%, val 15%, test 15%
    train, test = train_test_split(all_files, test_size=0.15, random_state=seed)
    train, val = train_test_split(train, test_size=0.15 / 0.85, random_state=seed)

    train_ds = EmotionDataset(train, transform=default_transform)
    val_ds = EmotionDataset(val, transform=default_transform)
    test_ds = EmotionDataset(test, transform=default_transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


def create_splits(processed_dir, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    random.seed(seed)

    processed_dir = Path(processed_dir)
    all_files = [f for f in processed_dir.glob("*.png")]

    # -------------------------------------------------------
    # 1. Extract BASE AUDIO IDs
    #    e.g. 03-01-05-02-02-02-12_mel.png → 03-01-05-02-02-02-12
    # -------------------------------------------------------
    base_ids = sorted(list(set([
        f.stem.split("_")[0]   # before _mel/_mfcc/_chroma
        for f in all_files
    ])))

    print(f"[INFO] Found {len(base_ids)} unique audio samples")

    # -------------------------------------------------------
    # 2. Split at AUDIO level, not image level
    # -------------------------------------------------------
    train_ids, tmp_ids = train_test_split(
        base_ids, test_size=(1 - train_ratio), random_state=seed
    )

    rel_test_ratio = test_ratio / (test_ratio + val_ratio)  # normalize inside remainder

    val_ids, test_ids = train_test_split(
        tmp_ids, test_size=rel_test_ratio, random_state=seed
    )

    print(f"[INFO] Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)} | Test IDs: {len(test_ids)}")

    # -------------------------------------------------------
    # 3. Group feature images by split
    # -------------------------------------------------------
    def collect_files(ids):
        return [
            str(f)
            for f in all_files
            if f.stem.split("_")[0] in ids
        ]

    train_files = collect_files(train_ids)
    val_files   = collect_files(val_ids)
    test_files  = collect_files(test_ids)

    print(f"[INFO] Train images: {len(train_files)}")
    print(f"[INFO] Val images:   {len(val_files)}")
    print(f"[INFO] Test images:  {len(test_files)}")

    # -------------------------------------------------------
    # 4. Write split lists
    # -------------------------------------------------------
    with open(processed_dir / "train.txt", "w") as f:
        f.write("\n".join(train_files))

    with open(processed_dir / "val.txt", "w") as f:
        f.write("\n".join(val_files))

    with open(processed_dir / "test.txt", "w") as f:
        f.write("\n".join(test_files))

    print("[SUCCESS] Created train/val/test split files!")
