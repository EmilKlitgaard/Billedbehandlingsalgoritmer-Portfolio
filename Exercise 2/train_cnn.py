#!/usr/bin/env python3
"""
Train a small CNN for digit recognition and export to ONNX.
Requirements: torch, torchvision, pillow, numpy
Run:
  python3 -m venv venv
  source venv/bin/activate
  pip install torch torchvision pillow numpy onnxscript
  python train_cnn.py

This script looks for training digit images in `numbers/1.png` ... `numbers/9.png`.
It augments them by random affine transforms to create a larger dataset.
The trained model is saved as `digit_cnn.onnx`.
"""
import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DIGIT_SIDE = 32
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
MODEL_PATH = "digit_cnn.onnx"
CHECKPOINT_PATH = "digit_cnn_checkpoint.pt"
FOREGROUND_THRESHOLD = 10


class AddGaussianNoise(nn.Module):
    def __init__(self, std=0.15):
        super().__init__()
        self.std = std

    def forward(self, tensor):
        noisy = tensor + torch.randn_like(tensor) * self.std
        return torch.clamp(noisy, 0.0, 1.0)


def extract_digit_crops(img, min_height=12, min_width=8, min_pixels=35):
    arr = np.asarray(img, dtype=np.uint8)
    mask = arr > FOREGROUND_THRESHOLD
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return [img]

    # Split connected row runs into separate candidate digits.
    splits = np.where(np.diff(rows) > 1)[0] + 1
    row_groups = np.split(rows, splits)

    crops = []
    for group in row_groups:
        r0, r1 = int(group[0]), int(group[-1])
        submask = mask[r0:r1 + 1, :]
        cols = np.where(submask.any(axis=0))[0]
        if len(cols) == 0:
            continue
        c0, c1 = int(cols[0]), int(cols[-1])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < min_height or w < min_width:
            continue
        if int(submask[:, c0:c1 + 1].sum()) < min_pixels:
            continue
        crops.append(img.crop((c0, r0, c1 + 1, r1 + 1)))

    return crops if crops else [img]

class DigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for d in range(1, 10):
            path = os.path.join(root_dir, f"{d}.png")
            if os.path.exists(path):
                base_img = Image.open(path).convert('L')
                crops = extract_digit_crops(base_img)
                for crop in crops:
                    self.samples.append((crop, d - 1))
        if not self.samples:
            raise RuntimeError(f"No training images found in {root_dir}")

    def __len__(self):
        return 10000  # synthetic size (we will sample with replacement)

    def __getitem__(self, idx):
        img, label = random.choice(self.samples)
        if self.transform:
            img = self.transform(img)
        return img, label

class SmallCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            # Global pooling reduces sensitivity to exact feature location.
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)


def main():
    transform = T.Compose([
        T.Resize((int(DIGIT_SIDE * 0.9), int(DIGIT_SIDE * 0.9))),
        T.Pad(int(DIGIT_SIDE * 0.3), fill=0),
        T.RandomCrop((DIGIT_SIDE, DIGIT_SIDE)),
        T.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.85, 1.15), shear=5),
        T.ToTensor(),
        # Add noise to a subset of samples to improve robustness.
        T.RandomApply([AddGaussianNoise(std=0.12)], p=0.35),
        T.Normalize((0.5,), (0.5,))
    ])

    ds = DigitDataset('numbers', transform=transform)
    num_workers = min(8, os.cpu_count() or 2)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    print(f"Loaded {len(ds.samples)} digit crops across 9 classes")
    sample_img, sample_label = ds[0]
    print(
        f"Sample tensor stats: shape={tuple(sample_img.shape)} "
        f"mean={sample_img.mean().item():.4f} std={sample_img.std().item():.4f} label={sample_label}"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resumed from checkpoint {CHECKPOINT_PATH} at epoch {start_epoch}")
        except Exception as exc:
            print(f"Could not resume checkpoint ({exc}). Starting a fresh training run.")
            start_epoch = 0

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        running = 0.0
        for i, (imgs, labels) in enumerate(dl):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % 50 == 49:
                print(f"Epoch {epoch+1}/{EPOCHS}, batch {i}, loss {running/50:.4f}")
                running = 0.0
            # limit batches per epoch to speed up
            if i > 200:
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH} after epoch {epoch+1}")

    print('Training complete')

    # Export to ONNX using a sample input
    model.eval()
    dummy = torch.randn(1, 1, DIGIT_SIDE, DIGIT_SIDE, device=device)
    onnx_path = MODEL_PATH
    try:
        torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)
        print(f'Exported ONNX model to {onnx_path}')
    except ModuleNotFoundError as exc:
        print(f"ONNX export failed because a dependency is missing: {exc}")
        print("Install it with: pip install onnxscript")
        print(f"Your trained weights are محفوظ in {CHECKPOINT_PATH}, so you do not need to retrain.")
    except Exception as exc:
        print(f"ONNX export failed: {exc}")
        print(f"Your trained weights are saved in {CHECKPOINT_PATH}, so you do not need to retrain.")

if __name__ == '__main__':
    main()
