#!/usr/bin/env python3
"""
Train a small CNN for digit recognition and export to ONNX.
Requirements: torch, torchvision, pillow, numpy
Run:
  python3 -m venv venv
  source venv/bin/activate
  pip install torch torchvision pillow numpy onnxscript
  python train_cnn.py
  python train_cnn.py --use-generated-data --generated-data-dir generated_data

This script looks for training digit images in `numbers/1.png` ... `numbers/9.png`.
It augments them by random affine transforms to create a larger dataset.
The trained model is saved as `digit_cnn.onnx`.
"""
import argparse
import os
import random
import re
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DIGIT_SIDE = 64
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_PATH = "digit_cnn.onnx"
CHECKPOINT_PATH = "digit_cnn_checkpoint.pt"
FOREGROUND_THRESHOLD = 10
ONNX_OPSET_VERSION = 18


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
    def __init__(self, root_dir, transform=None, use_generated_data=False, generated_data_dir="generated_data"):
        self.samples = []
        self.transform = transform
        self.base_sample_count = 0
        self.generated_sample_count = 0

        self._load_base_samples(root_dir)
        if use_generated_data:
            self._load_generated_samples(generated_data_dir)

        if not self.samples:
            raise RuntimeError(
                "No training images found. "
                f"Checked base dir '{root_dir}'"
                + (f" and generated dir '{generated_data_dir}'." if use_generated_data else ".")
            )

    def _load_base_samples(self, root_dir):
        for d in range(1, 10):
            path = os.path.join(root_dir, f"{d}.png")
            if os.path.exists(path):
                with Image.open(path) as base_img:
                    base_img = base_img.convert('L').copy()
                crops = extract_digit_crops(base_img)
                for crop in crops:
                    self.samples.append((crop, d - 1))
                    self.base_sample_count += 1

    def _load_generated_samples(self, generated_data_dir):
        if not os.path.isdir(generated_data_dir):
            print(f"Generated data dir '{generated_data_dir}' not found, skipping it.")
            return

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        for d in range(1, 10):
            digit_dir = os.path.join(generated_data_dir, str(d))
            if not os.path.isdir(digit_dir):
                continue
            for name in sorted(os.listdir(digit_dir)):
                ext = os.path.splitext(name)[1].lower()
                if ext not in exts:
                    continue
                path = os.path.join(digit_dir, name)
                with Image.open(path) as img:
                    self.samples.append((img.convert('L').copy(), d - 1))
                    self.generated_sample_count += 1

        # Also support flat files in generated_data_dir, like "4_*.png" or "4.png".
        for name in sorted(os.listdir(generated_data_dir)):
            path = os.path.join(generated_data_dir, name)
            if os.path.isdir(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            match = re.match(r"^([1-9])(?:\D.*)?\.[^.]+$", name)
            if not match:
                continue
            d = int(match.group(1))
            with Image.open(path) as img:
                self.samples.append((img.convert('L').copy(), d - 1))
                self.generated_sample_count += 1

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
    parser = argparse.ArgumentParser(description="Train Sudoku digit CNN and export ONNX.")
    parser.add_argument(
        "--numbers-dir",
        default="numbers",
        help="Directory containing base images 1.png..9.png (default: numbers)"
    )
    parser.add_argument(
        "--use-generated-data",
        action="store_true",
        help="Include labeled samples from --generated-data-dir"
    )
    parser.add_argument(
        "--generated-data-dir",
        default="generated_data",
        help="Directory with generated labels (default: generated_data)"
    )
    args = parser.parse_args()
    # Manual toggle: set to False if you want to ignore generated_data even when present.
    INCLUDE_GENERATED_DATA_IF_NONEMPTY = True

    generated_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    generated_file_count = 0
    if os.path.isdir(args.generated_data_dir):
        for root, _, files in os.walk(args.generated_data_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in generated_exts:
                    generated_file_count += 1

    use_generated_data = (
        INCLUDE_GENERATED_DATA_IF_NONEMPTY
        and generated_file_count > 0
    ) or args.use_generated_data

    if use_generated_data:
        print(
            f"Generated data enabled from '{args.generated_data_dir}' "
            f"({generated_file_count} files found)."
        )
    else:
        print("Generated data disabled.")

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

    ds = DigitDataset(
        args.numbers_dir,
        transform=transform,
        use_generated_data=use_generated_data,
        generated_data_dir=args.generated_data_dir,
    )
    num_workers = min(8, os.cpu_count() or 2)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    print(
        f"Loaded {len(ds.samples)} training samples "
        f"({ds.base_sample_count} base, {ds.generated_sample_count} generated)"
    )
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
    # Explicitly drop DataLoader workers before export to avoid noisy shutdown traces
    # when the process is interrupted right after finishing.
    del dl

    # Export to ONNX using a sample input
    model.eval()
    dummy = torch.randn(1, 1, DIGIT_SIDE, DIGIT_SIDE, device=device)
    onnx_path = MODEL_PATH
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=ONNX_OPSET_VERSION
        )
        print(f'Exported ONNX model to {onnx_path} (opset {ONNX_OPSET_VERSION})')
    except ModuleNotFoundError as exc:
        print(f"ONNX export failed because a dependency is missing: {exc}")
        print("Install it with: pip install onnxscript")
        print(f"Your trained weights are محفوظ in {CHECKPOINT_PATH}, so you do not need to retrain.")
    except Exception as exc:
        print(f"ONNX export failed: {exc}")
        print(f"Your trained weights are saved in {CHECKPOINT_PATH}, so you do not need to retrain.")

if __name__ == '__main__':
    main()
