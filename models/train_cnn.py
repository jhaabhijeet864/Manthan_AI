import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, dict]:
    """Builds train and validation dataloaders."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, train_dataset.class_to_idx


def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = False, freeze_backbone: bool = False) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for name, p in model.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False
    elif arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for name, p in model.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False
    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained.
        model = models.mobilenet_v3_small(weights=weights)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for name, p in model.named_parameters():
                if not name.startswith("classifier."):
                    p.requires_grad = False
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_sum += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct * 100.0 / max(total, 1)
    return avg_loss, acc


def train(
    data_dir: str,
    output_dir: str,
    arch: str,
    img_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    pretrained: bool,
    freeze_backbone: bool,
    amp: bool,
    seed: int,
    label_smoothing: float,
):
    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, class_to_idx = build_dataloaders(data_dir, img_size, batch_size, num_workers)
    num_classes = len(class_to_idx)

    model = build_model(num_classes, arch=arch, pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.15
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    best_acc = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            seen += targets.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct * 100.0 / max(seen, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        state = {
            "epoch": epoch,
            "arch": arch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
            "class_to_idx": class_to_idx,
            "img_size": img_size,
        }
        torch.save(state, out_dir / "last.pt")
        if is_best:
            torch.save(state, out_dir / "best.pt")

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} acc={val_acc:.2f}% | "
            f"best_acc={best_acc:.2f}%"
        )

    dur = time.time() - start_time
    print(f"Finished training in {dur/60:.1f} min. Best val acc: {best_acc:.2f}%")
    print(f"Artifacts saved to: {out_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN classifier on an ImageFolder dataset.")
    p.add_argument("--data_dir", type=str, default="data", help="Path with train/ and val/ subfolders")
    p.add_gument("--output_dir", type=str, default="runs/train_cnn", help="Where to save checkpoints")
    p.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet_v3_small"], help="Backbone architecture")
    p.add_argument("--img_size", type=int, default=224, help="Input image size")
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Max learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights (may download)")
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze all but classifier head")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for CE loss")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        amp=args.amp,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
    )
