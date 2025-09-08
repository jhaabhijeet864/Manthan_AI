import os
import shutil
from pathlib import Path
import random

BASE_DIR = Path("models/data/raw_data/images")
SPLIT_DIRS = ["train", "val"]
CLASSES = ["positive", "negative"]
SPLIT_RATIO = 0.8  # 80% train, 20% val

def split_class_images(class_name):
    src_dir = BASE_DIR / class_name
    images = list(src_dir.glob("*.jpg"))
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split, split_imgs in zip(SPLIT_DIRS, [train_imgs, val_imgs]):
        split_class_dir = BASE_DIR / split / class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            shutil.copy(img, split_class_dir / img.name)

def main():
    for class_name in CLASSES:
        split_class_images(class_name)
    print("✅ Images split into train/val folders!")

if __name__ == "__main__":
    main()