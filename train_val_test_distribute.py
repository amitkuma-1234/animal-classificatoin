import os
import random
import shutil
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

SOURCE_DIR = "Animal_Dataset"

OUTPUT_DIR = "Animal_Data"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ==============================

random.seed(RANDOM_SEED)

source = Path(SOURCE_DIR)
output = Path(OUTPUT_DIR)

# Remove old output folder if exists
if output.exists():
    shutil.rmtree(output)

# Create folders
for split in ["train", "val", "test"]:
    (output / split).mkdir(parents=True, exist_ok=True)

# Supported image formats
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

# Process every class
for class_folder in source.iterdir():

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    images = []

    for file in class_folder.iterdir():
        if file.suffix.lower() in image_extensions:
            images.append(file)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, split_images in splits.items():

        class_output = output / split_name / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, class_output / img.name)

    print(f"{class_name:25s} -> Train:{len(train_imgs):4d}  Val:{len(val_imgs):4d}  Test:{len(test_imgs):4d}")

print("\n======================================")
print("Dataset Successfully Split!")
print("======================================")
print(f"Train Folder : {output/'train'}")
print(f"Validation   : {output/'val'}")
print(f"Test Folder  : {output/'test'}")