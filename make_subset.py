import os
import shutil
import random

# ============================================================
# CONFIG
# ============================================================

DATASET = r"D:\Project\yolo11\datasets\VisDrone"
OUTPUT = r"D:\Project\yolo11\datasets\VisDrone_subset"

# Number of training images to use
TRAIN_IMAGES = 1200

# Keep all validation images
# This gives us a better measurement of whether training helped.

SEED = 42

random.seed(SEED)

# ============================================================
# PATHS
# ============================================================

train_img_src = os.path.join(DATASET, "images", "train")
train_lbl_src = os.path.join(DATASET, "labels", "train")

val_img_src = os.path.join(DATASET, "images", "val")
val_lbl_src = os.path.join(DATASET, "labels", "val")

train_img_dst = os.path.join(OUTPUT, "images", "train")
train_lbl_dst = os.path.join(OUTPUT, "labels", "train")

val_img_dst = os.path.join(OUTPUT, "images", "val")
val_lbl_dst = os.path.join(OUTPUT, "labels", "val")

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in [
    train_img_dst,
    train_lbl_dst,
    val_img_dst,
    val_lbl_dst
]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# GET TRAINING IMAGES
# ============================================================

images = [
    f for f in os.listdir(train_img_src)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Total training images available: {len(images)}")

random.shuffle(images)

selected = images[:min(TRAIN_IMAGES, len(images))]

print(f"Selecting {len(selected)} training images...")

# ============================================================
# COPY TRAINING DATA
# ============================================================

copied = 0

for image in selected:

    src_img = os.path.join(train_img_src, image)
    dst_img = os.path.join(train_img_dst, image)

    label = os.path.splitext(image)[0] + ".txt"

    src_lbl = os.path.join(train_lbl_src, label)
    dst_lbl = os.path.join(train_lbl_dst, label)

    if not os.path.exists(src_lbl):
        continue

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)

    copied += 1

print(f"Training images copied: {copied}")

# ============================================================
# COPY VALIDATION DATA
# ============================================================

val_images = [
    f for f in os.listdir(val_img_src)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Copying {len(val_images)} validation images...")

for image in val_images:

    src_img = os.path.join(val_img_src, image)
    dst_img = os.path.join(val_img_dst, image)

    label = os.path.splitext(image)[0] + ".txt"

    src_lbl = os.path.join(val_lbl_src, label)
    dst_lbl = os.path.join(val_lbl_dst, label)

    if not os.path.exists(src_lbl):
        continue

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)

print("Validation data copied.")

# ============================================================
# CREATE YAML
# ============================================================

yaml_path = os.path.join(OUTPUT, "data.yaml")

yaml_content = """path: D:/Project/yolo11/datasets/VisDrone_subset

train: images/train
val: images/val

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print()
print("============================================================")
print("SUBSET CREATION COMPLETE")
print("============================================================")
print(f"Dataset: {OUTPUT}")
print(f"Train images: {copied}")
print(f"Validation images: {len(val_images)}")
print(f"YAML: {yaml_path}")
print("============================================================")