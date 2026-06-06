import os
import cv2
import random
import shutil


# =========================================================
# 1. OPTIONAL: SPLIT RAW DATA FIRST (if not already split)
# =========================================================
def split_dataset(img_dir, lbl_dir, val_ratio=0.2):

    val_img_dir = img_dir.replace("train", "val")
    val_lbl_dir = lbl_dir.replace("train", "val")

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    images = [f for f in os.listdir(img_dir)
              if f.endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)
    split = int(len(images) * val_ratio)

    val_images = images[:split]

    for img in val_images:
        label = img.replace(".jpg", ".txt")

        shutil.move(os.path.join(img_dir, img),
                    os.path.join(val_img_dir, img))

        if os.path.exists(os.path.join(lbl_dir, label)):
            shutil.move(os.path.join(lbl_dir, label),
                        os.path.join(val_lbl_dir, label))

    print("✅ Train/Val split done")


# =========================================================
# 2. TILE IMAGE WITH PADDING
# =========================================================
def tile_image(image, tile=320):

    h, w = image.shape[:2]

    pad_h = (tile - h % tile) % tile
    pad_w = (tile - w % tile) % tile

    padded = cv2.copyMakeBorder(
        image,
        0, pad_h,
        0, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    ph, pw = padded.shape[:2]
    tiles = []

    for y in range(0, ph, tile):
        for x in range(0, pw, tile):
            crop = padded[y:y + tile, x:x + tile]
            tiles.append((crop, x, y))

    return tiles


# =========================================================
# 3. CONVERT YOLO BOX → TILE SPACE
# =========================================================
def convert_bbox_to_tile(bbox, tile_x, tile_y, tile_size, img_w, img_h):

    cls, xc, yc, w, h = bbox
    cls = int(cls)

    # YOLO → absolute
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    # shift into tile space
    x1 -= tile_x
    x2 -= tile_x
    y1 -= tile_y
    y2 -= tile_y

    # skip if outside tile
    if x2 <= 0 or x1 >= tile_size:
        return None
    if y2 <= 0 or y1 >= tile_size:
        return None

    # clip
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(tile_size, x2)
    y2 = min(tile_size, y2)

    # back to YOLO format
    xc = (x1 + x2) / 2 / tile_size
    yc = (y1 + y2) / 2 / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size

    return [cls, xc, yc, w, h]


# =========================================================
# 4. PROCESS DATASET → TILED DATASET
# =========================================================
def process_dataset(img_dir, label_dir, out_img, out_lbl, tile=320):

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    images = [f for f in os.listdir(img_dir)
              if f.endswith((".jpg", ".png", ".jpeg"))]

    for img_name in images:

        img_base = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_base + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        # read labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(list(map(float, line.strip().split())))

        tiles = tile_image(image, tile)

        for i, (crop, tx, ty) in enumerate(tiles):

            new_labels = []

            for bbox in labels:
                converted = convert_bbox_to_tile(
                    bbox, tx, ty, tile, w, h
                )
                if converted:
                    new_labels.append(converted)

            # keep some empty tiles (VERY IMPORTANT)
            if len(new_labels) == 0:
                if random.random() < 0.3:
                    pass
                else:
                    continue

            out_name = f"{img_base}_{i}.jpg"

            cv2.imwrite(os.path.join(out_img, out_name), crop)

            with open(os.path.join(out_lbl, out_name.replace(".jpg", ".txt")), "w") as f:
                for l in new_labels:
                    f.write(" ".join(map(str, l)) + "\n")


# =========================================================
# 5. MAIN PIPELINE
# =========================================================
if __name__ == "__main__":

    # OPTIONAL: only if your raw dataset is NOT split
    # split_dataset(
    #     "data/processed/images/train",
    #     "data/processed/labels/train"
    # )

    # -----------------------
    # TILE TRAIN SET
    # -----------------------
    process_dataset(
        img_dir="data/processed/images/train",
        label_dir="data/processed/labels/train",
        out_img="data/processed_tiled/images/train",
        out_lbl="data/processed_tiled/labels/train",
        tile=320
    )

    # -----------------------
    # TILE VAL SET
    # -----------------------
    process_dataset(
        img_dir="data/processed/images/val",
        label_dir="data/processed/labels/val",
        out_img="data/processed_tiled/images/val",
        out_lbl="data/processed_tiled/labels/val",
        tile=320
    )

    print("✅ Tiling complete (train + val)")