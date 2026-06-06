import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
RAW_IMG_DIR = Path("data/raw/NEU-DET/IMAGES")
RAW_ANN_DIR = Path("data/raw/NEU-DET/ANNOTATIONS")

OUT_IMG_TRAIN = Path("data/processed/images/train")
OUT_IMG_VAL = Path("data/processed/images/val")

OUT_LBL_TRAIN = Path("data/processed/labels/train")
OUT_LBL_VAL = Path("data/processed/labels/val")

# class mapping
classes = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}

# -----------------------------
# HELPERS
# -----------------------------
def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    xmin, ymin, xmax, ymax = box

    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return x, y, w, h


def convert_annotation(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []

    for obj in root.findall("object"):
        cls = obj.find("name").text

        if cls not in classes:
            continue

        cls_id = classes[cls]

        bnd = obj.find("bndbox")
        box = (
            float(bnd.find("xmin").text),
            float(bnd.find("ymin").text),
            float(bnd.find("xmax").text),
            float(bnd.find("ymax").text)
        )

        bb = convert_bbox((w, h), box)

        lines.append(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# -----------------------------
# BUILD DATASET
# -----------------------------
def main():
    images = list(RAW_IMG_DIR.glob("*.jpg"))

    random.shuffle(images)

    split = int(0.8 * len(images))
    train_imgs = images[:split]
    val_imgs = images[split:]

    def process(image_list, img_out, lbl_out):
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in image_list:
            ann_path = RAW_ANN_DIR / (img_path.stem + ".xml")

            if not ann_path.exists():
                continue

            # copy image
            shutil.copy(img_path, img_out / img_path.name)

            # convert annotation
            convert_annotation(
                ann_path,
                lbl_out / (img_path.stem + ".txt")
            )

    process(train_imgs, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
    process(val_imgs, OUT_IMG_VAL, OUT_LBL_VAL)

    print("Dataset build complete!")


if __name__ == "__main__":
    main()