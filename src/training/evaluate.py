
from ultralytics import YOLO

import multiprocessing as mp

def main():
    model = YOLO("runs/detect/models/yolo11s_neu-4/weights/best.pt")

    metrics = model.val(
        data="data/processed/dataset.yaml",
        workers=0
    )

    print(metrics)

if __name__ == "__main__":
    mp.set_start_method("fork", force="True")
    main()