import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import torch
torch.set_num_threads(4)

from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data="data/processed_tiled/dataset.yaml",
        epochs=150,
        imgsz=320,
        batch=8,
        workers = 4,
        device=0,
        project="models",
        name="yolo11s_neu_tiled",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5
    )

if __name__ == "__main__":
    main()
    