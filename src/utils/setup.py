import torch
import cv2
import mlflow
from ultralytics import YOLO

print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("OpenCV:", cv2.__version__)
print("MLflow:", mlflow.__version__)

print("Setup successful")