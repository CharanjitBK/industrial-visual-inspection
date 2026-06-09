# Industrial Visual Inspection Platform

## Overview

An AI-powered industrial defect detection system for steel surface inspection using YOLOv11 and PyTorch.

The project focuses on detecting and classifying surface defects from steel images, evaluating model performance through multiple experiments, and tracking results using MLflow.

---

## Dataset

### NEU Surface Defect Database

The dataset contains six defect categories:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

### Dataset Preparation

- Train / Validation / Test split
- YOLO annotation format conversion
- Dataset tiling for improved small-defect detection
- Label validation using OpenCV

---

## Tech Stack

### Machine Learning
- Python
- PyTorch
- YOLOv11 (Ultralytics)
- OpenCV
- NumPy
- Pandas

### Experiment Tracking
- MLflow

### Version Control
- Git
- GitHub

---

## Training Pipeline

1. Dataset preparation
2. YOLO dataset conversion
3. Model training (YOLOv11)
4. Model evaluation
5. MLflow experiment tracking
6. Model selection

---

## Experiments Conducted

### Models Evaluated
- YOLO11n
- YOLO11s
- YOLO11m

### Dataset Variants
- Original dataset
- Tiled dataset

### Image Sizes
- 320 × 320
- 640 × 640

### Training Configuration
- Epochs: 100
- Batch size: 16
- Learning rate: 0.01

---

## Experiment Results

| Model   | Dataset | Image Size | mAP50  | mAP50-95 |
|----------|---------|------------|--------|----------|
| YOLO11s  | Tiled   | 640        | 0.7814 | 0.4546   |
| YOLO11s  | Normal  | 640        | 0.7790 | 0.4513   |
| YOLO11n  | Tiled   | 320        | 0.7580 | 0.4382   |
| YOLO11s  | Tiled   | 320        | 0.7572 | 0.4371   |
| YOLO11m  | Tiled   | 320        | 0.7509 | 0.4361   |

---

## Best Model

- **Model:** YOLO11s
- **Dataset:** Tiled NEU Dataset
- **Image Size:** 640 × 640
- **Epochs:** 100

### Performance Metrics

| Metric     | Value  |
|------------|--------|
| Precision   | 0.7453 |
| Recall      | 0.7096 |
| mAP50       | 0.7814 |
| mAP50-95    | 0.4546 |

---

## MLflow Tracking

MLflow was used for:

- Tracking training runs
- Comparing multiple experiments
- Logging hyperparameters
- Storing evaluation metrics
- Saving model artifacts

---

## Key Findings

- YOLO11s outperformed YOLO11n and YOLO11m on this dataset
- 640×640 input resolution performed better than 320×320
- Dataset tiling improved detection of small defects
- Best configuration achieved strong balance between precision and recall