# Industrial Visual Inspection Platform

## Overview

An AI-powered industrial defect detection system for steel surface inspection using YOLOv11 and PyTorch.

The project focuses on detecting and classifying surface defects from steel images, evaluating model performance through multiple experiments, and tracking results using MLflow.

---

## Dataset

### NEU Surface Defect Database

The dataset contains six defect categories:

* Crazing
* Inclusion
* Patches
* Pitted Surface
* Rolled-in Scale
* Scratches

Dataset preparation included:

* Train / Validation split
* YOLO annotation conversion
* Dataset tiling for small defect detection
* Label validation using OpenCV

---

## Tech Stack

### Machine Learning

* Python
* PyTorch
* YOLOv11
* OpenCV
* NumPy
* Pandas

### Experiment Tracking

* MLflow

### Version Control

* Git
* GitHub

---

## Training Pipeline

1. Dataset Preparation
2. YOLO Dataset Conversion
3. Model Training
4. Model Evaluation
5. MLflow Experiment Tracking
6. Model Selection

---

## Experiments Conducted

### Models Evaluated

* YOLO11n
* YOLO11s
* YOLO11m

### Dataset Variants

* Original Dataset
* Tiled Dataset

### Image Sizes

* 320 × 320
* 640 × 640

### Training Configuration

* Epochs: 100
* Batch Size: 16
* Learning Rate: 0.01

---

## Experiment Results

| Model   | Dataset | Image Size | mAP50  | mAP50-95 |
| ------- | ------- | ---------- | ------ | -------- |
| YOLO11s | Tiled   | 640        | 0.7814 | 0.4546   |
| YOLO11s | Normal  | 640        | 0.7790 | 0.4513   |
| YOLO11n | Tiled   | 320        | 0.7580 | 0.4382   |
| YOLO11s | Tiled   | 320        | 0.7572 | 0.4371   |
| YOLO11m | Tiled   | 320        | 0.7509 | 0.4361   |

---

## Best Model

**Model:** YOLO11s

**Dataset:** Tiled NEU Dataset

**Image Size:** 640 × 640

**Epochs:** 100

### Performance

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.7453 |
| Recall    | 0.7096 |
| mAP50     | 0.7814 |
| mAP50-95  | 0.4546 |

---

## MLflow Tracking

MLflow was used to:

* Track training runs
* Compare model experiments
* Log hyperparameters
* Store evaluation metrics
* Save model artifacts

---

## Results

The experiments showed that:

* YOLO11s outperformed YOLO11n and YOLO11m on this dataset.
* 640×640 images performed better than 320×320 images.
* Dataset tiling improved detection performance compared to the original dataset.
* The best-performing configuration achieved an mAP50-95 score of 0.4546.

---

