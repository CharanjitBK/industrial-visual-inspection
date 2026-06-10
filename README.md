# Industrial Visual Inspection Platform

## Overview

An AI-powered industrial defect detection system for steel surface inspection using YOLOv11 and PyTorch.

The project focuses on detecting and classifying surface defects from steel images, evaluating model performance through multiple experiments, tracking results with MLflow, and serving predictions through a FastAPI-based inference service.

## Dataset

### NEU Surface Defect Database

The dataset contains six defect categories:

* Crazing
* Inclusion
* Patches
* Pitted Surface
* Rolled-in Scale
* Scratches

### Dataset Preparation

* Train / Validation split
* YOLO annotation format conversion
* Dataset tiling for improved small-defect detection
* Label validation using OpenCV


## Tech Stack

### Machine Learning

* Python
* PyTorch
* YOLOv11 (Ultralytics)
* OpenCV
* NumPy
* Pandas

### Experiment Tracking

* MLflow

### API Development

* FastAPI
* Swagger UI

### Version Control

* Git
* GitHub

## Training Pipeline

1. Dataset Preparation
2. YOLO Dataset Conversion
3. Dataset Tiling
4. YOLOv11 Model Training
5. Model Evaluation
6. MLflow Experiment Tracking
7. Best Model Selection

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

## Experiment Results

| Model   | Dataset | Image Size | mAP50  | mAP50-95 |
| ------- | ------- | ---------- | ------ | -------- |
| YOLO11s | Tiled   | 640        | 0.7814 | 0.4546   |
| YOLO11s | Normal  | 640        | 0.7790 | 0.4513   |
| YOLO11n | Tiled   | 320        | 0.7580 | 0.4382   |
| YOLO11s | Tiled   | 320        | 0.7572 | 0.4371   |
| YOLO11m | Tiled   | 320        | 0.7509 | 0.4361   |

## Best Model

* **Model:** YOLO11s
* **Dataset:** Tiled NEU Dataset
* **Image Size:** 640 × 640
* **Epochs:** 100

### Performance Metrics

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.7453 |
| Recall    | 0.7096 |
| mAP50     | 0.7814 |
| mAP50-95  | 0.4546 |


## Inference Pipeline

Image Upload

↓

YOLO11s Defect Detection

↓

Risk Scoring Engine

↓

Severity Classification

↓

Annotated Image Generation

↓

JSON Response


## Risk Scoring Engine

The inference pipeline includes a custom risk assessment module that evaluates:

* Number of detected defects
* Detection confidence scores
* Defect-specific severity weights

Severity levels:

* Low
* Medium
* Critical

## API Service

FastAPI is used to expose the trained model through a REST API.

### Endpoint

**POST /predict**

Input:

* Image file

Output:

* Detected defects
* Bounding boxes
* Confidence scores
* Risk score
* Severity classification
* Annotated image path

### Interactive Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## MLflow Tracking

MLflow was used for:

* Tracking training runs
* Comparing experiments
* Logging hyperparameters
* Storing evaluation metrics
* Saving model artifacts

## Key Findings

<<<<<<< HEAD
- YOLO11s outperformed YOLO11n and YOLO11m on this dataset
- 640×640 input resolution performed better than 320×320
- Dataset tiling improved detection of small defects
- Best configuration achieved strong balance between precision and recall
=======
* YOLO11s achieved the best overall performance on the NEU dataset
* 640 × 640 image resolution outperformed 320 × 320
* Dataset tiling improved detection performance for small defects
* MLflow simplified experiment comparison and model selection
* FastAPI successfully deployed the trained model as an inference service
* Risk scoring and visualization enhanced the interpretability of predictions
>>>>>>> ce59f53 (Add FastAPI defect detection API with risk scoring)
