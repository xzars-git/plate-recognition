# Training Guide

Practical training workflow for Teman Pamor ANPR models.

## Scope

This guide focuses on the current runnable training flow for:

- Plate Detector (YOLO)
- Color Classifier (when dataset is ready)

## Prerequisites

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run commands from repository root unless noted.

## Plate Detector Workflow

### 1. Build augmented dataset

```powershell
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/augment_dataset.py
```

Expected output dataset:

- `dataset/plate_detection_augmented/images/train`
- `dataset/plate_detection_augmented/images/val`
- `dataset/plate_detection_augmented/labels/train`
- `dataset/plate_detection_augmented/labels/val`

### 2. Create grouped validation split (anti leakage)

```powershell
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/create_grouped_val_split.py --dataset dataset/plate_detection_augmented --val-ratio 0.2 --seed 42
```

### 3. Train model

```powershell
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/train.py
```

### 4. Compare checkpoints

```powershell
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/compare_checkpoints.py
```

## Color Classifier (Current Status)

The folder structure is ready, but training depends on data readiness and script completeness in:

- `teman_pamor_anpr/02_training_platform/02_color_classifier/src/`

Use this when your dataset and train script are confirmed:

```powershell
python teman_pamor_anpr/02_training_platform/02_color_classifier/src/train.py
```

## Recommended Baseline Logging

Track these metrics per run:

- Precision
- Recall
- mAP50
- mAP50-95
- Inference speed (ms)
- Model size (MB)

Also record:

- Dataset source path
- Validation split seed and ratio
- Main hyperparameters (epochs, batch, imgsz)
