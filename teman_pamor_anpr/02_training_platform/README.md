# 🤖 Training Platform

Multi-model training with consistent structure

## Models

### 1. Plate Detector (`01_plate_detector_yolo/`)
- Architecture: YOLOv11n
- Status: ✅ Production
- Accuracy: 81.6% mAP

### 2. Color Classifier (`02_color_classifier/`)
- Architecture: MobileNetV2
- Status: 🔄 Development
- Classes: 4 (White, Black, Red, Yellow)

### 3. OCR Custom (`03_ocr_custom/`)
- Architecture: CRNN
- Status: 📋 Backlog

### 4. Anti-Spoofing (`04_anti_spoofing/`)
- Architecture: Binary Classifier
- Status: 📋 Backlog

## Structure (per model)

```
01_plate_detector_yolo/
├── experiments/      # Experiment tracking
├── configs/         # Training configs
├── src/            # Training code
├── checkpoints/    # Model weights
├── notebooks/      # Analysis
└── README.md       # Model docs
```

## Training

```bash
# Plate Detector
cd 01_plate_detector_yolo
python src/train.py --config configs/plat_jabar.yaml

# Color Classifier
cd 02_color_classifier
python src/train.py --epochs 30
```
