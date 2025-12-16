# 🚗 Teman Pamor ANPR - Hybrid Mobile-First MLOps

**Production-grade Automatic Number Plate Recognition** for Bapenda Jawa Barat

## 🏗️ Architecture: Hybrid "Mobile-First MLOps"

This project combines:
- **Mobile ML Pattern** → On-device TFLite inference
- **MLOps Best Practices** → Versioning, validation, monitoring
- **Platform Thinking** → Shared components, multi-model scalability

```
teman_pamor_anpr/
├── 00_platform/           # 🛠️  Shared infrastructure
├── 01_data_platform/      # 📊 Data management
├── 02_training_platform/  # 🤖 Model training
├── 03_deployment_platform/# 📱 Mobile deployment
└── 04_ci_cd/             # 🔄 Automation
```

## 🎯 4 AI Models

1. **Plate Detector** (YOLOv11) - ✅ Production (81.6% mAP)
2. **Color Classifier** (MobileNetV2) - 🔄 Development
3. **OCR Custom** (CRNN) - 📋 Backlog
4. **Anti-Spoofing** (Binary) - 📋 Backlog

## 🚀 Quick Start

### 1. Label Data
```bash
python 01_data_platform/labeling_tools/label_tool.py
```

### 2. Train Models
```bash
# Plate Detector
python 02_training_platform/01_plate_detector_yolo/src/train.py

# Color Classifier
python 02_training_platform/02_color_classifier/src/train.py
```

### 3. Test Inference
```bash
python 03_deployment_platform/inference/quick_test.py --image test.jpg
```

## 📚 Documentation

- [Architecture Decision](docs/architecture/01_ARCHITECTURE_DECISION.md) - Why this structure?
- [Color Classification Guide](docs/guides/02_COLOR_CLASSIFICATION.md) - Step-by-step
- [Training Guide](docs/guides/TRAINING_GUIDE.md) - Model training
- [Deployment Guide](docs/guides/DEPLOYMENT_GUIDE.md) - Mobile deployment

## 📊 Current Status

| Component | Status | Progress |
|-----------|--------|----------|
| Plate Detector | ✅ Production | 81.6% mAP (10K images) |
| Color Classifier | 🔄 Training | Dataset ready (55 images) |
| OCR Custom | 📋 Planned | Dataset acquired |
| Anti-Spoofing | 📋 Planned | Design phase |

## 🏆 Production Metrics

- **Latency**: < 200ms end-to-end (mobile)
- **Model Size**: < 10MB total (4 models)
- **Accuracy**: > 95% target
- **Platform**: Android + iOS (Flutter)

## 👥 Team

- ML Engineering Team
- Bapenda Jawa Barat

## 📄 License

Proprietary - Bapenda Jawa Barat
