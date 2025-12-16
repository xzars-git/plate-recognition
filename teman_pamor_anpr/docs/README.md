# 📚 Documentation Index

**Complete documentation** for Teman Pamor ANPR system.

---

## 🚀 Getting Started

New to the project? Start here:

1. **[Getting Started Guide](guides/01_GETTING_STARTED.md)** - Complete setup & first tasks
2. **[Architecture Decision Record](architecture/01_ARCHITECTURE_DECISION.md)** - Why this structure?
3. **[Project README](../README.md)** - Main documentation

---

## 📖 Guides

### For Data Scientists

- **[Color Classification Guide](guides/02_COLOR_CLASSIFICATION.md)** - Train color classifier
- **Training Guide** (TODO) - Model training workflows
- **Data Labeling Guide** (TODO) - Labeling best practices

### For ML Engineers

- **Deployment Guide** (TODO) - TFLite conversion & validation
- **Model Optimization Guide** (TODO) - Quantization, pruning
- **CI/CD Setup** (TODO) - Automation workflows

### For Mobile Developers

- **Flutter Integration Guide** (TODO) - Integrate TFLite models
- **Performance Tuning** (TODO) - Optimize mobile inference
- **API Documentation** (TODO) - Model inference APIs

---

## 🏗️ Architecture

- **[Architecture Decision Record](architecture/01_ARCHITECTURE_DECISION.md)** ⭐
  - 5 alternatives evaluated
  - Why Hybrid "Mobile-First MLOps"
  - Trade-offs & risks
  - Implementation plan

---

## 📋 Component Documentation

Each component has its own README:

### Platform Layer

- **[00_platform/README.md](../00_platform/README.md)** - Shared utilities

### Data Platform

- **[01_data_platform/README.md](../01_data_platform/README.md)** - Data management

### Training Platform

- **[02_training_platform/README.md](../02_training_platform/README.md)** - Model training
  - **[Plate Detector](../02_training_platform/01_plate_detector_yolo/README.md)** - YOLOv11n
  - **[Color Classifier](../02_training_platform/02_color_classifier/README.md)** - MobileNetV2
  - **[OCR Custom](../02_training_platform/03_ocr_custom/README.md)** - CRNN
  - **[Anti-Spoofing](../02_training_platform/04_anti_spoofing/README.md)** - Binary CNN

### Deployment Platform

- **[03_deployment_platform/README.md](../03_deployment_platform/README.md)** - Mobile deployment

### CI/CD

- **[04_ci_cd/README.md](../04_ci_cd/README.md)** - Automation

---

## 🔬 Research & References

### Papers

- **YOLOv11:** [Ultralytics YOLO11](https://docs.ultralytics.com/)
- **MobileNet:** [MobileNets: Efficient CNNs for Mobile Vision](https://arxiv.org/abs/1704.04861)
- **CRNN:** [An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/abs/1507.05717)
- **CTC Loss:** [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

### Industry Best Practices

- **Google MLOps:** [Practitioners Guide to MLOps](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf)
- **Uber Michelangelo:** [Meet Michelangelo: Uber's ML Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)
- **TFLite Best Practices:** [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)

### Books

- **"Building Machine Learning Powered Applications"** by Emmanuel Ameisen
- **"Designing Data-Intensive Applications"** by Martin Kleppmann
- **"Machine Learning Systems Design"** by Chip Huyen

---

## 📊 Performance Benchmarks

### Model Performance

| Model            | Accuracy  | Latency (Mobile) | Size   | Status         |
| ---------------- | --------- | ---------------- | ------ | -------------- |
| Plate Detector   | 81.6% mAP | 50ms             | 6.2 MB | ✅ Production  |
| Color Classifier | TBD       | <15ms            | 2.0 MB | 🔄 Development |
| OCR Custom       | TBD       | <50ms            | 8.0 MB | 📋 Backlog     |
| Anti-Spoofing    | TBD       | <30ms            | 3.0 MB | 📋 Backlog     |

### System Performance

- **End-to-end latency:** <200ms (target)
- **Total model size:** ~19 MB (target <20MB)
- **Mobile platforms:** Android 8.0+, iOS 12.0+
- **Inference framework:** TensorFlow Lite

---

## 🛠️ Tools & Technologies

### ML Frameworks

- **PyTorch** 2.0+ - Plate detector training
- **TensorFlow** 2.x - Color classifier, OCR, anti-spoofing
- **Ultralytics YOLO** 8.x - YOLO training framework

### Deployment

- **TensorFlow Lite** - Mobile inference
- **Flutter** - Mobile app (Android + iOS)
- **ONNX** - Model interoperability

### Development

- **Python** 3.10+ - Main language
- **Git** - Version control
- **VS Code** - IDE
- **Jupyter** - Data exploration

---

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**

```powershell
# Add project to PYTHONPATH
$env:PYTHONPATH = "."
```

**CUDA Not Available:**

```powershell
# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Out of Memory:**

- Reduce batch size in config
- Use gradient accumulation
- Close other applications

**More:** See [Getting Started - Troubleshooting](guides/01_GETTING_STARTED.md#troubleshooting)

---

## 🤝 Contributing

### Adding Documentation

1. Create new file in `guides/` or `architecture/`
2. Follow naming: `NN_TITLE.md` (e.g., `03_OCR_GUIDE.md`)
3. Update this index
4. Submit PR

### Documentation Standards

- Use **Markdown** format
- Add **emojis** for readability (🎯📊✅)
- Include **code examples** with language tags
- Add **images** to `docs/images/` if needed
- Keep it **concise** and **actionable**

---

## 📞 Support

### Internal

- **ML Team Channel:** Slack #ml-team
- **Issue Tracker:** GitHub Issues
- **Wiki:** GitHub Wiki

### External

- **Ultralytics Forum:** [community.ultralytics.com](https://community.ultralytics.com/)
- **TensorFlow Forum:** [tensorflow.org/community](https://www.tensorflow.org/community)
- **Stack Overflow:** Tag `tensorflow-lite`, `yolo`, `anpr`

---

## 📅 Changelog

### 2025-12-16 - Initial Documentation

- ✅ Created documentation structure
- ✅ Added Architecture Decision Record
- ✅ Added Getting Started guide
- ✅ Added Color Classification guide
- ✅ Migrated to Hybrid architecture

### Future

- ⏳ Add Training Guide
- ⏳ Add Deployment Guide
- ⏳ Add API Documentation
- ⏳ Add Performance Tuning guide

---

## 📝 Quick Links

| Document                                                     | Description            | Status      |
| ------------------------------------------------------------ | ---------------------- | ----------- |
| [Getting Started](guides/01_GETTING_STARTED.md)              | Setup & first tasks    | ✅ Complete |
| [Architecture ADR](architecture/01_ARCHITECTURE_DECISION.md) | Architecture decision  | ✅ Complete |
| [Color Classification](guides/02_COLOR_CLASSIFICATION.md)    | Color classifier guide | ✅ Complete |
| Training Guide                                               | Model training         | 📋 TODO     |
| Deployment Guide                                             | TFLite deployment      | 📋 TODO     |
| API Documentation                                            | Model APIs             | 📋 TODO     |

---

**Last Updated:** 2025-12-16  
**Version:** 1.0  
**Maintained by:** ML Engineering Team
