# 🚀 Getting Started with Teman Pamor ANPR

**Complete guide** untuk mulai development di project Teman Pamor ANPR.

---

## 📋 Prerequisites

### Required

- **Python 3.10+** (tested on 3.10.x, 3.11.x)
- **Git** (version control)
- **8GB+ RAM** (recommended 16GB)
- **Windows 10/11** or **Linux** or **macOS**

### Optional (Recommended)

- **NVIDIA GPU** (RTX 2060+) with CUDA 11.8+ for training
- **VS Code** with Python extension
- **Jupyter Notebook** for data exploration

---

## 🏗️ Understanding the Architecture

Sebelum mulai, penting untuk understand struktur project:

```
teman_pamor_anpr/
├── 00_platform/           # Shared utilities (USE EVERYWHERE)
│   ├── data_validation/   # Validate image quality
│   ├── preprocessing/     # Image preprocessing
│   └── model_registry/    # Track model versions
│
├── 01_data_platform/      # Data management (LABEL HERE)
│   ├── labeling_tools/    # GUI for labeling
│   ├── datasets/         # Organized data pipeline
│   │   ├── 00_raw/       # Original data (IMMUTABLE)
│   │   ├── 01_validated/ # Quality-checked data
│   │   └── 02_augmented/ # Training-ready data
│   └── notebooks/        # Data exploration
│
├── 02_training_platform/  # Train models (4 MODELS)
│   ├── 01_plate_detector_yolo/
│   ├── 02_color_classifier/
│   ├── 03_ocr_custom/
│   └── 04_anti_spoofing/
│
├── 03_deployment_platform/# Mobile deployment (TFLITE)
│   ├── conversion/       # Convert to TFLite
│   ├── validation/       # Validate converted models
│   ├── mobile_models/    # Versioned models (v1.0.0, v1.1.0)
│   └── flutter_integration/
│
└── 04_ci_cd/             # Automation (OPTIONAL)
    ├── .github/          # GitHub Actions
    └── scripts/          # Automation scripts
```

### **Key Principles:**

1. **DRY (Don't Repeat Yourself)** - Shared code di `00_platform/`
2. **Data Pipeline** - raw → validated → augmented → training
3. **Multi-Model** - 4 models, identical structure
4. **Mobile-First** - Always optimize for TFLite (size + latency)
5. **Versioning** - Track everything (data, models, experiments)

---

## 🔧 Initial Setup

### Step 1: Clone Repository

```powershell
# Clone from GitHub
git clone https://github.com/xzars-git/plate-recognition.git
cd plate-recognition
```

### Step 2: Create Virtual Environment

```powershell
# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
# Install root dependencies
pip install -r requirements.txt

# Enter main project
cd teman_pamor_anpr

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```powershell
# Check Python version
python --version  # Should be 3.10+

# Check key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
```

### Step 5: Test GPU (Optional)

```powershell
# Check CUDA availability
python 03_deployment_platform/inference/check_gpu.py
```

**Expected output:**

```
🔍 GPU Check:
   PyTorch CUDA: ✅ Available (CUDA 11.8)
   TensorFlow GPU: ✅ Available (2 devices)
   Device: NVIDIA GeForce RTX 3080 Ti
```

---

## 🎯 Your First Tasks

### Task 1: Explore Data (10 minutes)

```powershell
cd 01_data_platform/datasets/00_raw
```

**Check what data you have:**

- `plate_detection/` - For YOLO training (10K+ images)
- `color_classification/` - For color classifier (55+ images)
- `ocr_recognition/` - For OCR (TBD)
- `anti_spoofing/` - For anti-spoofing (TBD)

### Task 2: Run Label Tool (15 minutes)

```powershell
cd teman_pamor_anpr
python 01_data_platform/labeling_tools/label_tool.py
```

**Try:**

1. Load image folder
2. Draw bounding box (rectangle mode)
3. Draw polygon ([P] key)
4. Close polygon ([F] key)
5. Save annotations ([S] key)

### Task 3: Test Plate Detector (5 minutes)

```powershell
# Quick inference test
python 03_deployment_platform/inference/quick_test.py
```

**This will:**

- Load production model (`best.pt` - 81.6% mAP)
- Run inference on test image
- Show detection results

### Task 4: Browse Model Registry (5 minutes)

```powershell
# View model registry
cat 00_platform/model_registry/registry.json
```

**You'll see:**

- 4 models tracked
- Current versions
- Accuracy metrics
- Status (production / development / backlog)

---

## 📚 Next Steps: Choose Your Path

### Path A: Continue Plate Detector Training 🎯

**Best for:** Improving existing model

```powershell
cd 02_training_platform/01_plate_detector_yolo
python src/train.py
```

**Goals:**

- Current: 81.6% mAP
- Target: 90%+ mAP
- Add more training data
- Experiment with augmentation

**Read:** [YOLO Training Guide](TRAINING_GUIDE.md#plate-detector)

---

### Path B: Train Color Classifier 🎨

**Best for:** Quick wins, new model

```powershell
cd 02_training_platform/02_color_classifier
python src/train.py
```

**Goals:**

- Train MobileNetV2
- Classify plate colors (Hitam, Putih, Merah, Kuning)
- Target: >90% accuracy, <20ms latency

**Read:** [Color Classification Guide](02_COLOR_CLASSIFICATION.md)

---

### Path C: Implement OCR Custom 📝

**Best for:** Advanced ML, challenge

```powershell
cd 02_training_platform/03_ocr_custom
# Start from scratch (dataset available)
```

**Goals:**

- Implement CRNN architecture
- Train with CTC loss
- Handle O/0 confusion
- Target: >95% character accuracy

**Read:** [OCR Implementation Guide](03_OCR_CUSTOM.md) (TODO)

---

### Path D: Label More Data 🏷️

**Best for:** Data quality, immediate impact

```powershell
python 01_data_platform/labeling_tools/label_tool.py
```

**Goals:**

- Label 1000+ more images
- Improve data diversity
- Better validation set

**Read:** [Labeling Guide](LABELING_GUIDE.md) (TODO)

---

## 🔍 Understanding Data Pipeline

### Stage 1: Raw Data (`00_raw/`)

**Purpose:** Immutable source of truth

```
00_raw/
├── plate_detection/
│   ├── train/
│   │   ├── images/    # Original images
│   │   └── labels/    # YOLO format (.txt)
│   └── val/
```

**Rules:**

- ❌ NEVER modify files here
- ✅ Only add new data
- ✅ Keep backups

### Stage 2: Validated Data (`01_validated/`)

**Purpose:** Quality-checked, ready for training

```
01_validated/
└── plate_detection/
    ├── train/
    └── val/
```

**Process:**

```python
from platform.data_validation import ImageValidator

validator = ImageValidator()
validator.validate_directory("00_raw/plate_detection/")
validator.export_validated("01_validated/plate_detection/")
```

**Checks:**

- ✅ Image not corrupted
- ✅ Resolution >= 640x640
- ✅ File size > 10KB
- ✅ Labels valid format

### Stage 3: Augmented Data (`02_augmented/`)

**Purpose:** Training-ready with augmentation

```
02_augmented/
└── plate_detection/
    ├── train/        # 10K → 50K images (5x augmentation)
    └── val/          # No augmentation
```

**Process:**

```python
from platform.preprocessing import augment_dataset

augment_dataset(
    source="01_validated/plate_detection/train/",
    target="02_augmented/plate_detection/train/",
    augmentation_config="configs/augmentation.yaml"
)
```

---

## 🎓 Training Your First Model

### Example: Color Classifier

**Step 1: Prepare Data**

```powershell
cd 02_training_platform/02_color_classifier
```

Check data structure:

```
datasets/ → symlink to ../../01_data_platform/datasets/
```

**Step 2: Configure Training**

Edit `configs/training.yaml`:

```yaml
model:
  architecture: MobileNetV2
  input_size: [96, 96, 3]
  alpha: 0.35

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

**Step 3: Run Training**

```powershell
python src/train.py --config configs/training.yaml
```

**Step 4: Monitor Progress**

Training logs will be saved to:

```
experiments/
└── exp_001_baseline/
    ├── weights/
    │   ├── best.h5
    │   └── last.h5
    ├── logs/
    │   └── training.log
    └── plots/
        ├── loss.png
        └── accuracy.png
```

**Step 5: Convert to TFLite**

```powershell
cd ../../03_deployment_platform/conversion
python convert_to_tflite.py \
    --model ../02_training_platform/02_color_classifier/experiments/exp_001/weights/best.h5 \
    --output color_classifier_v1.0.0.tflite
```

**Step 6: Validate**

```powershell
cd ../validation
python validate_tflite.py \
    --model ../mobile_models/v1.0.0/color_classifier_v1.0.0.tflite \
    --test_data ../../01_data_platform/datasets/01_validated/color_classification/val/
```

---

## 🐛 Troubleshooting

### Issue: Import Error

```
ModuleNotFoundError: No module named 'platform.preprocessing'
```

**Solution:**

```powershell
# Make sure you're in teman_pamor_anpr/
cd teman_pamor_anpr

# Add to PYTHONPATH
$env:PYTHONPATH = "."
```

### Issue: CUDA Not Available

```
PyTorch CUDA: ❌ Not available
```

**Solution:**

```powershell
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of Memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**

- Reduce batch size in config
- Close other GPU applications
- Use gradient accumulation

### Issue: Label Tool Not Opening

```
No GUI window appears
```

**Solution:**

```powershell
# Install Tkinter (Windows)
# Already included in Python

# Install Tkinter (Linux)
sudo apt-get install python3-tk

# Test
python -c "import tkinter; print('OK')"
```

---

## 📖 Further Reading

### Internal Documentation

- [Architecture Decision Record](../architecture/01_ARCHITECTURE_DECISION.md)
- [Color Classification Guide](02_COLOR_CLASSIFICATION.md)
- [Training Guide](TRAINING_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

### External Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)
- [CRNN Paper](https://arxiv.org/abs/1507.05717)

---

## 🤝 Contributing

### Adding New Model

1. Copy template structure:

```powershell
cp -r 02_training_platform/02_color_classifier 02_training_platform/05_new_model
```

2. Update model name in:

- `README.md`
- `src/train.py`
- `configs/training.yaml`

3. Register in model registry:

```json
{
  "new_model": {
    "version": "0.0.1",
    "status": "development"
  }
}
```

### Code Style

- Use **Black** for formatting: `black .`
- Use **type hints**: `def function(x: int) -> str:`
- Add **docstrings**: Google style
- Write **tests**: `tests/unit/test_model.py`

---

## 🎯 Success Checklist

After completing this guide, you should be able to:

- [ ] Understand project structure (5 layers)
- [ ] Install dependencies and verify setup
- [ ] Run label tool and label images
- [ ] Test plate detector inference
- [ ] View model registry
- [ ] Choose development path (A/B/C/D)
- [ ] Run training for one model
- [ ] Convert model to TFLite
- [ ] Validate TFLite model

---

**Next:** Choose your path and start developing! 🚀

**Questions?** Check [FAQ](FAQ.md) or create GitHub issue.
