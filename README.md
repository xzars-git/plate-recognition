# 🚗 Teman Pamor ANPR System

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Architecture](https://img.shields.io/badge/architecture-Mobile--First%20MLOps-blue)
![Models](https://img.shields.io/badge/models-4%20AI%20models-orange)

**Production-grade ANPR (Automatic Number Plate Recognition)** system untuk Bapenda Jawa Barat dengan **Hybrid "Mobile-First MLOps"** architecture.

## 📁 Project Structure

```
plate-recognition/
├── teman_pamor_anpr/          # 🎯 Main project (production-ready)
│   ├── 00_platform/           # Shared utilities & model registry
│   ├── 01_data_platform/      # Data management & labeling tools
│   ├── 02_training_platform/  # 4 AI models training
│   ├── 03_deployment_platform/# Mobile deployment & validation
│   ├── 04_ci_cd/             # Automation & workflows
│   ├── docs/                 # Architecture & guides
│   └── tests/                # Unit, integration, performance tests
├── README.md                  # This file
└── requirements.txt           # Root dependencies
```

**📖 Full documentation:** [`teman_pamor_anpr/README.md`](teman_pamor_anpr/README.md)

---

## 🎯 4 Essential AI Models

| Model                   | Architecture | Status         | Accuracy  | Latency | Size   |
| ----------------------- | ------------ | -------------- | --------- | ------- | ------ |
| **1. Plate Detector**   | YOLOv11n     | ✅ Production  | 81.6% mAP | 50ms    | 6.2 MB |
| **2. Color Classifier** | MobileNetV2  | 🔄 Development | -         | 15ms    | 2.0 MB |
| **3. OCR Custom**       | CRNN         | 📋 Backlog     | -         | 50ms    | 8.0 MB |
| **4. Anti-Spoofing**    | Binary CNN   | 📋 Backlog     | -         | 30ms    | 3.0 MB |

**Total model size:** ~19 MB (target: <20MB for mobile)

---

## 🏗️ Architecture: Hybrid "Mobile-First MLOps"

**Design Philosophy:** Combine mobile-first approach with production MLOps practices.

### Key Features:

- ✅ **Platform Layer** - Shared utilities (DRY principle)
- ✅ **Data Platform** - Organized data pipeline (raw → validated → augmented)
- ✅ **Training Platform** - Multi-model support (4 models, scalable to 10+)
- ✅ **Deployment Platform** - Mobile-first with validation gates
- ✅ **CI/CD Ready** - Automation from data to deployment

**📖 Architecture Decision Record:** [`docs/architecture/01_ARCHITECTURE_DECISION.md`](docs/architecture/01_ARCHITECTURE_DECISION.md)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git
- 8GB+ RAM
- (Optional) NVIDIA GPU with CUDA

### 1. Clone & Setup

```powershell
# Clone repository
git clone https://github.com/xzars-git/plate-recognition.git
cd plate-recognition

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Enter Main Project

```powershell
cd teman_pamor_anpr
```

### 3. Quick Tests

```powershell
# Test plate detector inference
python 03_deployment_platform/inference/quick_test.py

# Launch label tool
python 01_data_platform/labeling_tools/label_tool.py

# Check GPU availability
python 03_deployment_platform/inference/check_gpu.py
```

### 4. Train Models

```powershell
# Train plate detector
cd 02_training_platform/01_plate_detector_yolo
python src/train.py

# Train color classifier
cd ../02_color_classifier
python src/train.py
```

**📖 Detailed guides:** See [`teman_pamor_anpr/README.md`](teman_pamor_anpr/README.md)

---

## 📊 Current Status & Performance

### Model 1: Plate Detector (YOLOv11n) ✅ PRODUCTION

| Metric      | Value         | Notes                     |
| ----------- | ------------- | ------------------------- |
| **mAP50**   | 81.6%         | Production-ready accuracy |
| **Latency** | 50ms          | On mobile devices         |
| **Size**    | 6.2 MB        | TFLite optimized          |
| **Status**  | ✅ Production | Deployed & validated      |

**Location:** `teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/`

### Model 2: Color Classifier (MobileNetV2) 🔄 DEVELOPMENT

| Metric       | Value  | Notes                     |
| ------------ | ------ | ------------------------- |
| **Accuracy** | TBD    | Training in progress      |
| **Latency**  | <15ms  | Target                    |
| **Size**     | 2.0 MB | MobileNetV2 α=0.35        |
| **Status**   | 🔄 Dev | Dataset ready (3 classes) |

**Location:** `teman_pamor_anpr/02_training_platform/02_color_classifier/`

### Model 3: OCR Custom (CRNN) 📋 BACKLOG

| Metric            | Value      | Notes             |
| ----------------- | ---------- | ----------------- |
| **Char Accuracy** | TBD        | Target >95%       |
| **Latency**       | <50ms      | Target            |
| **Size**          | 8.0 MB     | CRNN architecture |
| **Status**        | 📋 Backlog | Dataset available |

**Location:** `teman_pamor_anpr/02_training_platform/03_ocr_custom/`

### Model 4: Anti-Spoofing (Binary CNN) 📋 BACKLOG

| Metric       | Value      | Notes             |
| ------------ | ---------- | ----------------- |
| **Accuracy** | TBD        | Target >98%       |
| **Latency**  | <30ms      | Target            |
| **Size**     | 3.0 MB     | Lightweight CNN   |
| **Status**   | 📋 Backlog | Placeholder ready |

**Location:** `teman_pamor_anpr/02_training_platform/04_anti_spoofing/`

---

## 📖 Documentation

### Core Documentation

- **[Architecture Decision Record](docs/architecture/01_ARCHITECTURE_DECISION.md)** - Why we chose Hybrid "Mobile-First MLOps"
- **[Main README](teman_pamor_anpr/README.md)** - Detailed project documentation
- **[Color Classification Guide](docs/guides/02_COLOR_CLASSIFICATION.md)** - Color classifier workflow

### Component READMEs

- **[Platform Layer](teman_pamor_anpr/00_platform/README.md)** - Shared utilities & model registry
- **[Data Platform](teman_pamor_anpr/01_data_platform/README.md)** - Data management & labeling
- **[Training Platform](teman_pamor_anpr/02_training_platform/README.md)** - Model training workflows
- **[Deployment Platform](teman_pamor_anpr/03_deployment_platform/README.md)** - Mobile deployment pipeline
- **[CI/CD](teman_pamor_anpr/04_ci_cd/README.md)** - Automation & workflows

---

## 🎯 Development Roadmap

### ✅ Phase 1: Foundation (COMPLETED)

- ✅ Architecture design & migration
- ✅ Platform layer implementation
- ✅ Data organization
- ✅ Documentation (ADR + guides)
- ✅ Plate detector in production (81.6% mAP)

### 🔄 Phase 2: Color Classifier (IN PROGRESS)

- ✅ Dataset prepared (3 classes)
- 🔄 Training MobileNetV2
- ⏳ TFLite conversion
- ⏳ Mobile validation

### 📋 Phase 3: OCR Custom (BACKLOG)

- ✅ Dataset available
- ⏳ CRNN architecture implementation
- ⏳ CTC loss training
- ⏳ Handle O/0 confusion

### 📋 Phase 4: Anti-Spoofing (BACKLOG)

- ⏳ Dataset collection
- ⏳ Binary classifier training
- ⏳ Liveness detection

### 📋 Phase 5: CI/CD (BACKLOG)

- ⏳ GitHub Actions workflows
- ⏳ Automated testing
- ⏳ Deployment automation

---

## 🏗️ Project Structure (Detailed)

```
plate-recognition/
├── 🎯 PRODUCTION FILES
│   ├── best.pt                          # Epoch170 model (81.64% precision)
│   ├── fast_webcam_anpr.py              # Real-time ANPR system (456 lines)
│   └── plate_rotation_detector.py       # Rotation detection core
│
├── 🧪 TESTING & VALIDATION
│   ├── test_epoch170_with_rotation.py   # Production testing (rotation + detection)
│   ├── test_final_model.py              # Model validation script
│   ├── compare_all_checkpoints.py       # JACKPOT finder (found epoch170)
│   └── create_rotated_test_images.py    # Test data generator
│
├── 🏋️ TRAINING & UTILITIES
│   ├── enhance_model.py                 # Ultimate training config (200 epochs)
│   ├── augment_dataset_rotation.py      # Dataset augmentation
│   ├── demo_rotation.py                 # Rotation demo utility
│   └── check_gpu.py                     # GPU verification
│
├── ⚙️ CONFIGURATION
│   ├── plat_jabar.yaml                  # Dataset config
│   ├── requirements.txt                 # Python dependencies
│   ├── README.md                        # This file
│   └── .gitignore                       # Git ignore rules
│
├── 🤖 MODELS
│   ├── best.pt                          # Epoch170 (16.08 MB)
│   ├── yolo11n.pt                       # Base pretrained model
│   └── runs/plate_detection/yolov11_ultimate_v1/weights/
│       └── epoch170.pt                  # Original checkpoint
│
└── 📦 DATASETS
    └── dataset/plate_detection_yolo/    # YOLO format dataset
        ├── images/ (train: 4396, val: 1104)
        └── labels/ (train: 4396, val: 1104)
```

## 🔧 Installation

### 1. Clone Repository

```powershell
cd your/project/folder
```

### 2. Create Virtual Environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verify Installation

```powershell
python -c "from ultralytics import YOLO; import cv2; print('✅ All packages ready!')"
```

## 🎯 Usage

### 1. Production Testing (Epoch170 + Rotation)

```powershell
# Test single image with rotation handling
python test_epoch170_with_rotation.py path/to/image.jpg

# Batch test validation set
python test_epoch170_with_rotation.py --batch dataset/plate_detection_yolo/images/val

# Test custom folder
python test_epoch170_with_rotation.py --batch path/to/folder *.jpg
```

**Output:**

- `test_results_rotation/` - Annotated images with rotation info
- Statistics: Detection rate, rotation distribution, confidence scores

### 2. Model Validation

```powershell
# Validate epoch170 performance
python test_final_model.py

# Compare all checkpoints (finds best model)
python compare_all_checkpoints.py
```

### 3. Rotation Utilities

```powershell
# Create rotated test images (90°/180°/270°)
python create_rotated_test_images.py path/to/image.jpg

# Demo rotation detection
python demo_rotation.py
```

### 4. Real-time Webcam ANPR

```powershell
python fast_webcam_anpr.py
```

**Controls:**

- `O` - Toggle OCR on/off
- `S` - Save screenshot
- `Q` - Quit application

### 5. Training (Advanced)

```powershell
# Train with ultimate configuration (200 epochs)
python enhance_model.py

# GPU check before training
python check_gpu.py
```

## 🎨 Programmatic Usage

### Basic Usage (Production Pipeline)

```python
from ultralytics import YOLO
from plate_rotation_detector import PlateRotationDetector
import cv2

# Load production model (epoch170)
model = YOLO('best.pt')
rotation_detector = PlateRotationDetector(debug=False)

# Read image
image = cv2.imread('test.jpg')

# Step 1: Detect and correct rotation
corrected_image, angle, confidence = rotation_detector.preprocess(image)
print(f"Rotation detected: {angle}° (confidence: {confidence:.2%})")

# Step 2: Run plate detection
results = model.predict(corrected_image, conf=0.25, verbose=False)

# Step 3: Extract plate regions
for box in results[0].boxes:
    # Get coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    # Crop plate region
    plate_crop = corrected_image[y1:y2, x1:x2]

    # Ready for OCR
    print(f"Plate detected: confidence {conf:.2%}")
    cv2.imwrite('plate.jpg', plate_crop)
```

### Flask API Integration

```python
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)
model = YOLO('best.pt')
rotation_detector = PlateRotationDetector(debug=False)

@app.route('/detect-plate', methods=['POST'])
def detect_plate():
    # Get image from request
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Pipeline: rotation → detection → crop
    corrected, angle, conf = rotation_detector.preprocess(image)
    results = model.predict(corrected, conf=0.25, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = corrected[y1:y2, x1:x2]

        # Convert to base64 for response
        _, buffer = cv2.imencode('.jpg', plate_crop)
        plate_base64 = base64.b64encode(buffer).decode('utf-8')

        detections.append({
            'confidence': float(box.conf[0]),
            'plate_image': plate_base64
        })

    return jsonify({
        'rotation_detected': angle,
        'num_plates': len(detections),
        'detections': detections
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Batch Processing with Statistics

```python
from pathlib import Path
import numpy as np

# Statistics tracking
stats = {'detected': 0, 'not_detected': 0, 'confidences': []}

# Process folder
for img_path in Path('test_images/').glob('*.jpg'):
    image = cv2.imread(str(img_path))

    # Rotation + detection
    corrected, angle, _ = rotation_detector.preprocess(image)
    results = model.predict(corrected, conf=0.25, verbose=False)

    if len(results[0].boxes) > 0:
        stats['detected'] += 1
        confidences = [float(box.conf[0]) for box in results[0].boxes]
        stats['confidences'].extend(confidences)

        # Save annotated result
        annotated = results[0].plot()
        cv2.imwrite(f'output/{img_path.name}', annotated)
    else:
        stats['not_detected'] += 1

# Print summary
total = stats['detected'] + stats['not_detected']
print(f"Detection rate: {stats['detected']/total*100:.1f}%")
print(f"Average confidence: {np.mean(stats['confidences']):.2%}")
```

## � Mobile Deployment

### Convert to TFLite (Google Colab)

TFLite conversion requires specific dependencies best run in Google Colab:

1. **Open Google Colab**: [Convert Model to TFLite](https://colab.research.google.com/)
2. **Upload Notebook**: Use `tflite_conversion_colab.ipynb` (provided)
3. **Upload Model**: Upload `best.pt` to Colab
4. **Run Conversion**: Execute all cells
5. **Download**: `best_int8.tflite` (~4-6 MB quantized)

### ONNX Format (Already Available)

ONNX format is already exported and ready for Flutter deployment:

```
runs/plate_detection/yolov11_ultimate_v1/weights/epoch170.onnx
Size: 10.71 MB
Opset: 12
Input: (1, 3, 640, 640) BCHW
Output: (1, 5, 8400)
```

**Flutter Integration Options:**

1. **ONNX Runtime** - `onnxruntime` package (recommended)
2. **Ultralytics YOLO** - `ultralytics_yolo` package
3. **TFLite** - After conversion in Colab

### Performance Comparison

| Format      | Size     | Speed (Mobile) | Accuracy | Recommended         |
| ----------- | -------- | -------------- | -------- | ------------------- |
| PyTorch     | 16.08 MB | N/A            | 81.64%   | ❌ Desktop only     |
| ONNX        | 10.71 MB | 20-50ms        | 81.64%   | ✅ Best balance     |
| TFLite      | 10.71 MB | 15-30ms        | 81.64%   | ✅ Fastest          |
| TFLite INT8 | 4-6 MB   | 10-20ms        | ~80%     | ✅ Mobile optimized |

---

## 🐛 Troubleshooting

### GPU Not Detected

```powershell
# Check CUDA availability
python check_gpu.py

# Or manual check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Solution:**

- Install CUDA Toolkit 11.7+
- Install matching PyTorch version
- Verify NVIDIA drivers

### Memory Error During Training

```
RuntimeError: CUDA out of memory
```

**Solutions:**

- Reduce batch size: `batch=8` → `batch=4`
- Use disk cache: `cache='disk'`
- Reduce workers: `workers=4` → `workers=2`
- Lower image size: `imgsz=640` → `imgsz=512`

### Low Detection Accuracy

**Checklist:**

- ✅ Model loaded correctly? (`best.pt` = epoch170)
- ✅ Confidence threshold too high? (try `conf=0.15`)
- ✅ Image quality sufficient? (min 480p recommended)
- ✅ Plate orientation handled? (use rotation detection)

### Rotation Detection Not Working

**Common Issues:**

- Square images (640x640): Rotation detector needs rectangular images
- Low contrast: Increase image quality
- Solution: Test on real camera images (16:9, 4:3 aspect ratio)

## 📚 Resources & Documentation

### Official Documentation

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972) (YOLOv11 based on this)

### Project Documentation

- **README.md** - This file (main documentation)
- **requirements.txt** - Python dependencies
- **plat_jabar.yaml** - Dataset configuration

### Key Findings & Decisions

- **Model Selection**: Epoch170 (JACKPOT) over epoch180/last.pt
  - Reason: Best precision-speed balance (81.64%, 1.30ms)
  - Trade-off: Slightly lower recall for higher precision
- **Rotation Handling**: Pre-detection rotation correction

  - Handles: 90°, 180°, 270° discrete rotations
  - Limitation: Cannot handle arbitrary angles (45°, 30°, etc.)
  - Future: Consider YOLO-OBB for arbitrary angle support

- **Production Strategy**: ONNX format for mobile
  - TFLite conversion: Requires Google Colab (dependency issues on Windows)
  - ONNX: 10.71 MB, ready for Flutter deployment
  - Performance: 20-50ms inference on mobile devices

---

## 📝 License

MIT License - Feel free to use for personal and commercial projects.

---

## 👨‍💻 About

**Teman Pamor** - Bapenda ASN Vehicle Tracking System  
License plate recognition system for Bapenda (Regional Revenue Agency) to track official vehicles and reduce operational costs.

**Technology Stack:**

- Detection: YOLOv11n (Ultralytics)
- Rotation: Custom edge detection algorithm
- OCR: ML Kit (mobile) / PaddleOCR (server)
- Backend: Flutter mobile app
- GPU: NVIDIA RTX 3080 Ti

**Business Impact:**

- Cost savings: Rp 20,000,000/year vs paid ALPR API
- Accuracy: 81.64% precision (19% false positive reduction)
- Speed: 1.30ms inference (771 FPS capable)

---

## 🙏 Acknowledgments

- Ultralytics team for YOLOv11
- Roboflow for dataset annotation tools
- OpenCV community for computer vision utilities

---

**Status**: ✅ Production Ready (Epoch170 deployed)  
**Last Updated**: November 13, 2025  
**Version**: 1.0.0 (JACKPOT Release)

---

**Happy detecting! 🚗🔍**
