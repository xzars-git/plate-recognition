# 🚗 ANPR System - Automatic Number Plate Recognition

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![YOLOv11](https://img.shields.io/badge/YOLOv11-latest-orange)

Sistem **Automatic Number Plate Recognition (ANPR)** dengan fitur **automatic rotation correction** untuk mendeteksi plat nomor yang dirotasi (90°/180°/270°).

## ✨ Features

- ✅ **YOLOv11 Plate Detection** - Deteksi lokasi plat nomor (mAP50 = 87.47%)
- ✅ **PaddleOCR Text Recognition** - Pembacaan teks plat nomor
- ✅ **Rotation Auto-Correction** - Handle plat yang dirotasi 90°/180°/270° ⭐ NEW!
- ✅ **Real-time Webcam** - Support webcam dengan 18 FPS (CPU) / 60+ FPS (GPU)
- ✅ **Production Ready** - Tested dan documented

---

## 🎯 Pipeline

1. **Preprocessing**: Auto-detect dan koreksi rotasi gambar ⭐ NEW!
2. **Detection**: Deteksi lokasi plat nomor (YOLOv11)
3. **Recognition**: Pembacaan teks plat (PaddleOCR)

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Buat virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2. Test Detection (Normal)

```powershell
python test_images.py --source path/to/image.jpg
```

### 3. Test Detection (With Rotation Auto-Correction) ⭐ NEW!

```powershell
python test_images_with_rotation.py --source path/to/image.jpg --debug
```

### 4. Test Rotation Correction Only

```powershell
# Single image
python plate_rotation_detector.py image.jpg --debug

# Batch process folder
python plate_rotation_detector.py folder/ --folder --output corrected/
```

### 5. Webcam Real-time

```powershell
python fast_webcam_anpr.py
```

**Controls**: `O` - Toggle OCR | `S` - Screenshot | `Q` - Quit

---

## 📊 Performance

| Metric      | Value      |
| ----------- | ---------- |
| **mAP50**   | **87.47%** |
| Precision   | 77.47%     |
| Recall      | 87.13%     |
| Speed (CPU) | 18 FPS     |
| Speed (GPU) | 60+ FPS    |

**Training**: 150 epochs (~6-7 hours CPU / ~30 mins GPU)  
**Dataset**: 1,099 train + 276 val images

## 📁 Project Structure

```
plate-recognition/
├── best.pt                           # Trained YOLOv11 model
├── plate_rotation_detector.py        # Rotation detection & correction
├── test_images_with_rotation.py      # Inference with rotation (MAIN)
├── test_images.py                    # Simple inference (no rotation)
├── fast_webcam_anpr.py               # Real-time webcam detection
├── train_plate_detection.py          # Training script
├── plat_jabar.yaml                   # Dataset configuration
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation
└── dataset/
    └── plate_detection_yolo/         # YOLO format dataset
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
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

### 1. Test on Images (With Rotation Correction)

```powershell
# Test single image
python test_images_with_rotation.py --source test_image.jpg

# Test folder
python test_images_with_rotation.py --source test_folder/

# With debug info
python test_images_with_rotation.py --source image.jpg --debug
```

### 2. Test on Images (Simple, No Rotation)

```powershell
python test_images.py --source test_image.jpg
```

### 3. Rotation Correction Only

```powershell
# Single image
python plate_rotation_detector.py image.jpg --debug

# Batch process folder
python plate_rotation_detector.py input_folder/ --folder --output corrected/
```

### 4. Real-time Webcam

```powershell
python fast_webcam_anpr.py
```

**Webcam Controls:**
- `O` - Toggle OCR
- `S` - Save screenshot
- `Q` - Quit

### 5. Training (Optional)

```powershell
# Train new model or fine-tune
python train_plate_detection.py
```

## 🎨 Programmatic Usage

### Basic Usage

```python
from ultralytics import YOLO
from plate_rotation_detector import PlateRotationDetector
import cv2

# Load model and detector
model = YOLO('best.pt')
detector = PlateRotationDetector()

# Read image
image = cv2.imread('test.jpg')

# Step 1: Correct rotation
corrected, angle, confidence = detector.preprocess(image)
print(f"Rotation: {angle}° (confidence: {confidence:.2f})")

# Step 2: Detect plates
results = model.predict(corrected, conf=0.25)

# Step 3: Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        print(f"Plate at ({x1}, {y1}, {x2}, {y2}) - confidence: {conf:.2f}")
```

### Batch Processing

```python
from pathlib import Path

# Process all images in folder
image_folder = Path('test_images/')
for img_path in image_folder.glob('*.jpg'):
    # Correct rotation
    image = cv2.imread(str(img_path))
    corrected, angle, conf = detector.preprocess(image)
    
    # Detect
    results = model.predict(corrected, conf=0.25)
    
    # Save result
    if len(results[0].boxes) > 0:
        annotated = results[0].plot()
        cv2.imwrite(f'output/{img_path.name}', annotated)
```

## 🐛 Troubleshooting

### GPU tidak terdeteksi

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Memory Error saat training

- Kurangi `batch` size
- Gunakan model lebih kecil (yolo11n.pt)
- Kurangi `imgsz` (dari 640 ke 512)

### Akurasi rendah

- Tambah epochs
- Gunakan model lebih besar (yolo11m.pt atau yolo11l.pt)
- Cek kualitas label dataset
- Tambah data training

## 📚 Resources

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv11 Paper](https://arxiv.org/abs/2310.16764)

## 📝 License

MIT License

## 👨‍💻 Author

Project deteksi plat nomor kendaraan Jawa Barat menggunakan YOLOv11.

---

**Selamat mencoba! 🚗🔍**
