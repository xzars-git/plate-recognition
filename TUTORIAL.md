# 🚗 Complete ANPR Tutorial - Step by Step

Panduan lengkap untuk training dan testing sistem ANPR 2-stage.

## 📋 Prerequisites

1. ✅ Python 3.8+
2. ✅ GPU dengan CUDA (recommended) atau CPU
3. ✅ Dataset sudah siap di folder `dataset/`

## 🎯 Pipeline Overview

```
INPUT IMAGE
    ↓
[Stage 1: YOLOv11]
Detect plate location
    ↓
Crop plate region
    ↓
[Stage 2: PaddleOCR]
Read characters
    ↓
OUTPUT: Plate number text
```

---

## 🔧 Step 1: Setup Environment

### 1.1 Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 1.2 Install Dependencies

```powershell
# Install YOLOv11
pip install ultralytics

# Install OCR engine
pip install paddlepaddle paddleocr

# Install utilities
pip install opencv-python pandas
```

### 1.3 Verify Installation

```powershell
python -c "from ultralytics import YOLO; from paddleocr import PaddleOCR; print('✅ Setup complete!')"
```

---

## 📦 Step 2: Prepare Dataset

### 2.1 Konversi COCO ke YOLO Format

```powershell
python convert_coco_to_yolo.py
```

**Output:**

- `dataset/plate_detection_yolo/images/train/` - Training images
- `dataset/plate_detection_yolo/images/val/` - Validation images
- `dataset/plate_detection_yolo/labels/train/` - Training labels
- `dataset/plate_detection_yolo/labels/val/` - Validation labels
- `dataset/plate_detection_yolo/plate_detection.yaml` - Config file

### 2.2 Verify Dataset

Check jumlah images & labels:

```powershell
python -c "from pathlib import Path; print(f'Train images: {len(list(Path('dataset/plate_detection_yolo/images/train').glob('*')))}'); print(f'Val images: {len(list(Path('dataset/plate_detection_yolo/images/val').glob('*')))}')"
```

---

## 🎓 Step 3: Train Stage 1 - Plate Detection

### 3.1 Start Training

```powershell
python train_plate_detection.py
```

### 3.2 Monitor Training

Training progress akan ditampilkan di console. Perhatikan:

- **Loss**: Harus turun (target < 1.0)
- **mAP50**: Harus naik (target > 0.90)
- **mAP50-95**: Harus naik (target > 0.70)

### 3.3 Training Complete

Model terbaik akan disimpan di:

```
runs/plate_detection/yolov11_stage1/weights/best.pt
```

### 3.4 View Training Results

Training plots tersimpan di:

```
runs/plate_detection/yolov11_stage1/
  ├── results.png         # Training curves
  ├── confusion_matrix.png
  ├── F1_curve.png
  └── PR_curve.png
```

---

## 🧪 Step 4: Test ANPR System

### 4.1 Run Complete Pipeline

```powershell
python detect_anpr_complete.py
```

### 4.2 Input Test Image

Masukkan path gambar saat diminta:

```
Masukkan path gambar: dataset/plate_detection_dataset/images/Cars00081.png
```

### 4.3 View Results

System akan:

1. Detect plat nomor (Stage 1)
2. Read teks plat (Stage 2)
3. Display hasil di window
4. Save hasil ke `anpr_result.jpg`

**Contoh Output:**

```
✅ Detected 1 plate(s):
   1. H2169QB (confidence: 0.95)

💾 Result saved: anpr_result.jpg
```

---

## 📊 Step 5: Evaluate Performance

### 5.1 Test pada Multiple Images

Buat script batch testing:

```python
from detect_anpr_complete import ANPRSystem
from pathlib import Path

anpr = ANPRSystem('runs/plate_detection/yolov11_stage1/weights/best.pt')

test_images = Path('dataset/plate_detection_dataset/images').glob('*.png')

for img in test_images:
    result = anpr.process(str(img), visualize=False)
    if result['plates']:
        print(f"{img.name}: {result['plates'][0]['plate_number']}")
```

### 5.2 Calculate Metrics

```python
correct = 0
total = 0
# Compare dengan ground truth dari dataset
```

---

## 🚀 Step 6: Optimization (Optional)

### 6.1 Improve Stage 1 (Plate Detection)

Jika akurasi kurang:

1. **Tambah epochs:**
   Edit `train_plate_detection.py`, ubah `epochs=150` → `epochs=200`

2. **Gunakan model lebih besar:**
   Ubah `yolo11m.pt` → `yolo11l.pt`

3. **Adjust augmentation:**
   Kurangi augmentasi jika overfitting

### 6.2 Improve Stage 2 (OCR)

Jika pembacaan teks kurang akurat:

1. **Preprocessing image:**

   ```python
   # Tambah sebelum OCR
   plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
   plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   ```

2. **Gunakan EasyOCR sebagai alternatif:**
   ```python
   import easyocr
   reader = easyocr.Reader(['en'])
   result = reader.readtext(plate_img)
   ```

---

## 💾 Step 7: Deployment

### 7.1 Export Model

```python
from ultralytics import YOLO

model = YOLO('runs/plate_detection/yolov11_stage1/weights/best.pt')

# ONNX (cross-platform)
model.export(format='onnx')

# TensorRT (NVIDIA GPU)
model.export(format='engine')

# TFLite (mobile/edge)
model.export(format='tflite')
```

### 7.2 Create Standalone Script

Gabungkan semua dalam 1 file untuk deployment

---

## 🐛 Troubleshooting

### ❌ "CUDA out of memory"

**Solution:**

- Kurangi batch size di `train_plate_detection.py`
- Gunakan model lebih kecil (yolo11s.pt atau yolo11n.pt)

### ❌ "No plate detected"

**Solution:**

- Turunkan confidence threshold: `conf=0.25` → `conf=0.15`
- Check kualitas gambar input
- Retrain dengan augmentasi lebih sedikit

### ❌ "OCR text tidak akurat"

**Solution:**

- Preprocessing gambar (grayscale, threshold)
- Gunakan EasyOCR sebagai alternatif
- Crop dengan margin lebih besar

---

## 📈 Expected Results

### Stage 1 (Plate Detection)

- **mAP50**: 0.95+ (sangat baik)
- **mAP50-95**: 0.75+ (bagus)
- **Speed**: 50-100 FPS (GPU)

### Stage 2 (OCR)

- **Accuracy**: 90%+ untuk plat yang jelas
- **Speed**: 10-20 FPS

### Overall ANPR

- **End-to-end accuracy**: 85-90%
- **Processing time**: 100-200ms per image

---

## 🎓 Next Steps

1. ✅ Fine-tune hyperparameters
2. ✅ Collect more training data
3. ✅ Test pada real-world scenarios
4. ✅ Deploy to production (API/Web app)
5. ✅ Add video processing capability

---

**Happy Training! 🚀**
