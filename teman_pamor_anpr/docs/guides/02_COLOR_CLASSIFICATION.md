# 🎨 Color Classification Workflow

## Overview

Pipeline 2-stage untuk deteksi plat nomor + klasifikasi warna:
1. **Plate Detector** (YOLOv11) - Detect bounding box plat
2. **Color Classifier** (MobileNetV2) - Classify warna plat (Putih/Hitam/Merah/Kuning)

## Quick Start

### 1. Label Images dengan Auto-Export 📸

```bash
python label_tool.py
```

**Features:**
- ✅ **Auto-export color crops** - Otomatis crop & save plate setiap labeling
- ✅ **Polygon & Box mode** - Untuk plat miring gunakan polygon
- ✅ **4 warna Indonesia** - Putih, Hitam, Merah, Kuning
- ✅ **Keyboard shortcuts** - [1-4] pilih warna, [P/N] navigate
- ✅ **Zoom + Pan** - Mouse wheel zoom, Right-click drag pan
- ✅ **Ctrl+Drag** - Move box/polygon untuk adjust

**Auto-Export Checkbox:**
- ☑️ **Enabled** (default): Setiap save, plate auto di-crop dan save ke `dataset/plate_colors/train/{color}/`
- ☐ **Disabled**: Hanya save labels, tidak export crops

**Output:**
```
dataset/
  plate_detection_color/train/    # Original images + labels
    images/
    labels/  # .txt (YOLO) + .json (color)
  
  plate_colors/                   # Auto-exported crops
    train/
      white/  # 80% untuk training
      black/
      red/
      yellow/
    val/
      white/  # 20% untuk validation
      black/
      red/
      yellow/
```

### 2. (Optional) Bulk Prepare dari Existing Dataset

Kalau sudah punya labeled dataset tapi belum di-crop:

```bash
python prepare_color_dataset.py
```

**Output:**
```
🎨 PREPARING COLOR CLASSIFICATION DATASET
======================================================================
📁 Source: dataset\plate_detection_color\train
📁 Output: dataset\plate_colors

🔍 Found 260 images
⚙️ Target size: 96x96px
⚙️ Minimum crop size: 50px

🔄 Processing images...
100%|█████████████████████████████| 260/260 [00:07<00:00, 36.51it/s]

✅ DATASET PREPARATION COMPLETE!
======================================================================

📊 Statistics:
   Total crops: 156
   ⚪ White: 25
   ⚫ Black: 103
   🔴 Red: 0
   🟡 Yellow: 28

💡 Recommendations:
   ⚠️ Need more white plates: 25/50 (add 25 more)
   ⚠️ Need more red plates: 0/50 (add 50 more)
   ⚠️ Need more yellow plates: 28/50 (add 22 more)
```

### 3. Train Color Classifier 🤖

Setelah punya minimal **200+ crops** (50+ per warna):

```bash
python train_color_classifier.py
```

**Configuration:**
- Model: MobileNetV2 (alpha=0.35) - Lightweight!
- Input: 96x96 RGB
- Classes: 4 (white, black, red, yellow)
- Augmentation: RandomFlip, Rotation, Zoom, Contrast, Brightness
- Epochs: 30 (with early stopping)
- Batch: 32

**Output:**
```
🎨 TRAINING COLOR CLASSIFIER
======================================================================
📊 Dataset Statistics:
   Train: 120 images
      ⚪ White: 20
      ⚫ Black: 80
      🔴 Red: 10
      🟡 Yellow: 22
   Val: 36 images

🤖 Building model...

🔥 Starting training...
Epoch 1/30: loss: 0.8234 - accuracy: 0.6583 - val_loss: 0.5234 - val_accuracy: 0.7778
Epoch 2/30: loss: 0.4567 - accuracy: 0.8250 - val_loss: 0.3456 - val_accuracy: 0.8889
...
Epoch 15/30: loss: 0.1234 - accuracy: 0.9583 - val_loss: 0.1567 - val_accuracy: 0.9444

✅ TRAINING COMPLETE!
======================================================================

📊 Final Evaluation:
   Train Accuracy: 95.83%
   Val Accuracy: 94.44%

💾 Model saved:
   Best: models/color_classifier/best_model.h5
   Final: models/color_classifier/final_model.h5
```

**Model Size:**
- Keras (.h5): ~5-6 MB
- TFLite (quantized): ~2 MB

### 4. Test Model 🧪

```bash
python test_color_classifier.py --image path/to/plate_crop.jpg
```

**Output:**
```
Prediction: yellow
Confidence: 0.9876
Probabilities:
  white: 0.0034
  black: 0.0012
  red: 0.0078
  yellow: 0.9876
```

### 5. Convert to TFLite (untuk Mobile) 📱

```bash
python convert_to_tflite.py
```

**Output:**
```
models/
  color_classifier/
    best_model.h5           # Keras model (~5 MB)
    color_classifier.tflite # TFLite model (~2 MB)
    color_classifier_int8.tflite # Quantized (~1 MB)
```

## Data Requirements

### Minimum (untuk test):
- **100 total crops** (25 per warna)
- Akurasi: ~80-85%
- Cukup untuk proof-of-concept

### Recommended (production):
- **400-800 total crops** (100-200 per warna)
- Akurasi: ~95%+
- Production-ready

### Ideal (best performance):
- **1000+ total crops** (250+ per warna)
- Akurasi: ~98%+
- Robust terhadap variasi lighting, angle, dll

## Tips Labeling

### 1. Balance Dataset
Usahakan jumlah setiap warna seimbang:
```
✅ Good:
  White: 100
  Black: 95
  Red: 80
  Yellow: 110
  
❌ Bad (Imbalanced):
  White: 200
  Black: 10
  Red: 5
  Yellow: 300
```

### 2. Variasi Data
Label berbagai kondisi:
- ✅ Lighting: Siang, malam, mendung
- ✅ Angle: Frontal, miring, jauh, dekat
- ✅ Kondisi: Bersih, kotor, rusak, tertutup sebagian
- ✅ Environment: Jalan raya, parkiran, indoor, outdoor

### 3. Kualitas Annotation
- ✅ **Use Polygon mode** untuk plat miring/distorsi
- ✅ **Zoom in** untuk precision
- ✅ **Pilih warna yang benar**:
  - Putih (white): Pribadi
  - Hitam (black): Pemerintah/TNI/Polri
  - Merah (red): Sementara/diplomatic
  - Kuning (yellow): Angkutan umum/komersial

## Pipeline Integration

### Full Detection Pipeline:

```python
from ultralytics import YOLO
from tensorflow import keras
import cv2

# Load models
detector = YOLO('models/plate_detector/best.pt')
classifier = keras.models.load_model('models/color_classifier/best_model.h5')

# Detect & classify
image = cv2.imread('image.jpg')

# Stage 1: Detect plate
results = detector(image)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
    # Crop plate
    plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
    plate_resized = cv2.resize(plate_crop, (96, 96))
    
    # Stage 2: Classify color
    plate_input = plate_resized / 255.0
    plate_input = np.expand_dims(plate_input, axis=0)
    
    color_probs = classifier.predict(plate_input)[0]
    color_idx = np.argmax(color_probs)
    colors = ['white', 'black', 'red', 'yellow']
    
    print(f"Detected: {colors[color_idx]} ({color_probs[color_idx]:.2%})")
```

## Troubleshooting

### Dataset kecil (<100 images)
**Problem:** Model overfit, accuracy rendah

**Solution:**
1. Label lebih banyak images (target 200+)
2. Use data augmentation (sudah di-implement)
3. Reduce model complexity (pakai alpha=0.25)

### Imbalanced classes
**Problem:** Model bias ke class mayoritas

**Solution:**
1. Balance dataset dengan label lebih banyak class minoritas
2. Use class weights saat training
3. Data augmentation lebih agresif untuk class minoritas

### Low accuracy pada warna tertentu
**Problem:** Putih/Kuning sulit dibedakan

**Solution:**
1. Label lebih banyak samples warna yang confusing
2. Add more lighting variations
3. Review annotation quality

### Auto-export tidak jalan
**Problem:** Checkbox enabled tapi tidak ada crops

**Solution:**
1. Check path `dataset/plate_colors/` exists
2. Check image size (minimal 50x50px)
3. Check JSON labels punya field 'color'

## File Structure

```
plate-recognition/
├── label_tool.py                   # Main labeling tool
├── prepare_color_dataset.py        # Bulk crop preparation
├── train_color_classifier.py       # Training script
├── test_color_classifier.py        # Testing script
├── convert_to_tflite.py           # TFLite conversion
│
├── dataset/
│   ├── plate_detection_color/     # Original images + labels
│   │   └── train/
│   │       ├── images/
│   │       └── labels/  # .txt + .json
│   │
│   └── plate_colors/              # Color dataset (crops)
│       ├── train/
│       │   ├── white/
│       │   ├── black/
│       │   ├── red/
│       │   └── yellow/
│       └── val/
│           ├── white/
│           ├── black/
│           ├── red/
│           └── yellow/
│
└── models/
    ├── plate_detector/            # YOLOv11 detector
    │   └── best.pt
    └── color_classifier/          # Color classifier
        ├── best_model.h5
        ├── final_model.h5
        └── color_classifier.tflite
```

## Performance Metrics

### Expected Performance (with 400+ balanced images):

**Training:**
- Train Accuracy: 95-98%
- Val Accuracy: 93-96%
- Loss: <0.15

**Inference:**
- Speed (CPU): ~10-30ms per crop
- Speed (GPU): ~1-5ms per crop
- Speed (Mobile TFLite): ~15-40ms per crop

**Full Pipeline (Detector + Color):**
- Desktop (GPU): ~60-100ms per image
- Mobile (TFLite): ~150-250ms per image
- Acceptable for real-time (<10 FPS)

## Next Steps

1. ✅ Label 200-400 images dengan color
2. ✅ Train color classifier
3. ✅ Convert to TFLite
4. 🔄 Integrate ke mobile app (Flutter)
5. 🔄 Add OCR (PaddleOCR/Tesseract)
6. 🔄 Deploy to production

## Support

Questions? Check:
- Main README.md
- GitHub Issues
- Code comments
