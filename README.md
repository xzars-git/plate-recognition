# 🚗 ANPR Jawa Barat - Automatic Number Plate Recognition

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue)
![YOLOv11](https://img.shields.io/badge/YOLOv11-latest-orange)

Sistem **Automatic Number Plate Recognition (ANPR)** lengkap untuk plat nomor kendaraan Jawa Barat menggunakan **YOLOv11** + **PaddleOCR**.

## ✅ Project Status: COMPLETE!

- ✅ **Stage 1**: YOLOv11 plate detection (mAP50 = 87.47%)
- ✅ **Stage 2**: PaddleOCR text recognition
- ✅ **Real-time**: Webcam support (18 FPS)
- ✅ **Production Ready**: Tested and documented

---

## 🎯 2-Stage Pipeline:

1. **Stage 1**: Deteksi lokasi plat nomor (YOLOv11) → **TRAINED ✅**
2. **Stage 2**: Pembacaan karakter/teks plat (PaddleOCR) → **INTEGRATED ✅**

---

## 🚀 Quick Start

### Run Webcam (Real-time Detection)

```bash
python test_webcam.py
```

**Controls**: `'o'` toggle OCR | `'s'` screenshot | `'q'` quit

### Test on Images

```bash
python test_images.py --source path/to/images
```

### Complete ANPR Pipeline

```bash
python test_complete_anpr.py
```

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

## 📁 Struktur Dataset

```
anpr-jabar/
├── dataset/
│   ├── plate_detection_dataset/      # Stage 1: Plate Detection
│   │   ├── annotations/
│   │   │   └── annotations.json      # COCO format
│   │   └── images/
│   ├── plate_text_dataset/           # Stage 2: Character Recognition
│   │   ├── label.csv                 # Text labels
│   │   └── dataset/                  # Cropped plate images
│   └── plate_detection_yolo/         # Converted YOLO format (auto-generated)
├── convert_coco_to_yolo.py           # Konversi COCO → YOLO
├── train_plate_detection.py          # Training Stage 1
├── train_char_recognition.py         # Info Stage 2
├── detect_anpr_complete.py           # Complete ANPR pipeline
└── README.md
```

## 🔧 Instalasi

### 1. Install Python (3.8 atau lebih baru)

### 2. Activate Virtual Environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install ultralytics opencv-python paddlepaddle paddleocr
```

Atau install semua sekaligus:

```powershell
pip install -r requirements.txt
```

### 4. Cek Instalasi

```powershell
python -c "from ultralytics import YOLO; from paddleocr import PaddleOCR; print('✅ All packages installed!')"
```

## 🎯 Cara Menggunakan

### Step 1: Konversi Dataset COCO ke YOLO

Dataset plate detection dalam format COCO, perlu dikonversi ke YOLO:

```powershell
python convert_coco_to_yolo.py
```

Output: `dataset/plate_detection_yolo/` dengan struktur YOLO format

### Step 2: Training Stage 1 - Plate Detection

Train model YOLOv11 untuk mendeteksi lokasi plat nomor:

**For CPU (Windows stability):**

```powershell
python train_plate_detection.py
```

**For GPU (RTX 3080 Ti optimized):**

```powershell
# Install PyTorch CUDA first (one-time)
python install_pytorch_cuda.py

# Then train
python train_plate_detection_gpu.py
```

**Model yang digunakan:**

- `yolo11n.pt` - Nano (tercepat)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium (RECOMMENDED)
- `yolo11l.pt` - Large
- `yolo11x.pt` - XLarge (akurasi tertinggi)

Model terbaik akan disimpan di: `runs/plate_detection/yolov11_stage1/weights/best.pt`

### Step 3: Evaluate Model Performance

Check mAP, precision, recall setelah training:

```powershell
python evaluate_model.py
```

Target: mAP50 >= 0.85

### Step 4: Test with Images

Test model pada gambar:

```powershell
# Test on validation set
python test_images.py

# Test on specific image/folder
python test_images.py --source path/to/image.jpg
python test_images.py --source path/to/folder/
```

### Step 5: Test with Webcam

Real-time detection dengan webcam:

```powershell
python test_webcam.py
```

**Keyboard controls:**

- `Q` - Quit
- `S` - Save screenshot
- `O` - Toggle OCR (if PaddleOCR installed)

### Step 6: Complete ANPR Pipeline (Stage 1 + 2)

Gunakan complete pipeline (Stage 1 + Stage 2):

```powershell
python detect_anpr_complete.py
```

System akan:

1. ✅ Detect lokasi plat nomor dengan YOLOv11
2. ✅ Crop region plat nomor
3. ✅ Read teks dengan PaddleOCR
4. ✅ Tampilkan hasil dengan bounding box + teks

## 📊 Monitoring Training

Ultralytics menyediakan monitoring otomatis:

1. **TensorBoard** (lokal):

```bash
tensorboard --logdir runs/detect
```

2. **Weights & Biases** (cloud) - Tambahkan di `train_yolov11.py`:

```python
results = model.train(
    ...
    project='wandb',  # Enable W&B logging
)
```

3. **Comet ML** - Tambahkan di `train_yolov11.py`:

```python
results = model.train(
    ...
    project='comet',  # Enable Comet logging
)
```

## 🎨 Contoh Penggunaan Programmatic

### Training

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(data='plat_jabar.yaml', epochs=100, imgsz=640)
```

### Inference

```python
from ultralytics import YOLO

model = YOLO('runs/detect/plat_jabar_yolov11/weights/best.pt')
results = model('path/to/image.jpg')

# Akses hasil
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, BBox: {box.xyxy}")
```

### Batch Prediction

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('runs/detect/plat_jabar_yolov11/weights/best.pt')

# Proses semua gambar di folder
image_folder = Path('test_images/')
for img_path in image_folder.glob('*.jpg'):
    results = model(img_path, save=True)
```

## ⚙️ Hyperparameter Tuning

Untuk hasil optimal, gunakan hyperparameter tuning:

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.tune(data='plat_jabar.yaml', epochs=30, iterations=300)
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
