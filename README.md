# 🚗 ANPR System - 2-Stage Detection

Project **Automatic Number Plate Recognition (ANPR)** menggunakan **YOLOv11** + **PaddleOCR** untuk deteksi plat nomor kendaraan Jawa Barat.

## 🎯 2-Stage Pipeline:

1. **Stage 1**: Deteksi lokasi plat nomor (YOLOv11)
2. **Stage 2**: Pembacaan karakter/teks plat (PaddleOCR)

## 🚀 Fitur YOLOv11

YOLOv11 merupakan model terbaru (Oktober 2024) dengan keunggulan:

- ✅ Akurasi lebih tinggi dari YOLOv8/v9/v10
- ✅ Kecepatan inferensi lebih cepat
- ✅ Model lebih efisien (parameter lebih sedikit)
- ✅ Support untuk berbagai task (detection, segmentation, classification, pose, OBB)

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

```powershell
python train_plate_detection.py
```

**Model yang digunakan:**

- `yolo11n.pt` - Nano (tercepat)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium (RECOMMENDED)
- `yolo11l.pt` - Large
- `yolo11x.pt` - XLarge (akurasi tertinggi)

Model terbaik akan disimpan di: `runs/plate_detection/yolov11_stage1/weights/best.pt`

### Step 3: Testing Complete ANPR System

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
