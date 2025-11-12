# 🚀 QUICK START GUIDE

## Installation (5 minutes)

```powershell
# 1. Activate virtual environment (if using)
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 🖼️ Test on Images (RECOMMENDED)

```powershell
# With rotation correction (handles rotated plates)
python test_images_with_rotation.py --source test.jpg

# Simple test (no rotation)
python test_images.py --source test.jpg
```

### 🎥 Real-time Webcam

```powershell
python fast_webcam_anpr.py
```

Controls: `O`=OCR | `S`=Screenshot | `Q`=Quit

### 🔄 Rotation Correction Only

```powershell
# Single image
python plate_rotation_detector.py image.jpg --debug

# Batch folder
python plate_rotation_detector.py folder/ --folder
```

### 🏋️ Training (Optional)

```powershell
python train_plate_detection.py
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `best.pt` | Trained model |
| `plate_rotation_detector.py` | Rotation correction |
| `test_images_with_rotation.py` | Main inference (with rotation) |
| `test_images.py` | Simple inference |
| `fast_webcam_anpr.py` | Webcam detection |
| `train_plate_detection.py` | Training script |

---

## Need Help?

Read full documentation: `README.md`
