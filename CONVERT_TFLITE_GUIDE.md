# 🚀 Guide: Convert Model ke TFLite di Google Colab

## 📝 Persiapan

**Yang Dibutuhkan:**
- ✅ File `best.pt` (16.08 MB) - model epoch170 
- ✅ File `tflite_conversion_colab.ipynb` - notebook conversion
- ✅ Akun Google (untuk akses Colab)
- ✅ Koneksi internet stabil

---

## 🔧 Step-by-Step Guide

### **Step 1: Buka Google Colab**

1. Buka browser, ke: https://colab.research.google.com
2. Login dengan akun Google kamu
3. Klik **File → Upload notebook**
4. Pilih file `tflite_conversion_colab.ipynb` dari project folder
5. Tunggu notebook terbuka

**Screenshot lokasi:** Menu File → Upload notebook

---

### **Step 2: Jalankan Cell Install Dependencies**

1. Klik cell pertama yang ada kode `!pip install`
2. Tekan **Shift + Enter** atau klik tombol ▶️ (Play)
3. Tunggu instalasi selesai (~1-2 menit)

**Output yang diharapkan:**
```
✅ All dependencies installed!
```

**Packages yang diinstall:**
- `ultralytics` - YOLO framework
- `tensorflow==2.15.0` - TFLite converter
- `onnx>=1.12.0` - ONNX format support
- `onnxsim>=0.4.1` - ONNX simplification
- `onnxruntime>=1.16.0` - ONNX inference

---

### **Step 3: Upload Model File**

1. Jalankan cell kedua (Upload Model)
2. Klik tombol **"Choose Files"** yang muncul
3. Pilih file `best.pt` dari folder project kamu
4. Tunggu upload selesai

**Progress upload:**
```
Saving best.pt to best.pt
100%
✅ best.pt uploaded successfully!
   Size: 16.08 MB
```

**Tips:**
- Pastikan koneksi internet stabil
- Upload bisa 30 detik - 2 menit tergantung kecepatan internet
- Jika gagal, refresh page dan ulangi dari Step 2

---

### **Step 4: Verify Model**

Jalankan cell ketiga untuk cek model:

**Output:**
```
📥 Loading model...
✅ Model loaded successfully!

📊 Model Information:
Model summary: 225 layers, 2,592,784 parameters, 0 gradients, 6.4 GFLOPs

🎯 Expected Performance:
   Precision: 81.64%
   mAP50: 49.14%
   Speed: 1.30ms (PyTorch on RTX 3080 Ti)
```

**Jika ada error:**
- Pastikan file `best.pt` terupload dengan benar
- Size harus 16.08 MB
- Coba upload ulang

---

### **Step 5: Export ke ONNX**

Jalankan cell keempat:

**Proses:**
```
🔄 Exporting to ONNX format...
Ultralytics YOLOv8.0.196 🚀 Python-3.10.12 torch-2.1.0+cu121 CPU (Intel Xeon 2.20GHz)

YOLOv11n summary (fused): 238 layers, 2,592,784 parameters, 0 gradients, 6.4 GFLOPs
✅ ONNX export complete!
   File: best.onnx
   Size: 10.71 MB
   Export time: 3.2s
```

**Format ONNX:**
- Ukuran: ~10.71 MB
- Platform: Desktop/Server
- Akurasi: 81.64% (sama dengan PyTorch)
- Speed: 20-50ms (mobile/server)

---

### **Step 6: Export ke TFLite FP16**

Jalankan cell kelima (FP16 Quantization):

**Proses:**
```
📱 Exporting to TFLite (FP16)...
✅ TFLite FP16 export complete!
   File: best_fp16.tflite
   Size: 10.23 MB
   Export time: 4.5s

📊 Quantization: FP16 (half precision)
   Expected performance: ~81% accuracy, 20-30ms mobile inference
```

**FP16 Details:**
- Ukuran: ~10 MB
- Quantization: Half precision (16-bit float)
- Akurasi: ~81% (minimal loss)
- Speed: 20-30ms (mobile)
- Platform: Android/iOS

---

### **Step 7: Export ke TFLite INT8 (FASTEST!)**

Jalankan cell keenam (INT8 Quantization):

**Proses:**
```
🚀 Exporting to TFLite (INT8)...
⚠️  This will take longer (~2-5 minutes)

Calibrating using coco128 dataset...
✅ TFLite INT8 export complete!
   File: best_int8.tflite
   Size: 5.12 MB
   Export time: 187.3s

📊 Quantization: INT8 (integer precision)
   Expected performance: ~80% accuracy, 10-20ms mobile inference
   Size reduction: 50.0% smaller than FP16
```

**INT8 Details:**
- Ukuran: ~5 MB (PALING KECIL!)
- Quantization: Integer 8-bit
- Akurasi: ~80% (acceptable loss)
- Speed: 10-20ms (PALING CEPAT!)
- Platform: Android/iOS optimized

**⚡ REKOMENDASI: Gunakan INT8 untuk mobile deployment!**

---

### **Step 8: Lihat Summary**

Jalankan cell ketujuh untuk comparison table:

```
================================================================================
📊 EXPORT SUMMARY
================================================================================
      Format            Filename  Size (MB)          Platform Est. Speed Accuracy
     PyTorch             best.pt      16.08           Desktop     1.30ms   81.64%
        ONNX          best.onnx      10.71    Mobile/Server    20-50ms   81.64%
  TFLite FP16   best_fp16.tflite      10.23            Mobile    20-30ms     ~81%
  TFLite INT8   best_int8.tflite       5.12  Mobile (Optimized)    10-20ms     ~80%
================================================================================

🎯 Recommendations:
   • Use ONNX for server/desktop deployment (best accuracy)
   • Use TFLite FP16 for mobile (good balance)
   • Use TFLite INT8 for mobile (fastest, smallest)
```

---

### **Step 9: Download Model Files**

Jalankan cell kedelapan untuk download semua model:

**Auto-download 3 files:**
1. ✅ `best.onnx` (10.71 MB)
2. ✅ `best_fp16.tflite` (10.23 MB)  
3. ✅ `best_int8.tflite` (5.12 MB) ⚡ **GUNAKAN INI!**

**File akan otomatis download ke folder Downloads laptop kamu**

**Jika download gagal:**
- Klik kanan di file explorer Colab → Download
- Atau jalankan ulang cell tersebut

---

### **Step 10 (Optional): Test Inference**

Jalankan cell kesembilan untuk test model:

1. Upload test image (foto plat nomor)
2. Model akan run detection
3. Hasil akan ditampilkan dengan bounding box

**Output contoh:**
```
🧪 Testing inference on sample image...
🔍 Running inference on test.jpg...
✅ Detection complete!
   Plates detected: 2
   Plate 1: 87.34% confidence
   Plate 2: 79.12% confidence
```

Gambar hasil detection akan ditampilkan langsung di notebook!

---

## 📱 Integrasi ke Flutter

### **Pilih Format Model:**

**Untuk Teman Pamor App:**
```
✅ Gunakan: best_int8.tflite
   - Ukuran: 5 MB (paling kecil)
   - Speed: 10-20ms (paling cepat)
   - Akurasi: ~80% (cukup untuk production)
```

### **Install Flutter Package:**

Edit `pubspec.yaml`:
```yaml
dependencies:
  tflite_flutter: ^0.10.0
  image: ^4.0.0
```

Run:
```bash
flutter pub get
```

### **Copy Model File:**

```bash
# Buat folder assets
mkdir -p assets/models

# Copy model
cp best_int8.tflite assets/models/

# Edit pubspec.yaml, tambahkan:
flutter:
  assets:
    - assets/models/best_int8.tflite
```

### **Load Model di Flutter:**

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class PlateDetector {
  late Interpreter interpreter;
  
  Future<void> loadModel() async {
    // Load TFLite model
    interpreter = await Interpreter.fromAsset(
      'assets/models/best_int8.tflite'
    );
    
    print('✅ Model loaded!');
    print('Input shape: ${interpreter.getInputTensors()}');
    print('Output shape: ${interpreter.getOutputTensors()}');
  }
  
  Future<List<Detection>> detectPlate(Uint8List imageBytes) async {
    // 1. Preprocess: resize to 640x640
    var input = preprocessImage(imageBytes);
    
    // 2. Run inference
    var output = List.filled(1 * 5 * 8400, 0.0).reshape([1, 5, 8400]);
    interpreter.run(input, output);
    
    // 3. Post-process: parse detections
    return parseYOLOOutput(output);
  }
  
  List<List<List<double>>> preprocessImage(Uint8List bytes) {
    // Resize to 640x640
    // Normalize to 0-1
    // Convert to [1, 3, 640, 640] tensor
    // ... implementation
  }
  
  List<Detection> parseYOLOOutput(List output) {
    // Parse YOLO v11 output format
    // Filter by confidence threshold (0.25)
    // Apply NMS (Non-Maximum Suppression)
    // ... implementation
  }
}
```

### **Integrasi dengan Rotation Detection:**

```dart
class RotationDetector {
  // Port dari plate_rotation_detector.py
  int detectRotation(Image image) {
    // 1. Edge detection (Canny)
    // 2. Find contours
    // 3. Calculate aspect ratio
    // 4. Determine rotation angle (0°/90°/180°/270°)
    // ... implementation
  }
  
  Image correctRotation(Image image, int angle) {
    // Rotate image by detected angle
    // ... implementation
  }
}

// Pipeline lengkap:
Future<List<Detection>> detectPlateWithRotation(Image image) async {
  // 1. Detect rotation
  final angle = rotationDetector.detectRotation(image);
  
  // 2. Correct rotation
  final corrected = rotationDetector.correctRotation(image, angle);
  
  // 3. Run YOLO detection
  final detections = await plateDetector.detectPlate(corrected);
  
  return detections;
}
```

---

## 🎯 Performance Tips

### **Optimasi Mobile Inference:**

1. **Enable GPU Delegate:**
```dart
final options = InterpreterOptions()
  ..useNnApiForAndroid = true  // Android Neural Networks API
  ..addDelegate(GpuDelegateV2());  // GPU acceleration
  
interpreter = await Interpreter.fromAsset(
  'assets/models/best_int8.tflite',
  options: options
);
```

2. **Pre-allocate Buffers:**
```dart
// Allocate tensor buffers once at startup
await interpreter.allocateTensors();
```

3. **Batch Processing:**
```dart
// Process multiple frames in batch
var batchInput = List.filled(batchSize * 3 * 640 * 640, 0.0);
var batchOutput = List.filled(batchSize * 5 * 8400, 0.0);
```

4. **Cache Preprocessing:**
```dart
// Reuse resize buffers
final resizedBuffer = Uint8List(640 * 640 * 3);
```

### **Expected Performance:**

**Samsung Galaxy S21:**
- Preprocessing: 5-8ms
- Inference (INT8): 10-15ms
- Post-processing: 3-5ms
- **Total: ~20-30ms (30-50 FPS)**

**iPhone 13:**
- Preprocessing: 3-5ms
- Inference (INT8): 8-12ms
- Post-processing: 2-4ms
- **Total: ~15-25ms (40-60 FPS)**

---

## ⚠️ Troubleshooting

### **Problem 1: Upload Stuck**
**Gejala:** Upload best.pt tidak selesai-selesai

**Solusi:**
1. Refresh page Colab
2. Ulangi dari Step 2 (install dependencies)
3. Pastikan koneksi internet stabil
4. Coba compress file dulu (zip) sebelum upload

---

### **Problem 2: ONNX Export Error**
**Gejala:** 
```
RuntimeError: Exporting the operator 'aten::upsample_nearest2d' 
to ONNX opset version 12 is not supported
```

**Solusi:**
```python
# Gunakan opset 11 atau 13
model.export(format='onnx', opset=11)
```

---

### **Problem 3: TFLite INT8 Export Lambat**
**Gejala:** Stuck di "Calibrating..." lebih dari 10 menit

**Solusi:**
1. **Normal behavior** - INT8 calibration memang lama (2-5 menit)
2. Pastikan tidak ada error di cell sebelumnya
3. Jika lebih dari 10 menit, restart runtime:
   - Runtime → Restart runtime
   - Ulangi dari Step 2

---

### **Problem 4: Download Error**
**Gejala:** File tidak ke-download otomatis

**Solusi:**
1. Buka file explorer di Colab (folder icon di kiri)
2. Klik kanan pada file → Download
3. Atau gunakan manual download:
```python
from google.colab import files
files.download('best_int8.tflite')
```

---

### **Problem 5: Model Size Terlalu Besar**
**Gejala:** best_int8.tflite masih >10 MB

**Solusi:**
- INT8 seharusnya ~5 MB
- Jika masih besar, coba export ulang dengan force INT8:
```python
model.export(
    format='tflite',
    int8=True,
    data='coco128.yaml',
    quantize=True  # Force full quantization
)
```

---

## 📞 Support

**Jika masih ada masalah:**
1. Cek error message di Colab
2. Screenshot error dan bagian mana yang stuck
3. Verifikasi file `best.pt` tidak corrupt (size harus 16.08 MB)

**Resources:**
- Ultralytics Docs: https://docs.ultralytics.com/modes/export/
- TFLite Docs: https://www.tensorflow.org/lite/guide
- Flutter TFLite: https://pub.dev/packages/tflite_flutter

---

## ✅ Checklist Conversion

- [ ] Google Colab dibuka
- [ ] Notebook `tflite_conversion_colab.ipynb` uploaded
- [ ] Dependencies installed (Cell 1)
- [ ] Model `best.pt` uploaded (Cell 2)
- [ ] Model verified (Cell 3)
- [ ] ONNX exported (Cell 4)
- [ ] TFLite FP16 exported (Cell 5)
- [ ] TFLite INT8 exported (Cell 6)
- [ ] Summary checked (Cell 7)
- [ ] All files downloaded (Cell 8)
- [ ] Model di-copy ke Flutter project
- [ ] pubspec.yaml updated
- [ ] Model loaded di Flutter
- [ ] Test inference works

**Selamat! Model siap di-deploy ke Teman Pamor! 🚀**
