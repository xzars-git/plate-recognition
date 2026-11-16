# Color Classifier

Klasifikasi warna plat nomor (MobileNetV2 α=0.35, 96×96) -> export `.tflite`.

## Status

v0.6.0 (staging), varian di `03_deployment_platform/mobile_models/color_classifier/` (hanya label + confusion matrix; `.tflite` tidak di-commit):

| Varian | Kelas | Akurasi val | Catatan |
|---|---|---|---|
| `v0.6.0_4class` | white, black, yellow, **red** | 96,95% (n=394) | Rekomendasi. Recall: white 97,5%, black 98,6%, yellow 91,7%, red 88,5%. |
| `v0.6.0_3class` | white, black, yellow | - | Tanpa kelas merah. |

v0.6.0 dilatih dengan augmentasi padding acak dan "foto malam" dari plat merah asli, jadi tahan crop longgar
(padding ~0.15: 95,9% vs 75,3% model lama). **Keterbatasan merah:** plat merah gelap/maroon (malam hari) masih
kadang terbaca `black` (data sendiri 4-5/7). Untuk menaikkan recall merah dibutuhkan lebih banyak foto plat merah asli.

Detail: `00_platform/model_registry/registry.json`.

## Dataset eksternal & atribusi

Crop putih/hitam/kuning/merah tambahan diambil dari Roboflow Universe, "Plat Nomor" oleh JMTO (v22),
lisensi CC BY 4.0: https://universe.roboflow.com/jmto-9jtua/plat-nomor-szpuc
(`data.yaml` hasil export tertulis `license: Private`, sedangkan halaman proyek menulis CC BY 4.0; cek ulang sebelum rilis publik).

```bash
# setelah langkah 2 di bawah; --src = folder hasil extract Roboflow (format YOLOv8)
python teman_pamor_anpr/01_data_platform/labeling_tools/prepare_external_dataset.py \
  --src "dataset/plate_colors/Plat Nomor.v22i.yolov8" --synth-red 400 --real-red-to-train 6
python .../src/train.py --classes white,black,yellow,red --output models/color_classifier_4c
```

## Train -> Export (jalankan dari root repo)

```bash
# 1. Siapkan dataset berlabel di dataset/plate_detection_color (tidak ada di git, minta salinan dari tim).
#    Contoh 50 crop per warna ada di samples/.

# 2. Crop plat per warna -> dataset/plate_colors/{train,val}/<warna>/
python teman_pamor_anpr/01_data_platform/labeling_tools/prepare_color_dataset.py

# 3. Train -> models/color_classifier/best_model.keras (+ labels.txt, confusion_matrix.txt)
python teman_pamor_anpr/02_training_platform/02_color_classifier/src/train.py

# 4. Export + validasi gate -> models/color_classifier/color_classifier.tflite
python teman_pamor_anpr/03_deployment_platform/conversion/convert_to_tflite.py
```

Gate di langkah 4 (exit code 1 kalau gagal): akurasi TFLite turun < 2% dari Keras, prediksi sama >= 99%, ukuran < 10 MB.
Tambah `--mode float16` untuk model lebih kecil.

Tambah kelas merah setelah punya 50+ crop di `dataset/plate_colors/{train,val}/red/`:

```bash
python .../src/train.py --classes white,black,yellow,red
```

Urutan `--classes` = urutan output model = isi `labels.txt`.

## Kontrak untuk aplikasi (Panah Pasopati)

| | |
|---|---|
| Input | `[1, 96, 96, 3]` float32, **RGB**, piksel **0-255** (jangan dinormalisasi, sudah di dalam model) |
| Output | `[1, N]` float32, probabilitas softmax, urutan = `labels.txt` (`white, black, yellow`) |
| Crop | Dari bbox detector. **Padding sampai ~0.15 aman** (v0.6.0 dilatih dengan augmentasi padding acak: acc 95,9% dengan padding vs 75,3% pada model lama). Tanpa padding tetap paling akurat. |
| Resize | Langsung ke 96×96 (aspek plat di-squash, sama seperti saat training) |

Contoh uji cepat (Python):

```python
interp = tf.lite.Interpreter(model_path="color_classifier.tflite"); interp.allocate_tensors()
x = np.asarray(Image.open("crop.jpg").convert("RGB").resize((96, 96)), dtype=np.float32)[None]
interp.set_tensor(interp.get_input_details()[0]["index"], x); interp.invoke()
probs = interp.get_tensor(interp.get_output_details()[0]["index"])[0]
```

## Rilis

Salin hasil ke `03_deployment_platform/mobile_models/color_classifier/vX.Y.Z/` dan update `registry.json`.
Catatan: `*.tflite` ada di `.gitignore`, jadi file model tidak ikut ter-commit kecuali di-force-add (`git add -f`).
