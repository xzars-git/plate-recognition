# OCR Custom

OCR plat nomor Indonesia: CNN penuh-konvolusi + CTC (1,13 juta parameter), ekspor `.tflite`.
Detail keputusan, data, dan angka lengkap: [`docs/OCR_CUSTOM_BLUEPRINT.md`](../../docs/OCR_CUSTOM_BLUEPRINT.md).

## Status

v2 (staging). Gerbang 1 dan 2 lulus; gerbang 3 (kecepatan < 50 ms di HP kelas bawah) belum diukur.

| | Benar persis (494 plat uji terkunci) |
|---|---|
| OCR kustom v2 | **76,3%** (HP 74,1%, tol 78,5%) |
| ML Kit terbaik | 40,1% (55,5% dengan normalizer) |

Ekspor TFLite fp16: 2,28 MB, paritas 100% dengan PyTorch. Bentuk akhir: kustom saja, tanpa cadangan ML Kit.

## Alur (jalankan dari root repo)

```bash
# 1. Korpus + split tetap -> dataset/ocr_corpus/{crops,manifest.csv}
python teman_pamor_anpr/02_training_platform/03_ocr_custom/scripts/build_corpus.py

# 2. Plat sintetis (hanya data latih) -> dataset/ocr_corpus/synth
python teman_pamor_anpr/02_training_platform/03_ocr_custom/scripts/synth_plates.py --n 30000

# 3. Latih -> models/ocr_custom/best.pt (dipilih dengan validasi asli)
python teman_pamor_anpr/02_training_platform/03_ocr_custom/src/train.py --epochs 60

# 4. Ekspor TFLite + uji paritas
python teman_pamor_anpr/02_training_platform/03_ocr_custom/src/export_tflite.py models/ocr_custom/best.pt
```

Soal uji terkunci (`--eval-test`) hanya dijalankan sekali per versi besar.

## Kontrak untuk aplikasi

| | |
|---|---|
| Input | `[1, 48, 192, 1]` float32, abu-abu 0-255, letterbox ke 48×192 (normalisasi ada di dalam model) |
| Output | `[1, 48, 37]` logit per langkah waktu; indeks 0 = blank CTC, `i` = `A-Z0-9`[i-1]; dekode greedy di klien |

## Data

Dataset, bobot, dan `.tflite` tidak di-commit. Contoh 50 crop plat ada di `samples/`
(nama berkas = `sumber_NOMOR_i.jpg`, sumber: jmto/kaggle/own).
