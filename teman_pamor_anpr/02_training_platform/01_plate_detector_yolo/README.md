# Plate Detector (YOLO)

YOLO-based plate detector training module.

## Structure

- `src/train.py` - main training entrypoint
- `src/augment_dataset.py` - rotation augmentation builder
- `src/create_grouped_val_split.py` - grouped validation split generator
- `src/compare_checkpoints.py` - checkpoint comparison utility
- `configs/plat_jabar.yaml` - dataset config (canonical path)

## Current Canonical Dataset

- `dataset/plate_detection_augmented/plate_detection_augmented.yaml`

Dataset dan bobot model tidak di-commit. Contoh 50 gambar + label YOLO ada di `samples/`
(gambar diperkecil ke 1280 px; label ternormalisasi, jadi tetap valid).

## End-to-End Commands

From repository root:

```powershell
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/augment_dataset.py
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/create_grouped_val_split.py --dataset dataset/plate_detection_augmented --val-ratio 0.2 --seed 42
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/train.py
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/compare_checkpoints.py
```

## Expected Outputs

- Training runs under `runs/plate_detection/`
- Best checkpoint under `runs/plate_detection/<run_name>/weights/best.pt`
- Copied best model at repository root: `best_native_rotation.pt`

## Plat miring (kemiringan bebas)

Masalah: plat yang miring sekitar 30° sering tidak terdeteksi. Penyebabnya, dataset hanya punya rotasi 0/90/180/270°
dan training memakai `degrees` maksimal ±12°.

**Soal uji miring** (`src/rotate_dataset.py test`): 245 gambar validasi asli (265 plat), masing-masing diputar ke
0, ±15, ±30, dan ±45°. Labelnya dihitung ulang dari 4 sudut kotak asli, jadi tidak perlu melabel ulang.

Recall (IoU kotak tegak ≥ 0,5) per sudut, diukur 16 Nov 2026 (`src/eval_tilt.py`):

| Model | 0° | -15° | +15° | -30° | +30° | -45° | +45° |
|---|---|---|---|---|---|---|---|
| `plat_recognation.tflite` (di aplikasi, Colab 15 Okt 2025) | 85,7% | 46,0% | 71,7% | 0,8% | 12,1% | 0,0% | 0,0% |
| `checkpoints/best.pt` (epoch170) | 90,9% | 91,3% | 89,4% | 74,0% | 79,2% | 4,9% | 6,0% |

Model di aplikasi bukan epoch170. Input/output keduanya sama (`[1,640,640,3]` → `[1,5,8400]`), jadi epoch170
yang diekspor ke TFLite bisa langsung menggantikannya.

**Detektor OBB** (kotak miring + sudut, supaya crop bisa diluruskan sebelum OCR):

```bash
# dataset: setiap gambar train/val + 2 varian sudut acak ±45° -> dataset/plate_detection_obb
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/rotate_dataset.py obb --k 2 --max 45
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/train_obb.py
python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/eval_tilt.py --model runs/plate_detection/yolo11n_obb_tilt/weights/best.pt
```

Catatan: trik rotasi mengandaikan plat asli kurang lebih tegak. Tepi kanvas diisi `BORDER_REPLICATE`
(tidak memakai refleksi, karena refleksi memunculkan salinan plat yang tidak berlabel).
