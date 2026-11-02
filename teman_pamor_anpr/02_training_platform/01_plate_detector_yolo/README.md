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
