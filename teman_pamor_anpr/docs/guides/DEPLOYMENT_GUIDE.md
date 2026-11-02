# Deployment Guide

Current deployment workflow for plate detector model in this repository.

## Scope

This guide reflects scripts that already exist today.

## 1. Validate runtime environment

```powershell
python teman_pamor_anpr/03_deployment_platform/inference/check_gpu.py
```

## 2. Quick inference sanity test

```powershell
python teman_pamor_anpr/03_deployment_platform/inference/quick_test.py
```

## 3. Desktop functional test

```powershell
python teman_pamor_anpr/03_deployment_platform/inference/desktop_app.py
```

## 4. Optional training enhancement before packaging

```powershell
python teman_pamor_anpr/03_deployment_platform/conversion/enhance_model.py
```

## 5. Model packaging target

Place production artifacts under versioned directory:

- `teman_pamor_anpr/03_deployment_platform/mobile_models/vX.Y.Z/`

Recommended files per release:

- model file (`.onnx` or `.tflite`)
- metadata json (version, date, metrics)
- checksum

## Notes

- Dedicated automated scripts for TFLite conversion/validation are planned but not yet present in this repository.
- Keep deployment commands aligned with files that exist in `03_deployment_platform/`.
