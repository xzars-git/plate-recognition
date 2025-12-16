# 📱 Deployment Platform

Mobile deployment: TFLite conversion, validation, serving

## Structure

```
03_deployment_platform/
├── conversion/        # PyTorch/TF → TFLite
├── validation/        # Pre-deploy checks
├── mobile_models/     # Versioned models
├── flutter_integration/ # Flutter code
├── inference/         # Desktop testing
└── monitoring/        # Production monitoring
```

## Deployment Flow

```
Training → Conversion → Validation → Staging → Production
```

## Validation Checks

1. ✅ Accuracy (< 2% drop from training)
2. ✅ Latency (< 200ms on mobile)
3. ✅ Size (< 10MB per model)
4. ✅ Compatibility (TFLite ops)

## Usage

```bash
# Convert to TFLite
python conversion/to_tflite.py --model best.pt --output plate_detector.tflite

# Validate
python validation/accuracy_test.py --model plate_detector.tflite
python validation/latency_benchmark.py --model plate_detector.tflite

# Deploy
cp mobile_models/v1.0.0/*.tflite flutter_integration/assets/models/
```
