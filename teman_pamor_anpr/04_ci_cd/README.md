# 🔄 CI/CD Pipeline

Automated testing, validation, deployment

## Workflows

### 1. Data Validation (`01_data_validation.yml`)
- Trigger: New data uploaded
- Checks: Image quality, labels, duplicates

### 2. Model Training (`02_model_training.yml`)
- Trigger: Manual or schedule
- Runs: Training pipeline

### 3. Model Validation (`03_model_validation.yml`)
- Trigger: Training complete
- Checks: Accuracy, latency, size

### 4. Model Deployment (`04_model_deployment.yml`)
- Trigger: Validation passed
- Deploys: To mobile assets

## Setup (Phase by Phase)

### Phase 1: Manual Scripts (Week 1)
```bash
bash scripts/train_all_models.sh
```

### Phase 2: GitHub Actions (Week 2-3)
```yaml
# .github/workflows/02_model_training.yml
name: Train Models
on: [workflow_dispatch]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train Plate Detector
        run: python 02_training_platform/01_plate_detector_yolo/src/train.py
```

### Phase 3: Full Automation (Week 4+)
- Automatic trigger on data push
- Slack notifications
- Automated deployment
