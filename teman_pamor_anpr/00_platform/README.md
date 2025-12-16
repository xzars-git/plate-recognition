# 🛠️ Platform Layer

Shared infrastructure components used across all 4 models.

## Components

### 1. Data Validation (`data_validation/`)
- Image quality checks
- Label validation
- Duplicate detection

### 2. Preprocessing (`preprocessing/`)
- Image normalization
- Resizing utilities
- Color space conversion

### 3. Model Registry (`model_registry/`)
- Model metadata tracking
- Version management
- Performance metrics

### 4. Monitoring (`monitoring/`)
- Performance tracking
- Drift detection
- Alerting

## Usage

```python
# Example: Data validation
from platform.data_validation import ImageValidator

validator = ImageValidator(min_width=50, min_height=50)
result = validator.validate_image("image.jpg")
```
