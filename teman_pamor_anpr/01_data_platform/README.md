# 📊 Data Platform

Data management: labeling, validation, augmentation

## Structure

```
01_data_platform/
├── labeling_tools/    # Interactive labeling GUI
├── datasets/
│   ├── 00_raw/        # Raw unprocessed data
│   ├── 01_validated/  # Quality-checked data
│   └── 02_augmented/  # Training-ready data
└── notebooks/         # Data exploration
```

## Labeling Tool

```bash
python labeling_tools/label_tool.py
```

**Features:**
- ✅ Polygon & bounding box mode
- ✅ 4-color classification (White, Black, Red, Yellow)
- ✅ Auto-crop & export
- ✅ Zoom + Pan
- ✅ Keyboard shortcuts

## Data Flow

```
Raw Images → Validation → Augmentation → Training
```
