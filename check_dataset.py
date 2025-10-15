"""
Check Dataset Status
"""
from pathlib import Path

# Check labels
train_labels = list(Path('dataset/plate_detection_yolo/labels/train').glob('*.txt'))
val_labels = list(Path('dataset/plate_detection_yolo/labels/val').glob('*.txt'))

# Check images
train_images = list(Path('dataset/plate_detection_yolo/images/train').glob('*'))
val_images = list(Path('dataset/plate_detection_yolo/images/val').glob('*'))

print("="*60)
print("DATASET STATUS CHECK")
print("="*60)
print(f"\nTRAIN:")
print(f"  Images: {len(train_images)}")
print(f"  Labels: {len(train_labels)}")

print(f"\nVALIDATION:")
print(f"  Images: {len(val_images)}")
print(f"  Labels: {len(val_labels)}")

# Check label content
if train_labels:
    print(f"\nSample label (train): {train_labels[0].name}")
    with open(train_labels[0], 'r') as f:
        content = f.read()
        print(f"  Content: {content.strip()}")

if val_labels:
    print(f"\nSample label (val): {val_labels[0].name}")
    with open(val_labels[0], 'r') as f:
        content = f.read()
        print(f"  Content: {content.strip()}")

# Check for issues
print("\n" + "="*60)
print("ISSUES FOUND:")
print("="*60)

if len(train_labels) == 0:
    print("❌ NO TRAIN LABELS!")
if len(val_labels) == 0:
    print("❌ NO VALIDATION LABELS!")
if len(train_images) != len(train_labels):
    print(f"⚠️  Mismatch: {len(train_images)} images but {len(train_labels)} labels")
if len(val_images) != len(val_labels):
    print(f"⚠️  Mismatch: {len(val_images)} images but {len(val_labels)} labels")

# Check class ID in labels
print("\n" + "="*60)
print("CHECKING CLASS IDs:")
print("="*60)

invalid_labels = []
for label_file in train_labels[:10]:  # Check first 10
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id < 0:
                    invalid_labels.append((label_file.name, class_id))
                    print(f"❌ Invalid class ID: {class_id} in {label_file.name}")

if not invalid_labels:
    print("✅ Class IDs look good (checked first 10 files)")
