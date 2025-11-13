#!/usr/bin/env python
"""
Quick verification: Check if annotations are saved correctly
"""

from pathlib import Path

def verify_annotations(dataset_path):
    """Verify YOLO format annotations"""
    
    dataset = Path(dataset_path)
    images_dir = dataset / 'images'
    labels_dir = dataset / 'labels'
    
    print(f"\n🔍 Verifying: {dataset_path}")
    print("="*70)
    
    # Check directories exist
    if not images_dir.exists():
        print("❌ Images folder not found")
        return
    
    if not labels_dir.exists():
        print("❌ Labels folder not found")
        return
    
    # Get all images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    label_files = list(labels_dir.glob('*.txt'))
    
    print(f"📁 Images: {len(image_files)}")
    print(f"📝 Labels: {len(label_files)}")
    print()
    
    if len(image_files) == 0:
        print("⚠️  No images found (dataset empty - label some images first!)")
        return
    
    # Check matching
    matched = 0
    unmatched_images = []
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        if label_file.exists():
            matched += 1
            
            # Validate YOLO format
            try:
                with open(label_file) as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    
                    if len(parts) != 5:
                        print(f"❌ Invalid format in {label_file.name}: {line.strip()}")
                        continue
                    
                    class_id, x, y, w, h = map(float, parts)
                    
                    # Validate normalized values
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"❌ Invalid normalized values in {label_file.name}: {line.strip()}")
                        
            except Exception as e:
                print(f"❌ Error reading {label_file.name}: {e}")
        else:
            unmatched_images.append(img_file.name)
    
    # Results
    print(f"✅ Matched: {matched}/{len(image_files)}")
    
    if unmatched_images:
        print(f"⚠️  Unmatched: {len(unmatched_images)}")
        if len(unmatched_images) <= 5:
            for img in unmatched_images:
                print(f"   - {img}")
        else:
            print(f"   - {unmatched_images[0]}")
            print(f"   - {unmatched_images[1]}")
            print(f"   ... and {len(unmatched_images)-2} more")
    
    print()
    
    # Sample annotation
    if label_files:
        sample_label = label_files[0]
        print(f"📄 Sample annotation: {sample_label.name}")
        print("-"*70)
        with open(sample_label) as f:
            content = f.read()
            print(content if content else "(empty - no boxes drawn)")
        print("-"*70)
    
    print()
    
    if matched == len(image_files) and matched > 0:
        print("🎉 All images have matching labels!")
        print("✅ YOLO format validated")
        print("✅ Ready for training!")
    elif matched > 0:
        print(f"⚠️  {len(unmatched_images)} images still need labeling")
    else:
        print("⚠️  No labels found - start labeling with label_tool_gui.py!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔍 ANNOTATION VERIFICATION TOOL")
    print("="*70)
    
    # Check all datasets
    datasets = [
        'dataset/plate_detection_yolo/train',
        'dataset/plate_detection_yolo/val',
        'dataset/plate_detection_augmented/train',
        'dataset/plate_detection_augmented/val',
    ]
    
    for dataset in datasets:
        if Path(dataset).exists():
            verify_annotations(dataset)
    
    print("="*70)
    print("✅ Verification complete!")
    print()
