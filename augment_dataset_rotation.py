"""
Data Augmentation Script - Add 90°, 180°, 270° Rotation to Dataset
Membuat augmentasi dataset dengan rotasi untuk training yang lebih robust
"""

import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

def rotate_image_and_label(image_path, label_path, output_img_dir, output_label_dir, angle):
    """
    Rotate image and adjust YOLO label coordinates
    
    Args:
        image_path: Path to source image
        label_path: Path to source YOLO label
        output_img_dir: Directory to save rotated image
        output_label_dir: Directory to save rotated label
        angle: Rotation angle (90, 180, or 270)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Rotate image
    if angle == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return False
    
    # Generate output filename with rotation suffix
    stem = Path(image_path).stem
    ext = Path(image_path).suffix
    output_img_name = f"{stem}_rot{angle}{ext}"
    output_label_name = f"{stem}_rot{angle}.txt"
    
    # Save rotated image
    output_img_path = output_img_dir / output_img_name
    cv2.imwrite(str(output_img_path), rotated_img)
    
    # Process label if exists
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        rotated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_w = float(parts[3])
                bbox_h = float(parts[4])
                
                # Transform coordinates based on rotation
                if angle == 90:
                    # 90° clockwise: (x,y) -> (y, 1-x)
                    new_x = y_center
                    new_y = 1 - x_center
                    new_w = bbox_h
                    new_h = bbox_w
                elif angle == 180:
                    # 180°: (x,y) -> (1-x, 1-y)
                    new_x = 1 - x_center
                    new_y = 1 - y_center
                    new_w = bbox_w
                    new_h = bbox_h
                elif angle == 270:
                    # 270° clockwise (90° counter-clockwise): (x,y) -> (1-y, x)
                    new_x = 1 - y_center
                    new_y = x_center
                    new_w = bbox_h
                    new_h = bbox_w
                
                rotated_lines.append(f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")
        
        # Save rotated label
        output_label_path = output_label_dir / output_label_name
        with open(output_label_path, 'w') as f:
            f.writelines(rotated_lines)
    
    return True

def augment_dataset(dataset_dir, output_dir, angles=[90, 180, 270]):
    """
    Augment entire dataset with rotations
    
    Args:
        dataset_dir: Path to original YOLO dataset
        output_dir: Path to save augmented dataset
        angles: List of rotation angles to apply
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    print("=" * 70)
    print("🔄 DATA AUGMENTATION - ROTATION")
    print("=" * 70)
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Angles: {angles}")
    print("=" * 70)
    
    # Create output directory structure
    splits = ['train', 'val']
    
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    stats = {'train': {'original': 0, 'augmented': 0}, 
             'val': {'original': 0, 'augmented': 0}}
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split...")
        print(f"{'='*70}")
        
        # Source directories
        src_img_dir = dataset_dir / 'images' / split
        src_label_dir = dataset_dir / 'labels' / split
        
        # Output directories
        out_img_dir = output_dir / 'images' / split
        out_label_dir = output_dir / 'labels' / split
        
        if not src_img_dir.exists():
            print(f"⚠️  Warning: {src_img_dir} does not exist, skipping...")
            continue
        
        # Get all images
        image_files = list(src_img_dir.glob('*.jpg')) + \
                     list(src_img_dir.glob('*.png')) + \
                     list(src_img_dir.glob('*.jpeg'))
        
        if len(image_files) == 0:
            print(f"⚠️  Warning: No images found in {src_img_dir}")
            continue
        
        print(f"Found {len(image_files)} images")
        
        # Copy original files
        print("📋 Copying original files...")
        for img_path in tqdm(image_files, desc="Copying originals"):
            # Copy image
            shutil.copy2(img_path, out_img_dir / img_path.name)
            stats[split]['original'] += 1
            
            # Copy label if exists
            label_path = src_label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_label_dir / f"{img_path.stem}.txt")
        
        # Generate rotated versions
        print(f"\n🔄 Generating rotated versions ({angles})...")
        for img_path in tqdm(image_files, desc="Augmenting"):
            label_path = src_label_dir / f"{img_path.stem}.txt"
            
            for angle in angles:
                success = rotate_image_and_label(
                    img_path, label_path,
                    out_img_dir, out_label_dir,
                    angle
                )
                if success:
                    stats[split]['augmented'] += 1
        
        total = stats[split]['original'] + stats[split]['augmented']
        print(f"\n✅ {split.upper()} completed:")
        print(f"   - Original: {stats[split]['original']}")
        print(f"   - Augmented: {stats[split]['augmented']}")
        print(f"   - Total: {total}")
    
    # Create YAML config file
    yaml_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'license_plate'},
        'nc': 1
    }
    
    yaml_path = output_dir / 'plate_detection_augmented.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print("\n" + "="*70)
    print("✅ AUGMENTATION COMPLETED!")
    print("="*70)
    print("\n📊 SUMMARY:")
    total_original = sum(s['original'] for s in stats.values())
    total_augmented = sum(s['augmented'] for s in stats.values())
    total_images = total_original + total_augmented
    multiplier = total_images / total_original if total_original > 0 else 0
    
    print(f"   Original images:  {total_original:,}")
    print(f"   Augmented images: {total_augmented:,}")
    print(f"   Total images:     {total_images:,}")
    print(f"   Multiplier:       {multiplier:.1f}x")
    print(f"\n📁 Dataset config: {yaml_path}")
    print(f"📁 Dataset path:   {output_dir}")
    print("="*70)
    
    return yaml_path

if __name__ == "__main__":
    # Configuration
    ORIGINAL_DATASET = "dataset/plate_detection_yolo"
    AUGMENTED_DATASET = "dataset/plate_detection_augmented"
    ROTATION_ANGLES = [90, 180, 270]  # Tambahkan sudut rotasi yang diinginkan
    
    print("\n" + "🚀 STARTING DATA AUGMENTATION")
    print("="*70)
    
    # Check if original dataset exists
    if not Path(ORIGINAL_DATASET).exists():
        print(f"❌ ERROR: Dataset not found at {ORIGINAL_DATASET}")
        print("Please check the path and try again.")
        exit(1)
    
    # Run augmentation
    yaml_path = augment_dataset(
        dataset_dir=ORIGINAL_DATASET,
        output_dir=AUGMENTED_DATASET,
        angles=ROTATION_ANGLES
    )
    
    print("\n✨ Next step: Train model with augmented dataset")
    print(f"   Run: python train_plate_detection_augmented.py")
    print("\n" + "="*70)
