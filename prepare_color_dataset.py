#!/usr/bin/env python
"""
🎨 Prepare Color Classification Dataset
Auto-crop plates dari labeled images dan organize by color
"""

from PIL import Image
from pathlib import Path
import json
import shutil
from tqdm import tqdm

def prepare_color_dataset(
    source_dir='dataset/plate_detection_color/train',
    output_dir='dataset/plate_colors',
    min_size=50,  # Minimum crop size
    target_size=(96, 96)  # Resize untuk training
):
    """
    Crop plates dan organize by color
    
    Structure:
    dataset/plate_colors/
        train/
            white/
            black/
            red/
            yellow/
        val/
            white/
            black/
            red/
            yellow/
    """
    
    print("="*70)
    print("🎨 PREPARING COLOR CLASSIFICATION DATASET")
    print("="*70)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val']:
        for color in ['white', 'black', 'red', 'yellow']:
            (output_path / split / color).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Source: {source_path}")
    print(f"📁 Output: {output_path}")
    
    # Process images
    images_dir = source_path / 'images'
    labels_dir = source_path / 'labels'
    
    if not images_dir.exists():
        print(f"\n❌ ERROR: Images directory not found: {images_dir}")
        return
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    print(f"\n🔍 Found {len(image_files)} images")
    print(f"⚙️ Target size: {target_size[0]}x{target_size[1]}px")
    print(f"⚙️ Minimum crop size: {min_size}px")
    
    stats = {
        'total': 0,
        'white': 0,
        'black': 0,
        'red': 0,
        'yellow': 0,
        'skipped': 0
    }
    
    print("\n🔄 Processing images...")
    for img_path in tqdm(image_files):
        # Load image
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"\n⚠️ Failed to load {img_path.name}: {e}")
            stats['skipped'] += 1
            continue
        
        # Load JSON labels
        json_path = labels_dir / f"{img_path.stem}.json"
        
        if not json_path.exists():
            stats['skipped'] += 1
            continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"\n⚠️ Failed to load {json_path.name}: {e}")
            stats['skipped'] += 1
            continue
        
        # Process each box
        boxes = data.get('boxes', [])
        
        for idx, box in enumerate(boxes):
            color = box.get('color', 'white')
            
            # Get bounding box coordinates
            if box.get('type') == 'polygon':
                # Convert polygon to bounding box
                points = box['points']
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)
            else:
                # Box format (YOLO)
                yolo = box.get('yolo', box)
                _, x_center, y_center, width, height = yolo
                
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
            
            # Convert normalized coords to pixels
            img_w, img_h = img.size
            x1 = int(x_min * img_w)
            y1 = int(y_min * img_h)
            x2 = int(x_max * img_w)
            y2 = int(y_max * img_h)
            
            # Validate crop size
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            if crop_w < min_size or crop_h < min_size:
                stats['skipped'] += 1
                continue
            
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            
            # Crop plate
            try:
                plate_crop = img.crop((x1, y1, x2, y2))
                
                # Resize
                plate_resized = plate_crop.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save to color folder
                # 80% train, 20% val (based on file name hash)
                split = 'train' if hash(img_path.stem) % 5 != 0 else 'val'
                
                output_file = output_path / split / color / f"{img_path.stem}_{idx}.jpg"
                plate_resized.save(output_file, quality=95)
                
                stats['total'] += 1
                stats[color] += 1
                
            except Exception as e:
                print(f"\n⚠️ Failed to crop {img_path.name} box {idx}: {e}")
                stats['skipped'] += 1
                continue
    
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Statistics:")
    print(f"   Total crops: {stats['total']}")
    print(f"   ⚪ White: {stats['white']}")
    print(f"   ⚫ Black: {stats['black']}")
    print(f"   🔴 Red: {stats['red']}")
    print(f"   🟡 Yellow: {stats['yellow']}")
    print(f"   ⚠️ Skipped: {stats['skipped']}")
    
    # Distribution
    print(f"\n📈 Distribution:")
    total_valid = stats['total']
    if total_valid > 0:
        for color in ['white', 'black', 'red', 'yellow']:
            pct = (stats[color] / total_valid) * 100
            print(f"   {color.capitalize()}: {pct:.1f}%")
    
    # Check train/val split
    train_count = sum(len(list((output_path / 'train' / color).glob('*.jpg'))) 
                     for color in ['white', 'black', 'red', 'yellow'])
    val_count = sum(len(list((output_path / 'val' / color).glob('*.jpg'))) 
                   for color in ['white', 'black', 'red', 'yellow'])
    
    print(f"\n📁 Split:")
    print(f"   Train: {train_count} images ({train_count/(train_count+val_count)*100:.1f}%)")
    print(f"   Val: {val_count} images ({val_count/(train_count+val_count)*100:.1f}%)")
    
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    min_samples = 50
    for color in ['white', 'black', 'red', 'yellow']:
        if stats[color] < min_samples:
            print(f"   ⚠️ Need more {color} plates: {stats[color]}/{min_samples} (add {min_samples - stats[color]} more)")
    
    if all(stats[color] >= min_samples for color in ['white', 'black', 'red', 'yellow']):
        print(f"   ✅ Dataset ready for training!")
        print(f"   Run: python train_color_classifier.py")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare color classification dataset')
    parser.add_argument('--source', default='dataset/plate_detection_color/train',
                       help='Source directory with labeled images')
    parser.add_argument('--output', default='dataset/plate_colors',
                       help='Output directory for color dataset')
    parser.add_argument('--size', type=int, default=96,
                       help='Target size for crops (default: 96)')
    parser.add_argument('--min-size', type=int, default=50,
                       help='Minimum crop size (default: 50)')
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    
    stats = prepare_color_dataset(
        source_dir=args.source,
        output_dir=args.output,
        min_size=args.min_size,
        target_size=target_size
    )
