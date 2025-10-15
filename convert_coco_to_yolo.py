"""
Konversi COCO JSON ke YOLO Format untuk Plate Detection
"""

import json
import os
from pathlib import Path
import shutil

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    """
    Konversi COCO format annotations ke YOLO format
    
    Args:
        coco_json_path: Path ke file annotations.json
        images_dir: Path ke folder images
        output_dir: Path output untuk YOLO dataset
    """
    
    # Buat folder output
    output_dir = Path(output_dir)
    train_images = output_dir / 'images' / 'train'
    train_labels = output_dir / 'labels' / 'train'
    val_images = output_dir / 'images' / 'val'
    val_labels = output_dir / 'labels' / 'val'
    
    for folder in [train_images, train_labels, val_images, val_labels]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Buat mapping image_id ke filename dan size
    images_info = {}
    for img in coco_data['images']:
        images_info[img['id']] = {
            'file_name': img['file_name'],
            'width': int(img['width']),
            'height': int(img['height'])
        }
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Split: 80% train, 20% val
    total_images = len(images_info)
    train_count = int(total_images * 0.8)
    
    print(f"\n🔄 Converting {total_images} images...")
    print(f"   Train: {train_count} images")
    print(f"   Val: {total_images - train_count} images\n")
    
    converted = 0
    for idx, (img_id, img_info) in enumerate(images_info.items()):
        # Tentukan train atau val
        is_train = idx < train_count
        dest_images = train_images if is_train else val_images
        dest_labels = train_labels if is_train else val_labels
        
        # Copy image
        src_image = Path(images_dir) / img_info['file_name']
        if src_image.exists():
            shutil.copy(src_image, dest_images / img_info['file_name'])
        else:
            print(f"⚠️  Image not found: {src_image}")
            continue
        
        # Convert annotations to YOLO format
        label_file = dest_labels / f"{Path(img_info['file_name']).stem}.txt"
        
        if img_id in annotations_by_image:
            with open(label_file, 'w') as f:
                for ann in annotations_by_image[img_id]:
                    # COCO format: [x, y, width, height] (absolute coordinates)
                    # YOLO format: [class x_center y_center width height] (normalized 0-1)
                    
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    
                    # Convert to YOLO format (normalized)
                    img_width = img_info['width']
                    img_height = img_info['height']
                    
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width_norm = w / img_width
                    height_norm = h / img_height
                    
                    # Class ID (untuk plate detection hanya 1 class: 0)
                    # YOLO class ID dimulai dari 0
                    category_id = ann.get('category_id', 1)
                    if category_id > 0:
                        class_id = category_id - 1  # COCO category_id mulai dari 1
                    else:
                        class_id = 0  # Default ke class 0
                    
                    # Write to file
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        converted += 1
        if (converted % 100) == 0:
            print(f"   Converted {converted}/{total_images} images...")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Train images: {len(list(train_images.glob('*')))}")
    print(f"   Val images: {len(list(val_images.glob('*')))}")

if __name__ == '__main__':
    # Path ke dataset
    coco_json = 'dataset/plate_detection_dataset/annotations/annotations.json'
    images_dir = 'dataset/plate_detection_dataset/images'
    output_dir = 'dataset/plate_detection_yolo'
    
    print("="*60)
    print("COCO to YOLO Converter - Plate Detection")
    print("="*60)
    
    if not Path(coco_json).exists():
        print(f"❌ File not found: {coco_json}")
    elif not Path(images_dir).exists():
        print(f"❌ Directory not found: {images_dir}")
    else:
        convert_coco_to_yolo(coco_json, images_dir, output_dir)
        
        # Buat YAML config file
        yaml_content = f"""# Plate Detection Dataset (YOLO Format)
# Converted from COCO format

path: {Path(output_dir).absolute()}
train: images/train
val: images/val

# Class information
nc: 1
names: ['license_plate']
"""
        
        yaml_path = Path(output_dir) / 'plate_detection.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n📄 YAML config created: {yaml_path}")
        print("\n🚀 Ready to train! Use:")
        print(f"   python train_plate_detection.py")
