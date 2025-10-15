"""
Auto-annotation Script for Character Detection Dataset
Menggunakan PaddleOCR untuk auto-generate bounding boxes per character
"""

import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from pathlib import Path
import os
from tqdm import tqdm
import shutil
from PIL import Image

class CharacterDatasetPreparator:
    def __init__(self, source_dir, output_dir, label_csv):
        """
        Prepare character detection dataset from plate images
        
        Args:
            source_dir: Directory with plate images
            output_dir: Output directory for YOLO format dataset
            label_csv: CSV file with image filenames and text labels
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.label_csv = label_csv
        
        # Character mapping (0-9, A-Z)
        self.char_to_class = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
            'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
            'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
            'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
            'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
            'Z': 35
        }
        
        # Initialize PaddleOCR for auto-annotation
        print("🔧 Loading PaddleOCR for auto-annotation...")
        self.ocr = PaddleOCR(lang='en', use_textline_orientation=False)
        print("   ✅ PaddleOCR loaded!")
    
    def create_directory_structure(self):
        """Create YOLO dataset directory structure"""
        print("\n📁 Creating directory structure...")
        
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val'
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {d}")
    
    def normalize_bbox(self, bbox, img_width, img_height):
        """
        Convert bounding box to YOLO format (normalized)
        
        Args:
            bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            img_width, img_height: Image dimensions
            
        Returns:
            (x_center, y_center, width, height) - all normalized [0-1]
        """
        # Extract coordinates
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        
        # Get bounding box
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Calculate center and size
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return x_center, y_center, width, height
    
    def auto_annotate_image(self, image_path, expected_text):
        """
        Auto-annotate single image using PaddleOCR
        
        Args:
            image_path: Path to image
            expected_text: Ground truth text (e.g., "H2251RB")
            
        Returns:
            List of annotations: [(class, x_center, y_center, width, height), ...]
        """
        # Read image
        img = Image.open(str(image_path))
        if img is None:
            return None
        
        w, h = img.size
        
        # Run OCR to get character bounding boxes
        try:
            result = self.ocr.predict(str(image_path))
            
            # Check result structure
            if result is None or not hasattr(result, 'boxes') or len(result.boxes) == 0:
                return None
            
            # Extract detections from new API
            detections = []
            for i in range(len(result.boxes)):
                bbox = result.boxes[i]  # Bounding box coordinates
                text = result.text[i] if hasattr(result, 'text') else ''
                conf = result.scores[i] if hasattr(result, 'scores') else 1.0
                detections.append((bbox, (text, conf)))
            
            if len(detections) == 0:
                return None
            
            # Sort detections by x coordinate (left to right)
            detections.sort(key=lambda x: min([p[0] for p in x[0]]) if isinstance(x[0], (list, tuple)) else x[0][0])
            
            # Match detected text with expected text
            detected_text = ''.join([det[1][0].upper() for det in detections])
            detected_text = ''.join(c for c in detected_text if c.isalnum())
            
            # Clean expected text
            expected_clean = ''.join(c for c in expected_text.upper() if c.isalnum())
            
            # If lengths don't match, try to align
            if len(detections) != len(expected_clean):
                print(f"   ⚠️ Mismatch: detected={detected_text} vs expected={expected_clean}")
                
                # If detection found more characters, limit to expected length
                if len(detections) > len(expected_clean):
                    detections = detections[:len(expected_clean)]
                else:
                    # If too few detections, skip this image
                    return None
            
            # Create annotations
            annotations = []
            for i, (bbox, (text, conf)) in enumerate(detections):
                if i >= len(expected_clean):
                    break
                
                # Use expected character (ground truth)
                char = expected_clean[i]
                
                # Skip if character not in our mapping
                if char not in self.char_to_class:
                    continue
                
                # Get class ID
                class_id = self.char_to_class[char]
                
                # Normalize bounding box
                x_c, y_c, width, height = self.normalize_bbox(bbox, w, h)
                
                # Add annotation
                annotations.append((class_id, x_c, y_c, width, height))
            
            return annotations if annotations else None
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            return None
    
    def save_annotation(self, annotations, output_path):
        """Save annotations in YOLO format"""
        with open(output_path, 'w') as f:
            for class_id, x_c, y_c, w, h in annotations:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    
    def process_dataset(self, train_split=0.8):
        """
        Process entire dataset
        
        Args:
            train_split: Fraction of data for training (0.8 = 80% train, 20% val)
        """
        print("\n🔄 Processing dataset...")
        
        # Load labels
        df = pd.read_csv(self.label_csv)
        print(f"   📊 Total images: {len(df)}")
        
        # Statistics
        total = len(df)
        processed = 0
        skipped = 0
        train_count = 0
        val_count = 0
        
        # Process each image
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            filename = row['filename']
            label = row['label']
            
            # Source image path
            source_path = self.source_dir / filename
            
            if not source_path.exists():
                # Try without extension
                source_path = self.source_dir / f"{label}.jpg"
                if not source_path.exists():
                    skipped += 1
                    continue
            
            # Auto-annotate
            annotations = self.auto_annotate_image(source_path, label)
            
            if annotations is None:
                skipped += 1
                continue
            
            # Determine train or val split
            is_train = np.random.rand() < train_split
            split = 'train' if is_train else 'val'
            
            # Output paths
            img_out = self.output_dir / 'images' / split / filename
            txt_out = self.output_dir / 'labels' / split / f"{Path(filename).stem}.txt"
            
            # Copy image
            shutil.copy(source_path, img_out)
            
            # Save annotation
            self.save_annotation(annotations, txt_out)
            
            # Update stats
            processed += 1
            if is_train:
                train_count += 1
            else:
                val_count += 1
        
        # Print summary
        print(f"\n📊 Processing Summary:")
        print(f"   Total: {total}")
        print(f"   Processed: {processed} ({processed/total*100:.1f}%)")
        print(f"   Skipped: {skipped} ({skipped/total*100:.1f}%)")
        print(f"   Train: {train_count}")
        print(f"   Val: {val_count}")
    
    def create_yaml_config(self):
        """Create YAML config file for YOLO training"""
        yaml_path = self.output_dir / 'character_detection.yaml'
        
        yaml_content = f"""# Character Detection Dataset Config
path: {self.output_dir.absolute()}
train: images/train
val: images/val

# Number of classes (0-9, A-Z)
nc: 36

# Class names
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z']
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ YAML config created: {yaml_path}")
        return yaml_path
    
    def visualize_sample(self, num_samples=5):
        """Visualize sample annotations"""
        print(f"\n👁️ Visualizing {num_samples} samples...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("   ⚠️ matplotlib not installed. Skipping visualization.")
            return
        
        # Get random samples from train
        train_imgs = list((self.output_dir / 'images' / 'train').glob('*.jpg'))
        if len(train_imgs) == 0:
            train_imgs = list((self.output_dir / 'images' / 'train').glob('*.png'))
        
        if len(train_imgs) == 0:
            print("   ⚠️ No training images found.")
            return
            
        samples = np.random.choice(train_imgs, min(num_samples, len(train_imgs)), 
                                  replace=False)
        
        for img_path in samples:
            # Read image
            img = Image.open(str(img_path))
            w, h = img.size
            
            # Read annotation
            txt_path = self.output_dir / 'labels' / 'train' / f"{img_path.stem}.txt"
            
            if not txt_path.exists():
                continue
            
            # Create figure
            fig, ax = plt.subplots(1, figsize=(10, 6))
            ax.imshow(img)
            
            # Parse annotation
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            text_parts = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_c, y_c, width, height = map(float, parts[1:])
                
                # Convert to pixel coordinates
                x1 = int((x_c - width/2) * w)
                y1 = int((y_c - height/2) * h)
                bbox_w = int(width * w)
                bbox_h = int(height * h)
                
                # Get character
                char = list(self.char_to_class.keys())[class_id]
                text_parts.append(char)
                
                # Draw rectangle
                rect = patches.Rectangle((x1, y1), bbox_w, bbox_h,
                                        linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, char, color='green', fontsize=12, weight='bold')
            
            # Add full text as title
            full_text = ''.join(text_parts)
            ax.set_title(f'Sample: {img_path.name} - Text: {full_text}', fontsize=14)
            ax.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        print("   ✅ Visualization complete!")


def main():
    """Main function"""
    print("=" * 60)
    print("🔤 Character Detection Dataset Preparation")
    print("=" * 60)
    
    # Configuration
    SOURCE_DIR = 'dataset/plate_text_dataset/dataset'
    OUTPUT_DIR = 'dataset/character_recognition_yolo'
    LABEL_CSV = 'dataset/plate_text_dataset/label.csv'
    
    # Create preparator
    preparator = CharacterDatasetPreparator(SOURCE_DIR, OUTPUT_DIR, LABEL_CSV)
    
    # Step 1: Create directory structure
    preparator.create_directory_structure()
    
    # Step 2: Process dataset (auto-annotate with PaddleOCR)
    preparator.process_dataset(train_split=0.8)
    
    # Step 3: Create YAML config
    yaml_path = preparator.create_yaml_config()
    
    # Step 4: Visualize samples
    print("\n📸 Do you want to visualize samples? (y/n): ", end='')
    response = input().strip().lower()
    if response == 'y':
        preparator.visualize_sample(num_samples=5)
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"\n📂 Dataset location: {OUTPUT_DIR}")
    print(f"📄 Config file: {yaml_path}")
    print("\n🚀 Next steps:")
    print("   1. Verify annotations (check samples)")
    print("   2. Train model: python train_character_detection.py")
    print("   3. Test OCR: python test_yolo_ocr.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
