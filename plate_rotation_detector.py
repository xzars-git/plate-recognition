"""
Plate Rotation Detector & Preprocessor
Mendeteksi dan memperbaiki orientasi plat nomor yang dirotasi (90°/180°/270°)
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


class PlateRotationDetector:
    """
    Deteksi dan koreksi rotasi plat nomor
    Menggunakan teknik:
    1. Edge detection
    2. Aspect ratio analysis
    3. Text region detection (horizontal vs vertical)
    """
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def detect_rotation_angle(self, image):
        """
        Deteksi sudut rotasi gambar plat nomor
        
        Returns:
            angle: 0, 90, 180, atau 270 (derajat)
            confidence: confidence score (0-1)
        """
        h, w = image.shape[:2]
        
        # Method 1: Aspect Ratio
        # Plat nomor Indonesia: landscape (lebih lebar dari tinggi)
        # Ratio normal: ~2.5:1 (width:height)
        aspect_ratio = w / h if h > 0 else 0
        
        if self.debug:
            print(f"  📐 Aspect ratio: {aspect_ratio:.2f} (w={w}, h={h})")
        
        # Decision logic
        if 1.5 < aspect_ratio < 4.0:
            # Landscape - kemungkinan sudah benar (0°) atau terbalik (180°)
            angle, conf = self._check_upside_down(image)
            return angle, conf
        
        elif 0.25 < aspect_ratio < 0.67:
            # Portrait - kemungkinan rotasi 90° atau 270°
            angle, conf = self._check_90_or_270(image)
            return angle, conf
        
        else:
            # Aspect ratio aneh - return 0 dengan confidence rendah
            return 0, 0.3
    
    def _check_upside_down(self, image):
        """Check apakah gambar terbalik (180°)"""
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Analisa distribusi edge di top vs bottom half
        top_half = edges[:h//2, :]
        bottom_half = edges[h//2:, :]
        
        top_edges = np.sum(top_half > 0)
        bottom_edges = np.sum(bottom_half > 0)
        
        if self.debug:
            print(f"  📊 Edges - Top: {top_edges}, Bottom: {bottom_edges}")
        
        # Plat nomor biasanya punya lebih banyak edge di bagian tengah
        # Kalau bottom lebih banyak edge, kemungkinan terbalik
        
        total_edges = top_edges + bottom_edges
        if total_edges == 0:
            return 0, 0.5
        
        bottom_ratio = bottom_edges / total_edges
        
        if bottom_ratio > 0.6:
            # Kemungkinan terbalik
            return 180, 0.7
        else:
            # Kemungkinan sudah benar
            return 0, 0.7
    
    def _check_90_or_270(self, image):
        """Check apakah rotasi 90° atau 270°"""
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Analisa distribusi edge di left vs right half
        left_half = edges[:, :w//2]
        right_half = edges[:, w//2:]
        
        left_edges = np.sum(left_half > 0)
        right_edges = np.sum(right_half > 0)
        
        if self.debug:
            print(f"  📊 Edges - Left: {left_edges}, Right: {right_edges}")
        
        total_edges = left_edges + right_edges
        if total_edges == 0:
            return 90, 0.5
        
        right_ratio = right_edges / total_edges
        
        if right_ratio > 0.55:
            # Kemungkinan 270° (perlu rotate CCW)
            return 270, 0.7
        else:
            # Kemungkinan 90° (perlu rotate CW)
            return 90, 0.7
    
    def rotate_image(self, image, angle):
        """
        Rotate image by angle
        
        Args:
            image: input image
            angle: 0, 90, 180, or 270 degrees
        
        Returns:
            rotated image
        """
        if angle == 0:
            return image
        
        elif angle == 90:
            # Rotate 90° clockwise
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        elif angle == 180:
            # Rotate 180°
            return cv2.rotate(image, cv2.ROTATE_180)
        
        elif angle == 270:
            # Rotate 270° clockwise (= 90° counter-clockwise)
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        else:
            print(f"⚠️  Invalid angle: {angle}. Returning original image.")
            return image
    
    def preprocess(self, image):
        """
        Preprocess image: detect rotation and correct it
        
        Args:
            image: input image (BGR or grayscale)
        
        Returns:
            corrected_image: image with corrected orientation
            angle: detected rotation angle
            confidence: confidence score
        """
        # Detect rotation
        angle, confidence = self.detect_rotation_angle(image)
        
        if self.debug:
            print(f"  🔄 Detected rotation: {angle}° (confidence: {confidence:.2f})")
        
        # Rotate if needed
        if angle != 0 and confidence > 0.5:
            corrected = self.rotate_image(image, angle)
            if self.debug:
                print(f"  ✅ Image corrected: rotated by {angle}°")
            return corrected, angle, confidence
        else:
            if self.debug:
                print(f"  ℹ️  No rotation needed")
            return image, 0, confidence


def process_image(image_path, output_path=None, debug=False):
    """
    Process single image
    
    Args:
        image_path: path to input image
        output_path: path to save corrected image (None = auto)
        debug: print debug info
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        return
    
    print(f"\n📄 Processing: {Path(image_path).name}")
    
    # Preprocess
    detector = PlateRotationDetector(debug=debug)
    corrected, angle, confidence = detector.preprocess(image)
    
    # Save result
    if output_path is None:
        # Auto generate output path
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
    
    cv2.imwrite(str(output_path), corrected)
    print(f"✅ Saved to: {output_path}")
    print(f"   Rotation: {angle}° (confidence: {confidence:.2f})")
    
    # Show comparison if debug
    if debug:
        show_comparison(image, corrected, angle)


def process_folder(input_folder, output_folder=None, debug=False):
    """
    Process all images in folder
    
    Args:
        input_folder: path to input folder
        output_folder: path to output folder (None = create 'corrected' subfolder)
        debug: print debug info
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ Folder not found: {input_folder}")
        return
    
    # Create output folder
    if output_folder is None:
        output_path = input_path / "corrected"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    print(f"\n📁 Processing folder: {input_folder}")
    print(f"📁 Output folder: {output_path}")
    print(f"📊 Found {len(image_files)} images")
    print("="*60)
    
    # Process each image
    detector = PlateRotationDetector(debug=debug)
    
    stats = {
        0: 0,    # No rotation
        90: 0,   # 90° rotation
        180: 0,  # 180° rotation
        270: 0   # 270° rotation
    }
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_file.name}")
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  ⚠️  Cannot read, skipping...")
            continue
        
        # Preprocess
        corrected, angle, confidence = detector.preprocess(image)
        stats[angle] += 1
        
        # Save
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), corrected)
        
        print(f"  Rotation: {angle}° (conf: {confidence:.2f})")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"  No rotation (0°):     {stats[0]}")
    print(f"  Rotated 90° CW:       {stats[90]}")
    print(f"  Rotated 180°:         {stats[180]}")
    print(f"  Rotated 270° CW:      {stats[270]}")
    print("="*60)
    print(f"✅ All corrected images saved to: {output_path}")


def show_comparison(original, corrected, angle):
    """Show before/after comparison (for debugging)"""
    # Resize if too large
    h, w = original.shape[:2]
    if w > 800:
        scale = 800 / w
        new_w = 800
        new_h = int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
    
    h, w = corrected.shape[:2]
    if w > 800:
        scale = 800 / w
        new_w = 800
        new_h = int(h * scale)
        corrected = cv2.resize(corrected, (new_w, new_h))
    
    # Stack images
    cv2.imshow('Original', original)
    cv2.imshow(f'Corrected (rotated {angle}°)', corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Detect and correct plate rotation (90°/180°/270°)'
    )
    parser.add_argument('input', type=str,
                       help='Input image or folder path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: auto-generated)')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    parser.add_argument('--folder', action='store_true',
                       help='Process entire folder')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔄 PLATE ROTATION DETECTOR & CORRECTOR")
    print("="*60)
    
    if args.folder:
        process_folder(args.input, args.output, args.debug)
    else:
        process_image(args.input, args.output, args.debug)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
