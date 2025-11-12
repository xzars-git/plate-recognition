"""
Test Model dengan Gambar + Rotation Preprocessing
Test deteksi plat nomor dengan automatic rotation correction
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse
from plate_rotation_detector import PlateRotationDetector


def test_images_with_rotation(model_path, image_path=None, conf=0.25, enable_rotation=True, debug=False):
    """
    Test model pada gambar dengan rotation preprocessing
    
    Args:
        model_path: Path ke model (.pt file)
        image_path: Path ke gambar atau folder (None = pakai validation set)
        conf: Confidence threshold
        enable_rotation: Enable rotation detection & correction
        debug: Show debug info
    """
    print("="*60)
    print("🖼️  IMAGE TESTING - Plate Detection + Rotation")
    print("="*60)
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded!")
    
    # Initialize rotation detector
    if enable_rotation:
        print("\n🔄 Rotation correction: ENABLED")
        rotation_detector = PlateRotationDetector(debug=debug)
    else:
        print("\n🔄 Rotation correction: DISABLED")
        rotation_detector = None
    
    # Determine source
    if image_path is None:
        # Use validation images
        source = 'dataset/plate_detection_yolo/images/val'
        print(f"\n📁 Testing on validation set: {source}")
    else:
        source = image_path
        print(f"\n📁 Testing on: {source}")
    
    # Get image files
    source_path = Path(source)
    if source_path.is_file():
        image_files = [source_path]
    else:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
    
    print(f"\n📊 Found {len(image_files)} images")
    print(f"🔍 Running detection (conf={conf})...")
    print("="*60 + "\n")
    
    # Create output folder
    output_dir = Path('runs/detect/test_with_rotation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    total_detections = 0
    rotation_stats = {0: 0, 90: 0, 180: 0, 270: 0}
    
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_file.name}")
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  ⚠️  Cannot read, skipping...")
            continue
        
        original_image = image.copy()
        
        # Apply rotation correction if enabled
        if enable_rotation and rotation_detector:
            image, angle, confidence = rotation_detector.preprocess(image)
            rotation_stats[angle] += 1
            
            if angle != 0:
                print(f"  🔄 Corrected rotation: {angle}° (confidence: {confidence:.2f})")
        
        # Run detection
        results = model.predict(
            image,
            conf=conf,
            verbose=False,
            imgsz=640,
            max_det=10
        )
        
        # Get detections
        num_boxes = len(results[0].boxes) if results else 0
        total_detections += num_boxes
        
        if num_boxes > 0:
            confidences = [float(box.conf) for box in results[0].boxes]
            avg_conf = sum(confidences) / len(confidences)
            print(f"  ✅ {num_boxes} plate(s) detected (avg conf: {avg_conf:.2f})")
            
            # Draw results
            annotated = results[0].plot()
            
            # Add rotation info
            if enable_rotation and rotation_detector and angle != 0:
                h, w = annotated.shape[:2]
                text = f"Corrected: {angle} degrees"
                cv2.putText(annotated, text, (10, h-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save annotated image
            output_file = output_dir / f"{img_file.stem}_detected{img_file.suffix}"
            cv2.imwrite(str(output_file), annotated)
            
            # Also save corrected image (before detection)
            if enable_rotation and rotation_detector and angle != 0:
                corrected_file = output_dir / f"{img_file.stem}_corrected{img_file.suffix}"
                cv2.imwrite(str(corrected_file), image)
        else:
            print(f"  ℹ️  No plates detected")
    
    # Summary
    print("\n" + "="*60)
    print("📊 DETECTION SUMMARY:")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total plates detected: {total_detections}")
    print(f"Average plates per image: {total_detections/len(image_files):.2f}")
    
    if enable_rotation:
        print("\n🔄 ROTATION SUMMARY:")
        print("-"*60)
        print(f"No rotation (0°):     {rotation_stats[0]}")
        print(f"Rotated 90° CW:       {rotation_stats[90]}")
        print(f"Rotated 180°:         {rotation_stats[180]}")
        print(f"Rotated 270° CW:      {rotation_stats[270]}")
    
    print("\n" + "="*60)
    print("📁 Results saved to:")
    print("="*60)
    print(f"   {output_dir}")
    print("\n   Open the folder to see annotated images!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Test plate detection model with rotation preprocessing'
    )
    parser.add_argument('--model', type=str, 
                       default='runs/plate_detection/yolov11_stage1/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to image or folder (default: validation set)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-rotation', action='store_true',
                       help='Disable rotation correction')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    
    args = parser.parse_args()
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try alternative path
        alt_path = Path('best.pt')
        if alt_path.exists():
            model_path = alt_path
            print(f"ℹ️  Using model: {model_path}")
        else:
            print(f"❌ Model not found: {args.model}")
            
            # Find available models
            print("\nAvailable models:")
            for model in Path('runs/plate_detection').rglob('weights/best.pt'):
                print(f"  - {model}")
            
            if Path('best.pt').exists():
                print(f"  - best.pt")
            
            return
    
    test_images_with_rotation(
        str(model_path),
        args.source,
        args.conf,
        enable_rotation=not args.no_rotation,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
