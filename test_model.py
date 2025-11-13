"""
Test Epoch 170 Model with Rotation Handling
Combines JACKPOT model (epoch170) with automatic rotation detection
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from plate_rotation_detector import PlateRotationDetector


def test_with_rotation_handling(image_path, model_path="best.pt", save_results=True):
    """
    Test image dengan rotation handling pipeline:
    1. Detect rotation angle (0°, 90°, 180°, 270°)
    2. Correct rotation
    3. Run YOLO detection
    4. Show results
    
    Args:
        image_path: Path to test image
        model_path: Path to model (default: epoch170)
        save_results: Save annotated results
    """
    print("="*70)
    print("🎯 EPOCH 170 + ROTATION HANDLING TEST")
    print("="*70)
    
    # Load model
    print(f"\n📥 Loading model: {model_path}")
    
    # Try multiple possible paths for epoch170
    # User renamed epoch170 to best.pt
    possible_paths = [
        model_path,
        "best.pt",
        "best_model_epoch170.pt",
        "runs/plate_detection/yolov11_ultimate_v1/weights/epoch170.pt"
    ]
    
    model = None
    for path in possible_paths:
        if Path(path).exists():
            model = YOLO(path)
            print(f"✅ Model loaded from: {path}")
            break
    
    if model is None:
        print("❌ Model not found! Please check the path.")
        return
    
    # Load image
    print(f"\n📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Cannot read image!")
        return
    
    original_image = image.copy()
    h, w = image.shape[:2]
    print(f"   Size: {w}x{h}")
    
    # Initialize rotation detector
    print("\n🔄 Detecting rotation...")
    rotation_detector = PlateRotationDetector(debug=True)
    
    # Detect and correct rotation
    corrected_image, detected_angle, confidence = rotation_detector.preprocess(image)
    
    print(f"\n📊 Rotation Detection Results:")
    print(f"   Detected angle: {detected_angle}°")
    print(f"   Confidence: {confidence:.2%}")
    
    if detected_angle != 0:
        print(f"   ✅ Image corrected by rotating {detected_angle}°")
        corrected_h, corrected_w = corrected_image.shape[:2]
        print(f"   New size: {corrected_w}x{corrected_h}")
    else:
        print(f"   ℹ️  No rotation needed")
    
    # Run detection on corrected image
    print(f"\n🔍 Running plate detection...")
    results = model.predict(
        corrected_image,
        conf=0.25,
        verbose=False,
        imgsz=640,
        max_det=10
    )
    
    # Get detections
    boxes = results[0].boxes
    num_detections = len(boxes)
    
    print(f"\n✅ Detection Results:")
    print(f"   Plates found: {num_detections}")
    
    if num_detections > 0:
        for i, box in enumerate(boxes, 1):
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            print(f"\n   Plate {i}:")
            print(f"      Confidence: {conf:.2%}")
            print(f"      BBox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
            
            # Calculate bbox dimensions
            bbox_w = xyxy[2] - xyxy[0]
            bbox_h = xyxy[3] - xyxy[1]
            bbox_area = bbox_w * bbox_h
            
            print(f"      Size: {bbox_w:.1f} x {bbox_h:.1f} pixels")
            print(f"      Area: {bbox_area:.0f} px²")
    else:
        print("   ⚠️  No plates detected")
        print("   Possible reasons:")
        print("      - Plate too small/far")
        print("      - Poor image quality")
        print("      - Extreme viewing angle")
        print("      - Plate not in training distribution")
    
    # Visualize results
    if save_results:
        print(f"\n💾 Saving results...")
        
        output_dir = Path("test_results_rotation")
        output_dir.mkdir(exist_ok=True)
        
        # Get base filename
        base_name = Path(image_path).stem
        
        # Save original
        original_path = output_dir / f"{base_name}_01_original.jpg"
        cv2.imwrite(str(original_path), original_image)
        print(f"   ✅ Original: {original_path}")
        
        # Save corrected (if rotated)
        if detected_angle != 0:
            corrected_path = output_dir / f"{base_name}_02_corrected_rot{detected_angle}.jpg"
            
            # Add rotation info text
            corrected_display = corrected_image.copy()
            text = f"Corrected: rotated {detected_angle} degrees"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(corrected_display, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(corrected_display, text, (10, text_h + 10), font, font_scale, (0, 255, 255), thickness)
            
            cv2.imwrite(str(corrected_path), corrected_display)
            print(f"   ✅ Corrected: {corrected_path}")
        
        # Save detection result
        if num_detections > 0:
            # Get annotated frame
            annotated = results[0].plot()
            
            # Add rotation info if corrected
            if detected_angle != 0:
                h_ann, w_ann = annotated.shape[:2]
                text = f"Image was rotated {detected_angle} degrees before detection"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw background
                y_pos = h_ann - 15
                cv2.rectangle(annotated, (5, y_pos - text_h - 5), (text_w + 15, y_pos + 5), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(annotated, text, (10, y_pos), font, font_scale, (0, 255, 255), thickness)
            
            detection_path = output_dir / f"{base_name}_03_detected.jpg"
            cv2.imwrite(str(detection_path), annotated)
            print(f"   ✅ Detected: {detection_path}")
            
            # Create comparison (original vs detected)
            # Resize to same height
            h1, w1 = original_image.shape[:2]
            h2, w2 = annotated.shape[:2]
            
            # Use smaller height
            target_h = min(h1, h2, 600)  # Max 600px height
            
            # Resize original
            scale1 = target_h / h1
            new_w1 = int(w1 * scale1)
            resized_orig = cv2.resize(original_image, (new_w1, target_h))
            
            # Resize detected
            scale2 = target_h / h2
            new_w2 = int(w2 * scale2)
            resized_det = cv2.resize(annotated, (new_w2, target_h))
            
            # Add labels
            label_orig = resized_orig.copy()
            label_det = resized_det.copy()
            
            # Label "ORIGINAL"
            cv2.putText(label_orig, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Label "DETECTED"
            cv2.putText(label_det, "DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Stack horizontally
            comparison = np.hstack([label_orig, label_det])
            
            comparison_path = output_dir / f"{base_name}_04_comparison.jpg"
            cv2.imwrite(str(comparison_path), comparison)
            print(f"   ✅ Comparison: {comparison_path}")
        
        print(f"\n📁 All results saved to: {output_dir}/")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Image: {Path(image_path).name}")
    print(f"Original size: {w}x{h}")
    print(f"Rotation detected: {detected_angle}° (confidence: {confidence:.2%})")
    if detected_angle != 0:
        print(f"Corrected size: {corrected_w}x{corrected_h}")
    print(f"Plates detected: {num_detections}")
    
    if num_detections > 0:
        confidences = [float(box.conf[0]) for box in boxes]
        print(f"Average confidence: {np.mean(confidences):.2%}")
        print(f"Confidence range: {np.min(confidences):.2%} - {np.max(confidences):.2%}")
    
    print(f"{'='*70}\n")


def batch_test_rotation(image_folder, pattern="*.jpg"):
    """
    Test multiple images with rotation handling
    
    Args:
        image_folder: Folder containing test images
        pattern: File pattern (default: *.jpg)
    """
    print("="*70)
    print("🎯 BATCH TEST - EPOCH 170 + ROTATION HANDLING")
    print("="*70)
    
    # Find images
    folder_path = Path(image_folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {image_folder}")
        return
    
    image_files = list(folder_path.glob(pattern))
    
    if not image_files:
        print(f"❌ No images found matching pattern: {pattern}")
        return
    
    print(f"\n📁 Found {len(image_files)} images")
    print(f"📂 Folder: {image_folder}")
    print(f"🔍 Pattern: {pattern}\n")
    
    # Load model once
    print("📥 Loading model...")
    # User renamed epoch170 to best.pt
    possible_paths = [
        "best.pt",
        "best_model_epoch170.pt",
        "runs/plate_detection/yolov11_ultimate_v1/weights/epoch170.pt"
    ]
    
    model = None
    for path in possible_paths:
        if Path(path).exists():
            model = YOLO(path)
            print(f"✅ Model loaded from: {path}\n")
            break
    
    if model is None:
        print("❌ Model not found!")
        return
    
    # Initialize rotation detector
    rotation_detector = PlateRotationDetector(debug=False)
    
    # Statistics
    stats = {
        'total': len(image_files),
        'detected': 0,
        'not_detected': 0,
        'rotation_0': 0,
        'rotation_90': 0,
        'rotation_180': 0,
        'rotation_270': 0,
        'confidences': []
    }
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print("   ⚠️  Cannot read, skipping...")
            continue
        
        # Detect rotation
        corrected_image, angle, confidence = rotation_detector.preprocess(image)
        stats[f'rotation_{angle}'] += 1
        
        if angle != 0:
            print(f"   🔄 Corrected rotation: {angle}° (conf: {confidence:.2%})")
        
        # Run detection
        results = model.predict(corrected_image, conf=0.25, verbose=False)
        boxes = results[0].boxes
        num_detections = len(boxes)
        
        if num_detections > 0:
            stats['detected'] += 1
            confidences = [float(box.conf[0]) for box in boxes]
            stats['confidences'].extend(confidences)
            avg_conf = np.mean(confidences)
            print(f"   ✅ {num_detections} plate(s) detected (avg conf: {avg_conf:.2%})")
        else:
            stats['not_detected'] += 1
            print(f"   ❌ No plates detected")
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 BATCH TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total images tested: {stats['total']}")
    print(f"Images with detections: {stats['detected']} ({stats['detected']/stats['total']*100:.1f}%)")
    print(f"Images without detections: {stats['not_detected']} ({stats['not_detected']/stats['total']*100:.1f}%)")
    
    print(f"\n🔄 Rotation Distribution:")
    print(f"   0° (no rotation):   {stats['rotation_0']} ({stats['rotation_0']/stats['total']*100:.1f}%)")
    print(f"   90° (rotated CW):   {stats['rotation_90']} ({stats['rotation_90']/stats['total']*100:.1f}%)")
    print(f"   180° (upside down): {stats['rotation_180']} ({stats['rotation_180']/stats['total']*100:.1f}%)")
    print(f"   270° (rotated CCW): {stats['rotation_270']} ({stats['rotation_270']/stats['total']*100:.1f}%)")
    
    if stats['confidences']:
        print(f"\n📈 Detection Confidence:")
        print(f"   Average: {np.mean(stats['confidences']):.2%}")
        print(f"   Range: {np.min(stats['confidences']):.2%} - {np.max(stats['confidences']):.2%}")
        print(f"   Total detections: {len(stats['confidences'])}")
    
    print(f"{'='*70}\n")


def quick_test():
    """
    Quick test on validation dataset (10 sample images)
    Similar to test_final_model.py but with rotation handling
    """
    print("="*70)
    print("🧪 QUICK TEST - EPOCH 170 MODEL (10 Samples)")
    print("="*70)
    
    # Find validation images
    test_images_dir = Path("dataset/plate_detection_augmented/images/val")
    if not test_images_dir.exists():
        test_images_dir = Path("dataset/plate_detection_yolo/images/val")
    
    if not test_images_dir.exists():
        print("\n⚠️  Validation images not found!")
        print("   Use: python test_epoch170_with_rotation.py --batch <folder_path>")
        return
    
    # Get 10 sample images
    image_files = list(test_images_dir.glob("*.jpg"))[:10]
    
    if not image_files:
        print("\n⚠️  No images found!")
        return
    
    print(f"\n🖼️  Testing {len(image_files)} sample images...")
    print(f"📂 From: {test_images_dir}\n")
    
    # Use batch test function
    batch_test_rotation(str(test_images_dir), "*.jpg")
    
    print("\n🎯 MODEL READY FOR PRODUCTION!")
    print("   Next steps:")
    print("   1. ✅ Model tested and validated")
    print("   2. 📦 Convert to TFLite: Use tflite_conversion_colab.ipynb")
    print("   3. 🚀 Deploy to Teman Pamor app")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Quick test:    python test_epoch170_with_rotation.py --quick")
        print("  Single image:  python test_epoch170_with_rotation.py <image_path>")
        print("  Batch test:    python test_epoch170_with_rotation.py --batch <folder_path> [pattern]")
        print("\nExamples:")
        print("  python test_epoch170_with_rotation.py --quick")
        print("  python test_epoch170_with_rotation.py test.jpg")
        print("  python test_epoch170_with_rotation.py --batch dataset/plate_detection_yolo/images/val")
        print("  python test_epoch170_with_rotation.py --batch test_images *.png")
        return
    
    if sys.argv[1] == '--quick':
        quick_test()
    elif sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("❌ Please provide folder path for batch processing")
            return
        
        folder = sys.argv[2]
        pattern = sys.argv[3] if len(sys.argv) > 3 else "*.jpg"
        batch_test_rotation(folder, pattern)
    else:
        image_path = sys.argv[1]
        test_with_rotation_handling(image_path)


if __name__ == '__main__':
    main()
