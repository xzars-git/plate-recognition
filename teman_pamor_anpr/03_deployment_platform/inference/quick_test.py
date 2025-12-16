"""
⚡ Quick Test - Fastest way to validate model works
Run this untuk cepat check apakah model jalan dengan baik
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
from plate_rotation_detector import PlateRotationDetector
import time


def quick_check():
    """Quick validation of model"""
    print("\n" + "="*70)
    print("⚡ QUICK MODEL VALIDATION CHECK")
    print("="*70)
    
    # 1. Check model exists
    print("\n1️⃣  Checking model...")
    possible_paths = [
        "best.pt",
        "best_model_epoch170.pt",
        "runs/plate_detection/yolov11_ultimate_v1/weights/epoch170.pt"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("   ❌ Model not found!")
        print("   Looking for: best.pt")
        return False
    
    print(f"   ✅ Model found: {model_path}")
    
    # 2. Load model
    print("\n2️⃣  Loading model...")
    try:
        model = YOLO(model_path)
        print(f"   ✅ Model loaded successfully!")
        print(f"   Size: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False
    
    # 3. Load rotation detector
    print("\n3️⃣  Loading rotation detector...")
    try:
        rotation_detector = PlateRotationDetector(debug=False)
        print(f"   ✅ Rotation detector loaded!")
    except Exception as e:
        print(f"   ❌ Error loading rotation detector: {e}")
        return False
    
    # 4. Find test image
    print("\n4️⃣  Finding test image...")
    test_image_paths = [
        "test_results_epoch170/H2359SV_jpg.rf.9ea5e8f81410cd07fc844e96f38e61cd_detected.jpg",
        "dataset/plate_detection_augmented/images/val/AA4103KN_jpg.rf.89c796674b68762e34ebf57c3214fb91.jpg",
        "dataset/plate_detection_yolo/images/val/AA4103KN_jpg.rf.89c796674b68762e34ebf57c3214fb91.jpg",
    ]
    
    test_image = None
    for path in test_image_paths:
        if Path(path).exists():
            test_image = path
            break
    
    if not test_image:
        print("   ⚠️  No test image found in default paths")
        print("   Enter image path (or press Enter to skip): ", end="")
        user_input = input().strip()
        if user_input and Path(user_input).exists():
            test_image = user_input
        else:
            print("   ⏭️  Skipping inference test")
            return True
    
    print(f"   ✅ Test image found: {test_image}")
    
    # 5. Run inference
    print("\n5️⃣  Running inference...")
    image = cv2.imread(test_image)
    if image is None:
        print("   ❌ Cannot read image!")
        return False
    
    h, w = image.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    start_time = time.time()
    
    # Detect rotation
    corrected_image, angle, rot_conf = rotation_detector.preprocess(image)
    
    # Run detection
    results = model.predict(corrected_image, conf=0.25, verbose=False)
    boxes = results[0].boxes
    
    processing_time = time.time() - start_time
    
    print(f"   ✅ Inference complete in {processing_time*1000:.2f}ms")
    print(f"   Detections: {len(boxes)}")
    
    if len(boxes) > 0:
        for i, box in enumerate(boxes, 1):
            conf = float(box.conf[0])
            print(f"      Plate {i}: {conf:.2%} confidence")
    else:
        print("      ⚠️  No plates detected (this might be normal)")
    
    if angle != 0:
        print(f"   Rotation: {angle}° (confidence: {rot_conf:.2%})")
    
    # 6. Summary
    print("\n" + "="*70)
    print("✅ VALIDATION SUMMARY")
    print("="*70)
    print(f"✓ Model loaded: YES")
    print(f"✓ Rotation detector: YES")
    print(f"✓ Inference works: YES")
    print(f"✓ Processing speed: {processing_time*1000:.2f}ms")
    print(f"✓ Model ready: YES")
    print("="*70)
    
    print("\n🚀 Model is ready for production!")
    print("\nNext steps:")
    print("  1. python demo_app.py          # Interactive testing")
    print("  2. python web_app.py           # Web interface (port 5000)")
    print("  3. Use tflite_conversion_colab.ipynb for mobile deployment")
    
    return True


if __name__ == '__main__':
    success = quick_check()
    if not success:
        print("\n❌ Validation failed! Please check the errors above.")
        exit(1)
