#!/usr/bin/env python
"""
🧪 Test Native Rotation Model
Test model yang sudah belajar semua orientasi secara native (TANPA preprocessing)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

print("="*70)
print("🧪 TESTING: NATIVE ROTATION MODEL")
print("="*70)
print("\n📝 Testing model WITHOUT preprocessing rotation")
print("   Model should detect plates in ANY orientation natively")
print("\n" + "="*70 + "\n")

# ============================================================================
# LOAD MODEL
# ============================================================================

model_path = "best_native_rotation.pt"
if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    print("\n💡 Please train the model first:")
    print("   python train_native_rotation.py")
    exit(1)

print(f"🔧 Loading model: {model_path}")
model = YOLO(model_path)
print(f"✅ Model loaded ({Path(model_path).stat().st_size / (1024*1024):.2f} MB)\n")

# ============================================================================
# TEST IMAGES
# ============================================================================

# Find test images
test_folders = [
    "test_results_epoch170",
    "dataset/plate_detection_augmented/images/val",
    "dataset/plate_detection_yolo/images/val"
]

test_images = []
for folder in test_folders:
    folder_path = Path(folder)
    if folder_path.exists():
        test_images.extend(list(folder_path.glob("*.jpg")))
        test_images.extend(list(folder_path.glob("*.png")))

if not test_images:
    print("❌ No test images found")
    exit(1)

# Select sample images (including rotated ones if available)
sample_images = test_images[:10]

print(f"📸 Found {len(test_images)} test images")
print(f"🎯 Testing on {len(sample_images)} samples\n")

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_image(image_path, model, rotate_angle=0):
    """Test single image with optional rotation"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Rotate if needed (for testing)
    if rotate_angle != 0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Run detection (NO preprocessing!)
    start_time = time.time()
    results = model.predict(image, conf=0.25, verbose=False)
    processing_time = time.time() - start_time
    
    # Get results
    boxes = results[0].boxes
    detections = []
    
    for box in boxes:
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        detections.append({
            'confidence': conf,
            'bbox': xyxy
        })
    
    # Get annotated image
    annotated = results[0].plot()
    
    return {
        'detections': detections,
        'annotated': annotated,
        'processing_time': processing_time
    }

# ============================================================================
# RUN TESTS
# ============================================================================

print("="*70)
print("🧪 RUNNING TESTS (WITHOUT PREPROCESSING)")
print("="*70)

output_dir = Path("native_rotation_test_results")
output_dir.mkdir(exist_ok=True)

all_results = []
angles_to_test = [0, 90, 180, 270]  # Test all orientations

for img_path in sample_images:
    print(f"\n📷 Testing: {img_path.name}")
    
    for angle in angles_to_test:
        result = test_image(img_path, model, rotate_angle=angle)
        
        if result:
            num_detections = len(result['detections'])
            avg_conf = np.mean([d['confidence'] for d in result['detections']]) if result['detections'] else 0
            
            print(f"   Rotation {angle:3d}°: {num_detections} detections, "
                  f"Avg conf: {avg_conf:.2%}, Time: {result['processing_time']*1000:.1f}ms")
            
            # Save annotated image
            output_name = f"{img_path.stem}_rot{angle}_native.jpg"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), result['annotated'])
            
            all_results.append({
                'image': img_path.name,
                'angle': angle,
                'detections': num_detections,
                'confidence': avg_conf,
                'time': result['processing_time']
            })

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "="*70)
print("📊 OVERALL STATISTICS")
print("="*70)

total_tests = len(all_results)
total_detections = sum(r['detections'] for r in all_results)
avg_confidence = np.mean([r['confidence'] for r in all_results if r['detections'] > 0])
avg_time = np.mean([r['time'] for r in all_results])

detection_rate = sum(1 for r in all_results if r['detections'] > 0) / total_tests

print(f"\n📈 Performance:")
print(f"   Total tests: {total_tests}")
print(f"   Total detections: {total_detections}")
print(f"   Detection rate: {detection_rate:.2%}")
print(f"   Avg confidence: {avg_confidence:.2%}")
print(f"   Avg time: {avg_time*1000:.2f}ms")

# Per-angle statistics
print(f"\n🔄 Performance by Rotation:")
for angle in angles_to_test:
    angle_results = [r for r in all_results if r['angle'] == angle]
    angle_detections = sum(r['detections'] for r in angle_results)
    angle_detection_rate = sum(1 for r in angle_results if r['detections'] > 0) / len(angle_results)
    angle_avg_conf = np.mean([r['confidence'] for r in angle_results if r['detections'] > 0]) if angle_detections > 0 else 0
    
    print(f"   {angle:3d}°: Detection rate: {angle_detection_rate:.2%}, "
          f"Avg conf: {angle_avg_conf:.2%}, Detections: {angle_detections}")

print(f"\n💾 Annotated images saved to: {output_dir}/")

print("\n" + "="*70)
print("✅ TESTING COMPLETE!")
print("="*70)

# ============================================================================
# COMPARISON WITH OLD MODEL (if exists)
# ============================================================================

old_model_path = Path("best.pt")
if old_model_path.exists():
    print("\n🔄 COMPARISON WITH OLD MODEL (with preprocessing)")
    print("="*70)
    
    old_model = YOLO(str(old_model_path))
    from plate_rotation_detector import PlateRotationDetector
    rot_detector = PlateRotationDetector(debug=False)
    
    comparison_results = []
    
    for img_path in sample_images[:3]:  # Compare on 3 samples
        print(f"\n📷 {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Test with rotation 90°
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        # Old model (with preprocessing)
        corrected, angle, _ = rot_detector.preprocess(rotated)
        start = time.time()
        old_results = old_model.predict(corrected, conf=0.25, verbose=False)
        old_time = time.time() - start
        old_detections = len(old_results[0].boxes)
        
        # New model (native)
        start = time.time()
        new_results = model.predict(rotated, conf=0.25, verbose=False)
        new_time = time.time() - start
        new_detections = len(new_results[0].boxes)
        
        print(f"   Old model (preprocessing): {old_detections} detections, {old_time*1000:.1f}ms")
        print(f"   New model (native): {new_detections} detections, {new_time*1000:.1f}ms")
        print(f"   Speed improvement: {((old_time - new_time) / old_time * 100):.1f}%")

print("\n🎉 Native rotation model is ready for deployment!")
print("💡 Update desktop_app.py to use 'best_native_rotation.pt'")
