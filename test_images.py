"""
Test Model dengan Gambar - Stage 1: Plate Detection
Test deteksi plat nomor pada gambar
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse

def test_images(model_path, image_path=None, conf=0.25):
    """
    Test model pada gambar
    
    Args:
        model_path: Path ke model (.pt file)
        image_path: Path ke gambar atau folder (None = pakai validation set)
        conf: Confidence threshold
    """
    print("="*60)
    print("🖼️  IMAGE TESTING - Plate Detection")
    print("="*60)
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded!")
    
    # Determine source
    if image_path is None:
        # Use validation images
        source = 'dataset/plate_detection_yolo/images/val'
        print(f"\n📁 Testing on validation set: {source}")
    else:
        source = image_path
        print(f"\n📁 Testing on: {source}")
    
    # Run prediction
    print(f"\n🔍 Running detection (conf={conf})...")
    print("="*60 + "\n")
    
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
        imgsz=640,
        max_det=10,  # Max 10 plates per image
        project='runs/detect',
        name='test_images',
        exist_ok=True
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 DETECTION SUMMARY:")
    print("="*60)
    
    total_detections = 0
    for i, result in enumerate(results):
        num_boxes = len(result.boxes)
        total_detections += num_boxes
        if num_boxes > 0:
            confidences = [float(box.conf) for box in result.boxes]
            avg_conf = sum(confidences) / len(confidences)
            print(f"Image {i+1}: {num_boxes} plate(s) detected (avg conf: {avg_conf:.2f})")
    
    print(f"\n✅ Total images processed: {len(results)}")
    print(f"✅ Total plates detected: {total_detections}")
    print(f"✅ Average plates per image: {total_detections/len(results):.2f}")
    
    # Results location
    save_dir = results[0].save_dir if results else 'runs/detect/test_images'
    print("\n" + "="*60)
    print("📁 Results saved to:")
    print("="*60)
    print(f"   {save_dir}")
    print("\n   Open the folder to see annotated images!")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Test plate detection model on images')
    parser.add_argument('--model', type=str, 
                       default='runs/plate_detection/yolov11_stage1/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to image or folder (default: validation set)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        
        # Find available models
        print("\nAvailable models:")
        for model in Path('runs/plate_detection').rglob('weights/best.pt'):
            print(f"  - {model}")
        return
    
    test_images(args.model, args.source, args.conf)

if __name__ == '__main__':
    main()
