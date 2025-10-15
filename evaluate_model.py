"""
Evaluasi Model YOLOv11 - Stage 1: Plate Detection
Check performa model setelah training
"""

from ultralytics import YOLO
import torch

def main():
    print("="*60)
    print("📊 MODEL EVALUATION - Stage 1: Plate Detection")
    print("="*60)
    
    # Load trained model
    model_path = 'runs/plate_detection/yolov11_stage1/weights/best.pt'
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded!")
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("🔍 Running validation on test set...")
    print("="*60 + "\n")
    
    results = model.val(
        data='dataset/plate_detection_yolo/plate_detection.yaml',
        batch=8,
        imgsz=640,
        plots=True,
        save_json=True,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS:")
    print("="*60)
    print(f"\n🎯 mAP50: {results.box.map50:.4f}")
    print(f"🎯 mAP50-95: {results.box.map:.4f}")
    print(f"📏 Precision: {results.box.mp:.4f}")
    print(f"📏 Recall: {results.box.mr:.4f}")
    
    # Interpretation
    print("\n" + "="*60)
    print("💡 INTERPRETATION:")
    print("="*60)
    
    if results.box.map50 >= 0.95:
        print("🌟 EXCELLENT! Model sangat akurat!")
    elif results.box.map50 >= 0.85:
        print("✅ GOOD! Model sudah bagus untuk production")
    elif results.box.map50 >= 0.70:
        print("⚠️  DECENT - Bisa dipakai tapi perlu improvement")
    else:
        print("❌ POOR - Perlu training lebih lama atau tuning hyperparameter")
    
    print(f"\nTarget mAP50 untuk plate detection: >= 0.85")
    print(f"Your mAP50: {results.box.map50:.4f}")
    
    # Show plots location
    print("\n" + "="*60)
    print("📁 Results saved to:")
    print("="*60)
    
    # Get save directory from results
    save_dir = results.save_dir if hasattr(results, 'save_dir') else 'runs/detect/val'
    print(f"   {save_dir}")
    
    print("\n   Check plots:")
    print(f"   - confusion_matrix.png")
    print(f"   - F1_curve.png")
    print(f"   - PR_curve.png")
    print(f"   - P_curve.png")
    print(f"   - R_curve.png")
    print("="*60)

if __name__ == '__main__':
    main()
