"""
Quick Test Training - Plate Detection
Untuk verify training bisa jalan tanpa hang
"""

from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model = YOLO('yolo11n.pt')  # Gunakan NANO (paling kecil & cepat)
    
    print("="*60)
    print("🧪 QUICK TEST TRAINING")
    print("   Model: YOLOv11 Nano (tercepat)")
    print("   Epochs: 3 (hanya untuk test)")
    print("   Batch: 4 (kecil)")
    print("="*60)
    
    # Training dengan setting minimal
    results = model.train(
        data='dataset/plate_detection_yolo/plate_detection.yaml',
        epochs=3,                     # Hanya 3 epoch untuk test
        imgsz=320,                    # Image size lebih kecil (lebih cepat)
        batch=4,                      # Batch kecil
        device=device,
        workers=0,                    # HARUS 0 di Windows!
        cache=False,                  # Jangan cache (save RAM)
        
        # Augmentasi minimal
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,                    # No rotation untuk speed
        translate=0.1,
        scale=0.5,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,                   # Kurangi mosaic
        mixup=0.0,                    # No mixup
        
        # Optimizer
        optimizer='SGD',              # SGD lebih cepat dan stabil
        lr0=0.01,
        
        # Output
        project='runs/plate_detection_test',
        name='quick_test',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=False,                  # Skip plots untuk speed
        amp=False,                    # Disable AMP
    )
    
    print("\n" + "="*60)
    print("✅ Test training selesai!")
    print("   Jika ini berhasil, training full akan bisa jalan")
    print("="*60)

if __name__ == '__main__':
    main()
