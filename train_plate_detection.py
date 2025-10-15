"""
Training YOLOv11 untuk Stage 1: Plate Detection
Mendeteksi lokasi plat nomor di gambar kendaraan
"""

from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training menggunakan: {device}")
    
    # Load YOLOv11 model
    model = YOLO('yolo11m.pt')  # Medium model untuk balance speed & accuracy
    
    print("="*60)
    print("🚗 STAGE 1: PLATE DETECTION")
    print("   Tujuan: Deteksi lokasi plat nomor di gambar kendaraan")
    print("="*60)
    
    # Training
    results = model.train(
        data='dataset/plate_detection_yolo/plate_detection.yaml',
        epochs=1,                   # Full training: 150 epochs
        imgsz=640,
        batch=8,                      # Batch size 8 (baik untuk RAM)
        device=device,
        patience=30,
        save=True,
        save_period=10,
        cache=False,                  # Jangan cache di RAM (save memory)
        
        # Augmentasi untuk plate detection
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,                   # Rotasi lebih besar (mobil bisa miring)
        translate=0.2,                # Translasi lebih besar
        scale=0.5,
        perspective=0.0005,           # Perspective transform (sudut pandang berbeda)
        flipud=0.0,                   # Tidak flip vertical
        fliplr=0.5,                   # Flip horizontal 50%
        mosaic=1.0,
        mixup=0.1,                    # Mixup untuk generalization
        
        # Optimizer
        optimizer='SGD',              # Gunakan SGD (lebih stabil di Windows)
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Lainnya - CRITICAL untuk Windows
        workers=0,                    # HARUS 0 di Windows untuk stabilitas
        rect=False,                   # Rectangular training off
        close_mosaic=10,              # Close mosaic di 10 epochs terakhir
        project='runs/plate_detection',
        name='yolov11_stage1',
        exist_ok=True,                # Allow overwrite (untuk re-run)
        pretrained=True,
        verbose=True,
        plots=True,                   # Generate plots
        amp=False,                    # Disable AMP di CPU
    )
    
    print("\n" + "="*60)
    print("✅ Stage 1 Training selesai!")
    print(f"   Model terbaik: {results.save_dir}/weights/best.pt")
    print("="*60)

if __name__ == '__main__':
    main()
