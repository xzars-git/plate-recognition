"""
Training Script untuk Deteksi Plat Nomor dengan YOLOv11
Dataset: Plat Nomor Jawa Barat
"""

from ultralytics import YOLO
import torch

def main():
    # Cek apakah CUDA tersedia
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training menggunakan: {device}")
    
    # Load YOLOv11 model
    # Pilihan model: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium), 
    #                yolo11l.pt (large), yolo11x.pt (xlarge)
    # Nano paling cepat tapi akurasi lebih rendah, xlarge paling akurat tapi lambat
    model = YOLO('yolo11n.pt')  # Menggunakan YOLOv11 Nano (paling ringan)
    
    # Training parameters
    results = model.train(
        data='plat_jabar.yaml',      # Path ke file konfigurasi dataset
        epochs=100,                   # Jumlah epoch (sesuaikan dengan kebutuhan)
        imgsz=640,                    # Ukuran gambar input
        batch=16,                     # Batch size (sesuaikan dengan VRAM GPU)
        device=device,                # Device untuk training
        patience=20,                  # Early stopping patience
        save=True,                    # Save checkpoint
        save_period=10,               # Save setiap 10 epoch
        
        # Augmentasi data
        hsv_h=0.015,                  # Augmentasi hue
        hsv_s=0.7,                    # Augmentasi saturation
        hsv_v=0.4,                    # Augmentasi value
        degrees=10,                   # Rotasi random
        translate=0.1,                # Translasi random
        scale=0.5,                    # Scaling random
        shear=0.0,                    # Shear
        perspective=0.0,              # Perspective transformation
        flipud=0.0,                   # Flip vertical (0 karena plat nomor tidak terbalik)
        fliplr=0.5,                   # Flip horizontal
        mosaic=1.0,                   # Mosaic augmentation
        mixup=0.0,                    # Mixup augmentation
        
        # Optimizer settings
        optimizer='auto',             # Optimizer (auto, SGD, Adam, AdamW)
        lr0=0.01,                     # Initial learning rate
        lrf=0.01,                     # Final learning rate
        momentum=0.937,               # Momentum
        weight_decay=0.0005,          # Weight decay
        
        # Lainnya
        workers=8,                    # Jumlah worker untuk data loading
        project='runs/detect',        # Folder output
        name='plat_jabar_yolov11',    # Nama experiment
        exist_ok=False,               # Overwrite existing
        pretrained=True,              # Gunakan pretrained weights
        verbose=True,                 # Verbose output
    )
    
    print("\n" + "="*50)
    print("Training selesai!")
    print(f"Model terbaik disimpan di: {results.save_dir}")
    print("="*50)

if __name__ == '__main__':
    main()
