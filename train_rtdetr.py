"""
Training Script untuk Deteksi Plat Nomor dengan RT-DETR
RT-DETR = Real-Time Detection Transformer (lebih akurat dari YOLO)
"""

from ultralytics import RTDETR
import torch

def main():
    # Cek apakah CUDA tersedia
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training menggunakan: {device}")
    
    # Load RT-DETR model
    # Pilihan model: rtdetr-l.pt (large), rtdetr-x.pt (xlarge)
    model = RTDETR('rtdetr-l.pt')  # RT-DETR Large
    
    print("="*60)
    print("🚀 RT-DETR - Real-Time Detection Transformer")
    print("   Lebih akurat dari YOLO dengan kecepatan hampir sama")
    print("="*60)
    
    # Training parameters
    results = model.train(
        data='plat_jabar.yaml',
        epochs=100,
        imgsz=640,
        batch=8,                      # RT-DETR butuh memory lebih, pakai batch lebih kecil
        device=device,
        patience=20,
        save=True,
        save_period=10,
        
        # Augmentasi
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        
        # Optimizer
        optimizer='AdamW',            # RT-DETR biasanya pakai AdamW
        lr0=0.0001,                   # Learning rate lebih kecil untuk Transformer
        lrf=0.01,
        weight_decay=0.0001,
        
        # Lainnya
        workers=8,
        project='runs/detect',
        name='plat_jabar_rtdetr',
        exist_ok=False,
        pretrained=True,
        verbose=True,
    )
    
    print("\n" + "="*50)
    print("Training selesai!")
    print(f"Model terbaik disimpan di: {results.save_dir}")
    print("="*50)

if __name__ == '__main__':
    main()
