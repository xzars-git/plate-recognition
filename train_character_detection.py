"""
Train YOLOv11 Character Detection Model
For license plate OCR character recognition
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch

def main():
    print("=" * 70)
    print("🏋️ YOLOv11 Character Detection Training")
    print("=" * 70)
    
    # Configuration
    DATA_YAML = 'dataset/character_recognition_yolo/character_detection.yaml'
    MODEL_SIZE = 'yolo11n.pt'  # Options: n, s, m, l, x
    EPOCHS = 1
    BATCH = 16
    IMGSZ = 640
    DEVICE = 'cpu'  # GPU ID (0 for first GPU, 'cpu' for CPU)
    
    # Check if data.yaml exists
    if not Path(DATA_YAML).exists():
        print(f"❌ Error: {DATA_YAML} not found!")
        print("   Please run 'python prepare_character_dataset.py' first.")
        return
    
    # Load data config to verify
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Path: {data_config['path']}")
    print(f"   Train: {data_config['train']}")
    print(f"   Val: {data_config['val']}")
    print(f"   Classes: {data_config['nc']}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️ No GPU available, training on CPU (will be slow)")
        DEVICE = 'cpu'
    
    # Load model
    print(f"\n🤖 Loading model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)
    
    # Training parameters
    print(f"\n⚙️ Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH}")
    print(f"   Image size: {IMGSZ}")
    print(f"   Device: {DEVICE}")
    
    # Start training
    print("\n🚀 Starting training...")
    print("=" * 70)
    
    results = model.train(
        # Data
        data=DATA_YAML,
        
        # Training
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,              # Initial learning rate
        lrf=0.01,               # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Data Augmentation
        # IMPORTANT: Careful with flips for character recognition!
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation augmentation
        hsv_v=0.4,              # HSV-Value augmentation
        degrees=5.0,            # Rotation (small angle, characters sensitive)
        translate=0.1,          # Translation
        scale=0.2,              # Scale
        shear=2.0,              # Shear
        perspective=0.0001,     # Perspective transformation
        flipud=0.0,             # NO vertical flip (breaks characters!)
        fliplr=0.0,             # NO horizontal flip (breaks B/D, P/q, etc)
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation
        copy_paste=0.0,         # Copy-paste augmentation
        
        # Loss weights
        box=7.5,                # Box loss weight
        cls=0.5,                # Class loss weight
        dfl=1.5,                # DFL loss weight
        
        # Validation
        val=True,
        save=True,
        save_period=10,         # Save checkpoint every N epochs
        
        # Performance
        cache=True,             # Cache images in RAM for faster training
        workers=4,              # Number of dataloader workers
        
        # Misc
        project='runs/character_detect',
        name='yolo11n_chars',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        patience=50,            # Early stopping patience
        
        # Advanced
        amp=True,               # Automatic Mixed Precision
        fraction=1.0,           # Train on fraction of data
        profile=False,          # Profile ONNX and TensorRT speeds
        freeze=None,            # Freeze first N layers
        multi_scale=False,      # Multi-scale training (slower)
    )
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Print results
    print(f"\n📊 Training Results:")
    print(f"   Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Final Box Loss: {results.results_dict.get('train/box_loss', 0):.4f}")
    print(f"   Final Class Loss: {results.results_dict.get('train/cls_loss', 0):.4f}")
    
    # Best model path
    best_model = results.save_dir / 'weights' / 'best.pt'
    last_model = results.save_dir / 'weights' / 'last.pt'
    
    print(f"\n💾 Model saved:")
    print(f"   Best: {best_model}")
    print(f"   Last: {last_model}")
    
    # Validation
    print(f"\n🧪 Running validation on best model...")
    model = YOLO(best_model)
    val_results = model.val()
    
    print(f"\n📈 Validation Metrics:")
    print(f"   Precision: {val_results.box.p.mean():.4f}")
    print(f"   Recall: {val_results.box.r.mean():.4f}")
    print(f"   mAP50: {val_results.box.map50:.4f}")
    print(f"   mAP50-95: {val_results.box.map:.4f}")
    
    print("\n🚀 Next Steps:")
    print("   1. Test OCR: python test_yolo_ocr.py")
    print("   2. Update webcam script with new model")
    print("   3. Benchmark performance vs PaddleOCR")
    print("=" * 70)


if __name__ == '__main__':
    main()
