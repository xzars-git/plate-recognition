#!/usr/bin/env python
"""
🚀 Quick Training Test - 50 Epochs
Test polygon-labeled dataset dengan native rotation
"""

from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    print("="*70)
    print("🚀 QUICK TRAINING TEST - 500 EPOCHS")
    print("="*70)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✅ Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Paths
    dataset_yaml = "dataset/plate_detection_color/plate_color.yaml"
    base_model = "yolo11n.pt"  # Lightweight model untuk quick test
    
    # Verify paths
    if not Path(dataset_yaml).exists():
        print(f"\n❌ ERROR: Dataset YAML not found: {dataset_yaml}")
        return
    
    if not Path(base_model).exists():
        print(f"\n⚠️ WARNING: {base_model} not found, will download...")
    
    print(f"\n📁 Dataset: {dataset_yaml}")
    print(f"🤖 Base Model: {base_model}")
    
    # Load model
    print("\n⏳ Loading model...")
    model = YOLO(base_model)
    
    # Training configuration - QUICK TEST
    print("\n🎯 Training Configuration (Quick Test):")
    print("   • Epochs: 500 (quick test)")
    print("   • Image Size: 640")
    print("   • Batch Size: 16")
    print("   • Workers: 8")
    print("   • Augmentation: MEDIUM (optimized for speed)")
    print("   • Auto-split: 80% train / 20% val")
    
    # Train
    print("\n🔥 Starting training...")
    print("="*70)
    
    results = model.train(
        # Dataset
        data=dataset_yaml,
        
        # Quick test settings
        epochs=500,
        patience=10,  # Early stopping after 10 epochs without improvement
        
        # Image & batch
        imgsz=640,
        batch=16,
        
        # Performance
        device=device,
        workers=8,
        
        # Augmentation (medium untuk quick test)
        degrees=180.0,      # 360° rotation (±180)
        translate=0.1,      # 10% translation
        scale=0.3,          # 30% scale variation
        flipud=0.5,         # Vertical flip 50%
        fliplr=0.5,         # Horizontal flip 50%
        mosaic=0.8,         # Mosaic augmentation 80%
        mixup=0.3,          # Mixup 30% (reduced for speed)
        copy_paste=0.2,     # Copy-paste 20% (reduced for speed)
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,          # Initial learning rate
        lrf=0.01,           # Final learning rate (1% of initial)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,    # Reduced for quick test
        warmup_momentum=0.8,
        
        # Loss
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Validation
        val=True,
        split='train',      # Auto-split from train folder
        fraction=0.8,       # 80% train, 20% val
        
        # Saving
        save=True,
        save_period=10,     # Save checkpoint every 10 epochs
        project='runs/quick_test',
        name='plate_color_50ep',
        exist_ok=True,
        
        # Misc
        plots=True,
        verbose=True,
        seed=42,
        deterministic=False,  # Allow non-deterministic for speed
        
        # Advanced (untuk native rotation support)
        close_mosaic=5,     # Stop mosaic 5 epochs before end
        label_smoothing=0.05,  # Reduced for quick test
        hsv_h=0.015,        # Hue augmentation
        hsv_s=0.7,          # Saturation augmentation
        hsv_v=0.4,          # Value augmentation
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Results
    print(f"\n📊 Best Results:")
    print(f"   • mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   • mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   • Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   • Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}")
    
    # Best model path
    best_model = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"\n💾 Best Model: {best_model}")
    print(f"   Size: {best_model.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n🎯 Next Steps:")
    print("   1. Check training curves in runs/quick_test/plate_color_50ep/")
    print("   2. If good: train full 300 epochs with train_native_rotation.py")
    print("   3. Test model: python test_native_rotation.py")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
