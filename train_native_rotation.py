#!/usr/bin/env python
"""
🚀 Train Model dengan Native Rotation Support
Training TANPA preprocessing - model belajar semua orientasi secara native
Menggunakan augmentasi kuat untuk semua angle
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
from datetime import datetime

print("="*70)
print("🎯 TRAINING: NATIVE MULTI-ORIENTATION MODEL")
print("="*70)
print("\n📝 Strategy:")
print("   • NO preprocessing rotation needed")
print("   • Model learns ALL orientations (0°/90°/180°/270°)")
print("   • Heavy augmentation during training")
print("   • Better generalization for real-world scenarios")
print("\n" + "="*70 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_base': 'yolo11n.pt',  # Start from pretrained
    'data_yaml': 'dataset/plate_detection_augmented/plate_detection_augmented.yaml',
    'project': 'runs/plate_detection',
    'name': 'yolov11_native_rotation_v2',
    
    # Training parameters - EXTENDED untuk better learning
    'epochs': 300,  # ⭐ Lebih panjang = better convergence
    'imgsz': 640,
    'batch': 16,
    'patience': 80,  # ⭐ Lebih sabar sebelum early stop
    
    # Optimizer - TUNED
    'optimizer': 'AdamW',
    'lr0': 0.002,  # ⭐ Initial LR sedikit lebih tinggi
    'lrf': 0.001,  # ⭐ Final LR lebih rendah untuk fine-tuning
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,  # ⭐ Warmup untuk stable training
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Loss weights - TUNED untuk plate detection
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    
    # ⭐ AUGMENTATION - VERY AGGRESSIVE untuk rotation + quality
    'augmentation': {
        # Rotation augmentation (FULL 360°)
        'degrees': 360.0,  # Full rotation
        
        # Flip augmentation
        'flipud': 0.5,     
        'fliplr': 0.5,     
        
        # Scale & translate - AGGRESSIVE
        'scale': 0.7,      # ⭐ Lebih aggressive scale
        'translate': 0.3,  # ⭐ Lebih aggressive translate
        'shear': 10.0,     # ⭐ Lebih aggressive shear
        
        # Color augmentation - STRONG
        'hsv_h': 0.025,    # ⭐ Lebih banyak hue variation
        'hsv_s': 0.8,      # ⭐ Lebih banyak saturation variation
        'hsv_v': 0.6,      # ⭐ Lebih banyak brightness variation
        
        # Mosaic & mixup - ENABLED
        'mosaic': 1.0,     
        'mixup': 0.5,      # ⭐ Lebih banyak mixup
        'copy_paste': 0.3, # ⭐ Copy-paste augmentation
        
        # Spatial augmentation
        'perspective': 0.0005,  # ⭐ Perspective transform
        'erasing': 0.5,         # ⭐ Random erasing lebih aggressive
        'crop_fraction': 0.9,   # ⭐ Slight crop
        
        # Advanced augmentation
        'auto_augment': 'randaugment',  # ⭐ AutoAugment policy
        'bgr': 0.0,  # BGR channel flip (for robustness)
    },
    
    # ⭐ MULTI-SCALE TRAINING untuk better generalization
    'rect': False,  # Rectangular training (disable untuk rotation)
    'multi_scale': True,  # ⭐ Multi-scale training
    
    # Validation
    'val': True,
    'save': True,
    'save_period': 10,
    'plots': True,
    'verbose': True,
    
    # ⭐ LABEL SMOOTHING untuk better generalization
    'label_smoothing': 0.1,  # Prevent overconfidence
    
    # ⭐ NMS PARAMETERS untuk better detection
    'nms': True,
    'iou': 0.7,  # IoU threshold untuk NMS
    'max_det': 100,  # Max detections per image
    
    # ⭐ CLOSE MOSAIC untuk final fine-tuning
    'close_mosaic': 30,  # Disable mosaic di 30 epoch terakhir untuk fine-tune
}

# ============================================================================
# CHECK DATA
# ============================================================================

print("📂 Checking dataset...")
data_path = Path(CONFIG['data_yaml'])

if not data_path.exists():
    print(f"❌ Error: Data YAML not found: {data_path}")
    print("\n💡 Make sure you have augmented dataset:")
    print("   Run: python augment_dataset_rotation.py")
    exit(1)

with open(data_path) as f:
    data_config = yaml.safe_load(f)

print(f"✅ Dataset: {data_path}")
print(f"   Train: {data_config.get('train', 'N/A')}")
print(f"   Val: {data_config.get('val', 'N/A')}")
print(f"   Classes: {data_config.get('names', [])}")
print()

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"🔧 Loading base model: {CONFIG['model_base']}")
model = YOLO(CONFIG['model_base'])
print("✅ Model loaded\n")

# ============================================================================
# START TRAINING
# ============================================================================

print("="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Target: Native multi-orientation detection")
print(f"📊 Epochs: {CONFIG['epochs']}")
print(f"🔄 Rotation: 0-360° random augmentation")
print(f"💾 Output: {CONFIG['project']}/{CONFIG['name']}")
print("="*70 + "\n")

# Train
results = model.train(
    # Data
    data=CONFIG['data_yaml'],
    
    # Training config
    epochs=CONFIG['epochs'],
    imgsz=CONFIG['imgsz'],
    batch=CONFIG['batch'],
    patience=CONFIG['patience'],
    
    # Optimizer
    optimizer=CONFIG['optimizer'],
    lr0=CONFIG['lr0'],
    lrf=CONFIG['lrf'],
    momentum=CONFIG['momentum'],
    weight_decay=CONFIG['weight_decay'],
    warmup_epochs=CONFIG['warmup_epochs'],
    warmup_momentum=CONFIG['warmup_momentum'],
    warmup_bias_lr=CONFIG['warmup_bias_lr'],
    
    # Loss weights
    box=CONFIG['box'],
    cls=CONFIG['cls'],
    dfl=CONFIG['dfl'],
    
    # Augmentation (AGGRESSIVE ROTATION + MORE)
    degrees=CONFIG['augmentation']['degrees'],
    flipud=CONFIG['augmentation']['flipud'],
    fliplr=CONFIG['augmentation']['fliplr'],
    scale=CONFIG['augmentation']['scale'],
    translate=CONFIG['augmentation']['translate'],
    shear=CONFIG['augmentation']['shear'],
    hsv_h=CONFIG['augmentation']['hsv_h'],
    hsv_s=CONFIG['augmentation']['hsv_s'],
    hsv_v=CONFIG['augmentation']['hsv_v'],
    mosaic=CONFIG['augmentation']['mosaic'],
    mixup=CONFIG['augmentation']['mixup'],
    copy_paste=CONFIG['augmentation']['copy_paste'],
    perspective=CONFIG['augmentation']['perspective'],
    erasing=CONFIG['augmentation']['erasing'],
    crop_fraction=CONFIG['augmentation']['crop_fraction'],
    auto_augment=CONFIG['augmentation']['auto_augment'],
    bgr=CONFIG['augmentation']['bgr'],
    
    # Multi-scale & regularization
    rect=CONFIG['rect'],
    label_smoothing=CONFIG['label_smoothing'],
    nms=CONFIG['nms'],
    iou=CONFIG['iou'],
    max_det=CONFIG['max_det'],
    close_mosaic=CONFIG['close_mosaic'],
    
    # Output
    project=CONFIG['project'],
    name=CONFIG['name'],
    exist_ok=True,
    
    # Validation & saving
    val=CONFIG['val'],
    save=CONFIG['save'],
    save_period=CONFIG['save_period'],
    plots=CONFIG['plots'],
    verbose=CONFIG['verbose'],
    
    # Device
    device=0,  # Use GPU 0
)

# ============================================================================
# TRAINING COMPLETE
# ============================================================================

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💾 Model saved to: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
print()

# Validate
print("🔍 Running final validation...")
metrics = model.val()

print("\n📊 FINAL METRICS:")
print(f"   Precision: {metrics.box.p.mean():.4f}")
print(f"   Recall: {metrics.box.r.mean():.4f}")
print(f"   mAP50: {metrics.box.map50:.4f}")
print(f"   mAP50-95: {metrics.box.map:.4f}")

# Copy best model to root
best_model_src = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'
best_model_dst = Path('best_native_rotation.pt')

if best_model_src.exists():
    shutil.copy(best_model_src, best_model_dst)
    print(f"\n✅ Best model copied to: {best_model_dst}")
    print(f"   Size: {best_model_dst.stat().st_size / (1024*1024):.2f} MB")

print("\n" + "="*70)
print("🎉 NATIVE ROTATION MODEL READY!")
print("="*70)
print("\n💡 Next steps:")
print("   1. Test model: python test_native_rotation.py")
print("   2. Compare with old model: python compare_models.py")
print("   3. Use in desktop app: Update model path to 'best_native_rotation.pt'")
print("\n🔥 NO PREPROCESSING NEEDED - Model handles all rotations natively!")
print()
