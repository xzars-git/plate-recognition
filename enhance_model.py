"""
Enhanced Training Script untuk Meningkatkan Akurasi Model
Menggunakan augmentasi lebih agresif dan hyperparameter tuning
"""

from ultralytics import YOLO
import torch
import os

def train_enhanced_model():
    """Train model dengan konfigurasi enhanced untuk akurasi lebih tinggi"""
    
    print("\n" + "="*70)
    print("🚀 ENHANCED TRAINING - IMPROVED ACCURACY")
    print("="*70)
    
    # Check GPU
    print(f"\n🔍 System Check:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load pretrained model
    print(f"\n📥 Loading YOLOv11n pretrained model...")
    model = YOLO('yolo11n.pt')
    
    print("\n📊 Training Configuration (Enhanced):")
    print("="*70)
    
    # ULTIMATE CONFIGURATION - BEST PERFORMANCE 🏆
    config = {
        # Basic settings - EXTENDED TRAINING
        'data': 'dataset/plate_detection_augmented/plate_detection_augmented.yaml',
        'epochs': 200,  # 🏆 Extended untuk maximum convergence
        'batch': 16,
        'imgsz': 640,
        'device': 0,
        
        # Project settings
        'project': 'runs/plate_detection',
        'name': 'yolov11_ultimate_v1',
        'exist_ok': True,
        
        # Optimization - FINE-TUNED FOR BEST RESULTS 🏆
        'optimizer': 'AdamW',
        'lr0': 0.0012,  # 🏆 Optimal learning rate
        'lrf': 0.005,   # 🏆 Lower final LR untuk fine detail
        'momentum': 0.95,  # 🏆 Higher momentum
        'weight_decay': 0.001,  # 🏆 Proper regularization
        'warmup_epochs': 8.0,  # 🏆 Extended warmup untuk stability
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation - COMPREHENSIVE & INTELLIGENT 🏆
        'hsv_h': 0.025,     # 🏆 Full color variation
        'hsv_s': 0.9,       # 🏆 Strong saturation (different lighting)
        'hsv_v': 0.6,       # 🏆 Brightness range (day/night)
        'degrees': 12.0,    # 🏆 Rotation ±12° (optimal for plates)
        'translate': 0.12,  # 🏆 Translation (camera position)
        'scale': 0.8,       # 🏆 Scale (distance variation)
        'shear': 4.0,       # 🏆 Shear (perspective angles)
        'perspective': 0.0008, # 🏆 Perspective (camera angle)
        'flipud': 0.08,     # 🏆 Vertical flip (plat terbalik)
        'fliplr': 0.5,      # Horizontal flip
        'mosaic': 1.0,      # Mosaic augmentation
        'mixup': 0.12,      # 🏆 Mixup (intelligent blending)
        'copy_paste': 0.08, # 🏆 Copy-paste (more samples)
        'erasing': 0.4,     # 🏆 Random erasing (occlusion)
        'auto_augment': 'randaugment',  # 🏆 Auto augmentation
        
        # Loss weights - OPTIMIZED FOR DETECTION QUALITY 🏆
        'box': 9.0,         # 🏆 MAXIMUM box loss = precise localization
        'cls': 0.5,         # Classification (1 class)
        'dfl': 2.5,         # 🏆 MAXIMUM DFL = ultra-sharp boxes
        
        # Training settings - PATIENCE FOR QUALITY 🏆
        'patience': 50,     # 🏆 Very patient untuk best possible result
        'save': True,
        'save_period': 10,  # 🏆 Save checkpoint setiap 10 epoch
        'cache': True,      # 🏆 RAM cache (32GB cukup dengan optimisasi)
        'workers': 4,       # 🏆 Reduced untuk kurangi memory overhead
        'seed': 42,
        'deterministic': True,
        'plots': True,
        'verbose': True,
        
        # Validation
        'val': True,
        'amp': True,
        'fraction': 1.0,
        
        # Advanced optimization 🏆
        'cos_lr': True,     # Cosine LR scheduler
        'close_mosaic': 20, # 🏆 Close mosaic later (more augmentation)
        'overlap_mask': True,
        'rect': False,      # Rectangle training
        'multi_scale': True, # 🏆 Multi-scale training (robustness)
    }
    
    # Print enhanced features
    print("\n🏆 ULTIMATE CONFIGURATION - BEST PERFORMANCE")
    print("="*70)
    print("\n✨ TRAINING OPTIMIZATION:")
    print("   1. 🏆 200 Epochs - Maximum convergence")
    print("   2. 🏆 Extended warmup (8 epochs) - Stable start")
    print("   3. 🏆 Patience 50 - Won't stop too early")
    print("   4. 🏆 Cache enabled - Fast iteration")
    print("   5. 🏆 Multi-scale training - Robust detection")
    
    print("\n✨ AUGMENTATION STRATEGY:")
    print("   6. 🏆 Comprehensive color augmentation (HSV)")
    print("      • Day/night conditions")
    print("      • Different weather")
    print("      • Various lighting")
    print("   7. 🏆 Geometric transforms:")
    print("      • Rotation ±12° (optimal)")
    print("      • Scale 0.8 (distance variation)")
    print("      • Perspective (camera angles)")
    print("   8. 🏆 Advanced augmentations:")
    print("      • Mixup (intelligent blending)")
    print("      • Copy-paste (more samples)")
    print("      • Random erasing (occlusion)")
    print("      • RandAugment (auto augmentation)")
    
    print("\n✨ LOSS OPTIMIZATION:")
    print("   9. 🏆 Box loss: 9.0 - Precise localization")
    print("   10. 🏆 DFL: 2.5 - Ultra-sharp bounding boxes")
    print("   11. 🏆 Lower final LR (0.005) - Fine details")
    
    print("\n🎯 EXPECTED RESULTS (Conservative Estimates):")
    print("="*70)
    print("   Current → Target:")
    print("   • mAP50:     49% → 58-65% 🎯")
    print("   • mAP50-95:  35% → 42-48% 🎯")
    print("   • Precision: 75% → 82-88% 🎯")
    print("   • Recall:    48% → 60-68% 🎯")
    print("   • Speed:     1.4ms → 0.9-1.2ms 🎯")
    
    print("\n💎 BEST PRACTICES APPLIED:")
    print("   ✅ Proper warmup schedule")
    print("   ✅ Cosine learning rate decay")
    print("   ✅ High patience (quality over speed)")
    print("   ✅ Comprehensive augmentation")
    print("   ✅ Optimized loss weights")
    print("   ✅ Multi-scale robustness")
    print("   ✅ Frequent checkpointing")
    
    print("\n" + "="*70)
    print("🚀 STARTING ULTIMATE TRAINING")
    print("="*70)
    print("\n⏳ Estimasi waktu: ~8-10 jam untuk 200 epochs")
    print("   💡 Tip: Biarkan jalan overnight untuk hasil maksimal")
    print("   Tekan Ctrl+C kapan saja untuk stop training\n")
    
    try:
        # Start training
        results = model.train(**config)
        
        print("\n" + "="*70)
        print("✅ ULTIMATE TRAINING COMPLETED!")
        print("="*70)
        
        # Get best model
        best_model_path = f"runs/plate_detection/yolov11_ultimate_v1/weights/best.pt"
        
        if os.path.exists(best_model_path):
            print(f"\n📦 Best Model: {best_model_path}")
            
            # Validate final model
            print(f"\n🧪 Final Validation:")
            final_model = YOLO(best_model_path)
            metrics = final_model.val()
            
            print(f"\n📈 FINAL RESULTS:")
            print(f"   mAP50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
            print(f"   mAP50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
            print(f"   Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
            print(f"   Recall: {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
            
            # Calculate speed
            speed_ms = metrics.speed['inference']
            fps = 1000 / speed_ms if speed_ms > 0 else 0
            print(f"   Inference Speed: {speed_ms:.2f}ms ({fps:.1f} FPS)")
            
            # Compare with old model
            print(f"\n📊 IMPROVEMENT vs Old Model:")
            print("="*70)
            old_map50 = 0.4905
            old_precision = 0.7524
            old_recall = 0.4788
            old_map = 0.3549
            
            map_improvement = (metrics.box.map50 - old_map50) * 100
            precision_improvement = (metrics.box.mp - old_precision) * 100
            recall_improvement = (metrics.box.mr - old_recall) * 100
            map95_improvement = (metrics.box.map - old_map) * 100
            
            print(f"\n   Metric          Old      New      Δ")
            print(f"   {'─'*45}")
            print(f"   mAP50:       {old_map50*100:5.2f}%  {metrics.box.map50*100:5.2f}%  {map_improvement:+.2f}%")
            print(f"   mAP50-95:    {old_map*100:5.2f}%  {metrics.box.map*100:5.2f}%  {map95_improvement:+.2f}%")
            print(f"   Precision:   {old_precision*100:5.2f}%  {metrics.box.mp*100:5.2f}%  {precision_improvement:+.2f}%")
            print(f"   Recall:      {old_recall*100:5.2f}%  {metrics.box.mr*100:5.2f}%  {recall_improvement:+.2f}%")
            
            # Overall assessment
            print(f"\n🏆 OVERALL ASSESSMENT:")
            total_improvement = (map_improvement + precision_improvement + recall_improvement) / 3
            
            if total_improvement >= 10:
                print(f"   ⭐⭐⭐⭐⭐ EXCELLENT (+{total_improvement:.1f}% avg)")
                print(f"   Model significantly improved!")
            elif total_improvement >= 5:
                print(f"   ⭐⭐⭐⭐ VERY GOOD (+{total_improvement:.1f}% avg)")
                print(f"   Solid improvement across metrics")
            elif total_improvement >= 2:
                print(f"   ⭐⭐⭐ GOOD (+{total_improvement:.1f}% avg)")
                print(f"   Meaningful improvements")
            elif total_improvement >= 0:
                print(f"   ⭐⭐ MODERATE (+{total_improvement:.1f}% avg)")
                print(f"   Some improvements visible")
            else:
                print(f"   ⭐ NEEDS REVIEW ({total_improvement:.1f}% avg)")
                print(f"   Consider checking training logs")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training dihentikan oleh user.")
        print("   Model checkpoint tersimpan di: runs/plate_detection/yolov11_ultimate_v1/weights/")
        print("\n💡 Tip: Kamu bisa resume training dengan:")
        print("   python -c \"from ultralytics import YOLO; YOLO('runs/plate_detection/yolov11_ultimate_v1/weights/last.pt').train(resume=True)\"")
        return None
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_enhancement_tips():
    """Tips cepat untuk enhance model tanpa re-training"""
    
    print("\n" + "="*70)
    print("💡 QUICK ENHANCEMENT TIPS (Tanpa Re-training)")
    print("="*70)
    
    print("\n1. 🔧 TUNE CONFIDENCE THRESHOLD:")
    print("   • Default: 0.25")
    print("   • Coba: 0.3 - 0.5")
    print("   • Precision ↑, Recall mungkin ↓")
    print("   • Code: model.predict(conf=0.4)")
    
    print("\n2. 📏 TUNE IOU THRESHOLD:")
    print("   • Default: 0.7")
    print("   • Coba: 0.5 - 0.6")
    print("   • Lebih permisif untuk overlapping boxes")
    print("   • Code: model.predict(iou=0.5)")
    
    print("\n3. 🔍 TEST TIME AUGMENTATION (TTA):")
    print("   • Predict dengan multiple augmentations")
    print("   • Akurasi ↑ tapi speed ↓")
    print("   • Code: model.predict(augment=True)")
    
    print("\n4. 📐 MULTI-SCALE INFERENCE:")
    print("   • Test dengan berbagai ukuran image")
    print("   • Better untuk deteksi plat kecil/jauh")
    print("   • Code: model.predict(imgsz=[480, 640, 736])")
    
    print("\n5. 🎯 ENSEMBLE MODELS:")
    print("   • Combine predictions dari multiple checkpoints")
    print("   • Gunakan epoch90, epoch80, best.pt")
    print("   • Average predictions")

if __name__ == "__main__":
    import sys
    
    print("\n🎯 MODEL ENHANCEMENT OPTIONS")
    print("="*70)
    print("\n Pilih metode enhancement:")
    print("\n   1. FULL RE-TRAINING (Enhanced) - 6-8 jam")
    print("      → Akurasi terbaik, perlu waktu lama")
    print("\n   2. QUICK TIPS (No training)")
    print("      → Cepat, improve inference saja")
    
    print("\n" + "="*70)
    
    # Check if user wants to see tips first
    if '--tips' in sys.argv or len(sys.argv) == 1:
        quick_enhancement_tips()
        print("\n💡 Untuk mulai enhanced training, run:")
        print("   python enhance_model.py --train")
    
    elif '--train' in sys.argv:
        train_enhanced_model()
    
    else:
        quick_enhancement_tips()
        
        print("\n\n⚡ Mau langsung train enhanced model? (y/n): ", end='')
        try:
            choice = input().lower()
            if choice == 'y':
                train_enhanced_model()
            else:
                print("\n✅ Oke, cek tips di atas dulu!")
        except:
            print("\n\n✅ Run dengan flag --train untuk mulai training")
    
    print("\n" + "="*70)
