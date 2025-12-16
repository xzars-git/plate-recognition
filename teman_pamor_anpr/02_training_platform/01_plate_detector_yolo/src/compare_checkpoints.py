"""
Compare All Checkpoints - Find the JACKPOT Model! 🎰
Validates multiple checkpoint files to find the best performing model.
"""

from ultralytics import YOLO
import os
from pathlib import Path

def main():
    print("="*70)
    print("🎰 CHECKPOINT COMPARISON - FINDING THE JACKPOT!")
    print("="*70)

    weights_dir = Path("runs/plate_detection/yolov11_ultimate_v1/weights")
    yaml_path = "dataset/plate_detection_augmented/plate_detection_augmented.yaml"

    # Checkpoints to compare (select strategic epochs)
    checkpoints = [
        "best.pt",
        "last.pt",
        "epoch170.pt",
        "epoch175.pt",
        "epoch180.pt",
        "epoch165.pt",
        "epoch160.pt",
        "epoch155.pt",
        "epoch150.pt",
        "epoch145.pt",
    ]

    results_data = []

    for checkpoint in checkpoints:
        checkpoint_path = weights_dir / checkpoint
        
        if not checkpoint_path.exists():
            print(f"⏭️  Skipping {checkpoint} (not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"🔍 Testing: {checkpoint}")
        print(f"{'='*70}")
        
        try:
            # Load model
            model = YOLO(str(checkpoint_path))
            
            # Run validation (workers=0 to avoid Windows multiprocessing issues)
            results = model.val(
                data=yaml_path,
                workers=0,  # Changed from 1 to 0 for Windows compatibility
                verbose=False
            )
            
            # Extract metrics
            precision = results.box.p[0] if len(results.box.p) > 0 else 0
            recall = results.box.r[0] if len(results.box.r) > 0 else 0
            map50 = results.box.map50
            map50_95 = results.box.map
            speed = results.speed['inference']
            
            results_data.append({
                'checkpoint': checkpoint,
                'precision': precision,
                'recall': recall,
                'map50': map50,
                'map50_95': map50_95,
                'speed': speed,
                'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            })
            
            print(f"✅ {checkpoint:15s} | P: {precision*100:5.2f}% | R: {recall*100:5.2f}% | mAP50: {map50*100:5.2f}% | Speed: {speed:.2f}ms")
            
        except Exception as e:
            print(f"❌ Error testing {checkpoint}: {str(e)}")
            continue

    if not results_data:
        print("\n❌ No checkpoints could be validated!")
        return

    print(f"\n{'='*70}")
    print("🏆 FINAL COMPARISON - ALL CHECKPOINTS")
    print(f"{'='*70}\n")

    # Sort by different metrics
    print("📊 Sorted by PRECISION (Best for ALPR - fewer false positives):")
    print(f"{'Rank':<6} {'Checkpoint':<15} {'Precision':<12} {'Recall':<10} {'mAP50':<10} {'Speed':<10}")
    print("-" * 70)
    sorted_by_precision = sorted(results_data, key=lambda x: x['precision'], reverse=True)
    for i, result in enumerate(sorted_by_precision[:5], 1):
        print(f"{i:<6} {result['checkpoint']:<15} {result['precision']*100:5.2f}%      {result['recall']*100:5.2f}%   {result['map50']*100:5.2f}%   {result['speed']:.2f}ms")

    print(f"\n📊 Sorted by mAP50 (Best overall detection):")
    print(f"{'Rank':<6} {'Checkpoint':<15} {'mAP50':<12} {'Precision':<12} {'Recall':<10} {'Speed':<10}")
    print("-" * 70)
    sorted_by_map50 = sorted(results_data, key=lambda x: x['map50'], reverse=True)
    for i, result in enumerate(sorted_by_map50[:5], 1):
        print(f"{i:<6} {result['checkpoint']:<15} {result['map50']*100:5.2f}%      {result['precision']*100:5.2f}%      {result['recall']*100:5.2f}%   {result['speed']:.2f}ms")

    print(f"\n📊 Sorted by F1 Score (Best balance Precision-Recall):")
    print(f"{'Rank':<6} {'Checkpoint':<15} {'F1 Score':<12} {'Precision':<12} {'Recall':<10} {'Speed':<10}")
    print("-" * 70)
    sorted_by_f1 = sorted(results_data, key=lambda x: x['f1'], reverse=True)
    for i, result in enumerate(sorted_by_f1[:5], 1):
        print(f"{i:<6} {result['checkpoint']:<15} {result['f1']*100:5.2f}%      {result['precision']*100:5.2f}%      {result['recall']*100:5.2f}%   {result['speed']:.2f}ms")

    print(f"\n📊 Sorted by SPEED (Fastest inference):")
    print(f"{'Rank':<6} {'Checkpoint':<15} {'Speed':<12} {'Precision':<12} {'mAP50':<10}")
    print("-" * 70)
    sorted_by_speed = sorted(results_data, key=lambda x: x['speed'])
    for i, result in enumerate(sorted_by_speed[:5], 1):
        print(f"{i:<6} {result['checkpoint']:<15} {result['speed']:.2f}ms       {result['precision']*100:5.2f}%      {result['map50']*100:5.2f}%")

    # Find the JACKPOT (best for ALPR: high precision + good speed)
    print(f"\n{'='*70}")
    print("🎰 JACKPOT RECOMMENDATION FOR ALPR:")
    print(f"{'='*70}")

    # Calculate composite score: precision (70%) + map50 (20%) + speed_factor (10%)
    for result in results_data:
        # Normalize speed (lower is better, convert to 0-1 scale)
        max_speed = max([r['speed'] for r in results_data])
        min_speed = min([r['speed'] for r in results_data])
        speed_norm = 1 - ((result['speed'] - min_speed) / (max_speed - min_speed)) if max_speed != min_speed else 1
        
        # Composite score
        result['composite'] = (result['precision'] * 0.6) + (result['map50'] * 0.3) + (speed_norm * 0.1)

    best_overall = max(results_data, key=lambda x: x['composite'])

    print(f"\n🏆 WINNER: {best_overall['checkpoint']}")
    print(f"   ✅ Precision:  {best_overall['precision']*100:.2f}%")
    print(f"   ✅ Recall:     {best_overall['recall']*100:.2f}%")
    print(f"   ✅ mAP50:      {best_overall['map50']*100:.2f}%")
    print(f"   ✅ mAP50-95:   {best_overall['map50_95']*100:.2f}%")
    print(f"   ✅ F1 Score:   {best_overall['f1']*100:.2f}%")
    print(f"   ✅ Speed:      {best_overall['speed']:.2f}ms ({1000/best_overall['speed']:.1f} FPS)")
    print(f"   ✅ Composite:  {best_overall['composite']*100:.2f}%")

    print(f"\n💡 Why this is JACKPOT:")
    print(f"   - High Precision = Fewer false positives = Less wasted OCR calls")
    print(f"   - Good mAP50 = Reliable detection")
    print(f"   - Fast inference = Real-time ANPR capability")

    print(f"\n📦 Use this model:")
    print(f"   {weights_dir / best_overall['checkpoint']}")

    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()