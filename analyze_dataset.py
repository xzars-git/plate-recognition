"""
Analisis Dataset untuk Deteksi Plat Nomor
Script ini akan menganalisis dataset dan menampilkan statistik
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset(dataset_path='dataset'):
    """Analisis dataset"""
    dataset_path = Path(dataset_path)
    
    print("="*60)
    print("ANALISIS DATASET PLAT NOMOR JAWA BARAT")
    print("="*60)
    
    # Hitung jumlah gambar
    train_images = list((dataset_path / 'images' / 'train').glob('*'))
    val_images = list((dataset_path / 'images' / 'val').glob('*'))
    
    # Hitung jumlah label
    train_labels = list((dataset_path / 'labels' / 'train').glob('*.txt'))
    val_labels = list((dataset_path / 'labels' / 'val').glob('*.txt'))
    
    # Filter out classes.txt
    train_labels = [f for f in train_labels if f.name != 'classes.txt']
    val_labels = [f for f in val_labels if f.name != 'classes.txt']
    
    print(f"\n📊 STATISTIK DATASET:")
    print(f"   Training Images: {len(train_images)}")
    print(f"   Training Labels: {len(train_labels)}")
    print(f"   Validation Images: {len(val_images)}")
    print(f"   Validation Labels: {len(val_labels)}")
    print(f"   Total Images: {len(train_images) + len(val_images)}")
    
    # Analisis ukuran gambar
    if train_images:
        from PIL import Image
        sample_img = Image.open(train_images[0])
        print(f"\n📐 UKURAN GAMBAR (sample):")
        print(f"   Width x Height: {sample_img.size[0]} x {sample_img.size[1]}")
    
    # Analisis bounding boxes
    print(f"\n📦 ANALISIS BOUNDING BOXES:")
    total_boxes = 0
    box_sizes = []
    
    for label_file in train_labels[:100]:  # Analisis 100 file pertama
        with open(label_file, 'r') as f:
            lines = f.readlines()
            total_boxes += len(lines)
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    box_sizes.append((w, h))
    
    if box_sizes:
        avg_width = sum(w for w, h in box_sizes) / len(box_sizes)
        avg_height = sum(h for w, h in box_sizes) / len(box_sizes)
        print(f"   Rata-rata lebar box: {avg_width:.3f}")
        print(f"   Rata-rata tinggi box: {avg_height:.3f}")
        print(f"   Total boxes (sample 100 files): {total_boxes}")
    
    # Rekomendasi
    print(f"\n💡 REKOMENDASI TRAINING:")
    total_images = len(train_images) + len(val_images)
    
    if total_images < 100:
        print("   ⚠️  Dataset kecil (<100 gambar)")
        print("   → Gunakan augmentasi data yang agresif")
        print("   → Training dengan epochs lebih banyak (150-200)")
        print("   → Gunakan model kecil (yolo11n.pt atau yolo11s.pt)")
    elif total_images < 500:
        print("   ✓ Dataset sedang (100-500 gambar)")
        print("   → Augmentasi data standar sudah cukup")
        print("   → Training 100-150 epochs")
        print("   → Gunakan yolo11s.pt atau yolo11m.pt")
    else:
        print("   ✓✓ Dataset besar (>500 gambar)")
        print("   → Bisa gunakan augmentasi minimal")
        print("   → Training 80-100 epochs")
        print("   → Bisa gunakan model besar (yolo11m.pt atau yolo11l.pt)")
    
    print(f"\n🎯 PARAMETER YANG DISARANKAN:")
    print(f"   Image size: 640 (standar)")
    print(f"   Batch size: 16 (jika GPU 8GB+), 8 (jika GPU 4GB), 4 (jika GPU 2GB)")
    print(f"   Learning rate: 0.01 (default)")
    
    print("="*60)

if __name__ == '__main__':
    analyze_dataset()
