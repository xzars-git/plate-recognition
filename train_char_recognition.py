"""
Training YOLOv11 untuk Stage 2: Character Recognition
Mendeteksi karakter individual (huruf & angka) di plat nomor
"""

from ultralytics import YOLO
import torch
import pandas as pd
from pathlib import Path
import shutil

def prepare_character_dataset():
    """
    Prepare dataset untuk character recognition
    Asumsi: setiap gambar sudah berisi crop plat nomor dengan karakter
    """
    
    print("="*60)
    print("📋 Preparing Character Recognition Dataset...")
    print("="*60)
    
    # Baca label CSV
    label_csv = Path('dataset/plate_text_dataset/label.csv')
    if not label_csv.exists():
        print(f"❌ File not found: {label_csv}")
        return False
    
    df = pd.read_csv(label_csv)
    print(f"\n📊 Total samples: {len(df)}")
    print(f"   Sample labels: {df['label'].head().tolist()}")
    
    # Untuk OCR, kita akan menggunakan model yang berbeda
    # Alternatif 1: Gunakan PaddleOCR, EasyOCR, TrOCR
    # Alternatif 2: Train YOLO untuk detect karakter individual
    # Alternatif 3: Gunakan Tesseract OCR
    
    print("\n💡 REKOMENDASI untuk Character Recognition:")
    print("   Opsi 1: PaddleOCR (RECOMMENDED) - Akurasi tinggi, mudah")
    print("   Opsi 2: EasyOCR - Mudah digunakan")
    print("   Opsi 3: YOLO Character Detection - Perlu anotasi per karakter")
    print("   Opsi 4: TrOCR (Transformer-based) - State-of-the-art")
    
    return True

def main():
    """
    Untuk character recognition, kita akan menggunakan PaddleOCR
    karena lebih cocok untuk OCR task
    """
    
    print("="*60)
    print("🔤 STAGE 2: CHARACTER RECOGNITION")
    print("   Tujuan: Membaca teks dari plat nomor")
    print("="*60)
    
    prepare_character_dataset()
    
    print("\n" + "="*60)
    print("📝 NEXT STEPS:")
    print("="*60)
    print("""
1. Install PaddleOCR:
   pip install paddlepaddle paddleocr

2. Gunakan script detect_anpr_complete.py untuk full pipeline:
   - Stage 1: Detect plate location (YOLO)
   - Stage 2: Read text dari plate (PaddleOCR)

3. Atau gunakan EasyOCR (alternatif):
   pip install easyocr
   
PaddleOCR lebih direkomendasikan karena:
- Akurasi lebih tinggi untuk plat nomor
- Sudah di-train untuk OCR task
- Support banyak bahasa
- Lebih cepat dari Tesseract
""")

if __name__ == '__main__':
    main()
