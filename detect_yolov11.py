"""
Inference Script untuk Deteksi Plat Nomor dengan YOLOv11
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

def detect_image(model_path, image_path, save_dir='runs/detect/predict'):
    """
    Deteksi plat nomor pada satu gambar
    
    Args:
        model_path: Path ke model yang sudah ditraining
        image_path: Path ke gambar yang akan dideteksi
        save_dir: Direktori untuk menyimpan hasil
    """
    # Load model
    model = YOLO(model_path)
    
    # Prediksi
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,           # Save hasil dalam format txt
        save_conf=True,          # Save confidence score
        conf=0.25,               # Confidence threshold
        iou=0.45,                # IoU threshold untuk NMS
        project=save_dir,
        name='results',
        exist_ok=True,
    )
    
    return results

def detect_video(model_path, video_path, save_dir='runs/detect/predict'):
    """
    Deteksi plat nomor pada video
    
    Args:
        model_path: Path ke model yang sudah ditraining
        video_path: Path ke video yang akan dideteksi
        save_dir: Direktori untuk menyimpan hasil
    """
    # Load model
    model = YOLO(model_path)
    
    # Prediksi pada video
    results = model.predict(
        source=video_path,
        save=True,
        conf=0.25,
        iou=0.45,
        project=save_dir,
        name='video_results',
        exist_ok=True,
    )
    
    return results

def detect_webcam(model_path):
    """
    Deteksi plat nomor secara real-time dari webcam
    
    Args:
        model_path: Path ke model yang sudah ditraining
    """
    # Load model
    model = YOLO(model_path)
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    
    print("Tekan 'q' untuk keluar")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prediksi
        results = model(frame, conf=0.25)
        
        # Visualisasi hasil
        annotated_frame = results[0].plot()
        
        # Tampilkan
        cv2.imshow('YOLOv11 - Deteksi Plat Nomor', annotated_frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Path ke model terbaik hasil training
    # Ganti dengan path model Anda
    model_path = 'runs/detect/plat_jabar_yolov11/weights/best.pt'
    
    print("="*50)
    print("YOLOv11 - Deteksi Plat Nomor Jawa Barat")
    print("="*50)
    print("\nPilih mode:")
    print("1. Deteksi pada gambar")
    print("2. Deteksi pada video")
    print("3. Deteksi real-time (webcam)")
    print("4. Keluar")
    
    choice = input("\nPilihan Anda (1-4): ")
    
    if choice == '1':
        image_path = input("Masukkan path gambar: ")
        if Path(image_path).exists():
            print(f"\nMemproses gambar: {image_path}")
            results = detect_image(model_path, image_path)
            print(f"Selesai! Hasil disimpan di: runs/detect/predict/results")
        else:
            print("File tidak ditemukan!")
    
    elif choice == '2':
        video_path = input("Masukkan path video: ")
        if Path(video_path).exists():
            print(f"\nMemproses video: {video_path}")
            results = detect_video(model_path, video_path)
            print(f"Selesai! Hasil disimpan di: runs/detect/predict/video_results")
        else:
            print("File tidak ditemukan!")
    
    elif choice == '3':
        print("\nMemulai deteksi real-time...")
        detect_webcam(model_path)
    
    elif choice == '4':
        print("Keluar...")
    
    else:
        print("Pilihan tidak valid!")

if __name__ == '__main__':
    main()
