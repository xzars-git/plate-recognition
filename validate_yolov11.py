"""
Validasi Model YOLOv11 untuk Deteksi Plat Nomor
"""

from ultralytics import YOLO

def validate_model(model_path, data_yaml='plat_jabar.yaml'):
    """
    Validasi model pada validation set
    
    Args:
        model_path: Path ke model yang akan divalidasi
        data_yaml: Path ke file konfigurasi dataset
    """
    # Load model
    model = YOLO(model_path)
    
    # Validasi
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.45,
        device='cuda',  # Gunakan 'cpu' jika tidak ada GPU
        plots=True,     # Generate plot
        save_json=True, # Save hasil dalam format JSON
        verbose=True,
    )
    
    # Print metrics
    print("\n" + "="*50)
    print("HASIL VALIDASI")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p[0]:.4f}")
    print(f"Recall: {metrics.box.r[0]:.4f}")
    print("="*50)
    
    return metrics

if __name__ == '__main__':
    # Path ke model terbaik
    model_path = 'runs/detect/plat_jabar_yolov11/weights/best.pt'
    
    print("Memulai validasi model...")
    validate_model(model_path)
