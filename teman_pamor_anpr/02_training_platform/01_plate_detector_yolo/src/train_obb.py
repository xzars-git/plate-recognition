#!/usr/bin/env python
"""
Latih detektor plat OBB (kotak miring) dari dataset hasil rotate_dataset.py (mode obb).
Keluaran OBB memberi sudut plat, sehingga crop bisa diluruskan sebelum OCR / klasifikasi warna.

  python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/train_obb.py
  python .../src/eval_tilt.py --model runs/plate_detection/yolo11n_obb_tilt/weights/best.pt
"""
import argparse

import torch
from ultralytics import YOLO

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/plate_detection_obb/data.yaml")
    ap.add_argument("--model", default="yolo11n-obb.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=float, default=16,
                    help="jumlah gambar per batch, atau pecahan VRAM 0-1 (mis. 0.75 = ~9 GB dari 12 GB)")
    ap.add_argument("--name", default="yolo11n_obb_tilt")
    a = ap.parse_args()
    if a.batch >= 1:
        a.batch = int(a.batch)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "TIDAK ADA (akan sangat lambat)")

    YOLO(a.model).train(
        data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, patience=40,
        project="runs/plate_detection", name=a.name, exist_ok=False,
        # Sudut besar sudah dari rotasi offline (+-45); online ditambah sedikit supaya sudut tidak diskrit.
        degrees=15.0, shear=3.0, perspective=0.0005, translate=0.15, scale=0.6,
        fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=15, mixup=0.1,
    )
