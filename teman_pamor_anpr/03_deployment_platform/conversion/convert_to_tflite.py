#!/usr/bin/env python
"""
Convert color classifier (.keras) -> .tflite, lalu validasi terhadap model Keras.

Gate (dari 03_deployment_platform/README.md):
  - akurasi TFLite turun < 2% dibanding Keras
  - prediksi TFLite == Keras minimal 99% sampel
  - ukuran < 10 MB
Exit code 1 kalau gate gagal.

Input TFLite: RGB float32 [1, H, W, 3], piksel 0-255 (normalisasi ada di dalam model).
Urutan output = urutan baris di labels.txt.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def convert(model, out_path, mode):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    out_path.write_bytes(converter.convert())


def load_val(val_dir, classes, size):
    xs, ys = [], []
    for i, c in enumerate(classes):
        for p in sorted((val_dir / c).glob('*.jpg')):
            xs.append(np.asarray(Image.open(p).convert('RGB').resize((size, size)), dtype=np.float32))
            ys.append(i)
    return np.stack(xs), np.array(ys)


def run_tflite(tflite_path, x):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds, t0 = [], time.perf_counter()
    for img in x:
        interp.set_tensor(inp['index'], img[None])
        interp.invoke()
        preds.append(interp.get_tensor(out['index'])[0])
    ms = (time.perf_counter() - t0) / len(x) * 1000
    return np.array(preds), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/color_classifier/best_model.keras')
    ap.add_argument('--out', default=None, help='default: samping --model, nama color_classifier.tflite')
    ap.add_argument('--val-dir', default='dataset/plate_colors/val')
    ap.add_argument('--mode', choices=['float32', 'float16'], default='float32')
    args = ap.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.out) if args.out else model_path.with_name('color_classifier.tflite')
    classes = (model_path.with_name('labels.txt')).read_text().split()

    model = tf.keras.models.load_model(model_path)
    size = model.input_shape[1]
    convert(model, out_path, args.mode)

    x, y = load_val(Path(args.val_dir), classes, size)
    keras_pred = model.predict(x, verbose=0).argmax(1)
    tfl_probs, ms = run_tflite(out_path, x)
    tfl_pred = tfl_probs.argmax(1)

    keras_acc, tfl_acc = (keras_pred == y).mean() * 100, (tfl_pred == y).mean() * 100
    agree = (keras_pred == tfl_pred).mean() * 100
    size_mb = out_path.stat().st_size / 1e6

    print(f"classes     : {classes}")
    print(f"samples     : {len(y)}")
    print(f"keras acc   : {keras_acc:.2f}%")
    print(f"tflite acc  : {tfl_acc:.2f}%  (drop {keras_acc - tfl_acc:+.2f}%)")
    print(f"agreement   : {agree:.2f}%")
    print(f"size        : {size_mb:.2f} MB  ({out_path})")
    print(f"latency     : {ms:.1f} ms/img (desktop CPU, bukan angka mobile)")

    ok = keras_acc - tfl_acc < 2 and agree >= 99 and size_mb < 10
    print("GATE        :", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
