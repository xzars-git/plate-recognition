#!/usr/bin/env python
"""
Ekspor OCR kustom (PyTorch) ke TFLite lewat port Keras + salin bobot, lalu uji paritas.

  python .../src/export_tflite.py models/ocr_custom_v2/best.pt

Kontrak model TFLite:
  masukan : float32 [1,48,192,1], abu-abu 0..255 (normalisasi (x-127.5)/127.5 ada DI DALAM model),
            hasil letterbox(48x192) seperti train.py
  keluaran: float32 [1,48,37] logit per langkah waktu; indeks 0 = blank CTC, i = CHARS[i-1]; dekode greedy di sisi klien
"""

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train as T  # noqa: E402


def build_keras():
    L = tf.keras.layers

    def blk(x, o, k=3, padding="same"):
        x = L.Conv2D(o, (k, 1) if padding == "valid" else k, padding=padding, use_bias=False)(x)
        return L.ReLU()(L.BatchNormalization(epsilon=1e-5)(x))

    inp = tf.keras.Input((T.H, T.W, 1))
    x = L.Rescaling(1 / 127.5, offset=-1)(inp)  # == (x - 127.5) / 127.5
    x = blk(x, 32); x = L.MaxPool2D(2)(x)
    x = blk(x, 64); x = L.MaxPool2D(2)(x)
    x = blk(x, 128); x = blk(x, 128); x = L.MaxPool2D((2, 1))(x)
    x = blk(x, 192); x = blk(x, 192); x = L.MaxPool2D((2, 1))(x)
    x = blk(x, 192, k=3, padding="valid")  # kernel (3,1) menciutkan tinggi 3 -> 1
    x = L.Reshape((T.W // 4, 192))(x)
    for _ in range(2):
        x = L.Conv1D(192, 3, padding="same", use_bias=False)(x)
        x = L.ReLU()(L.BatchNormalization(epsilon=1e-5)(x))
    return tf.keras.Model(inp, L.Conv1D(T.NCLS, 1)(x))


def copy_weights(net, km):
    sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    convs = [k[:-7] for k in sd if k.endswith(".weight") and sd[k].ndim >= 3]
    kconv = [l for l in km.layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv1D))]
    kbn = [l for l in km.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert len(convs) == len(kconv) and len(kbn) == len(convs) - 1, (len(convs), len(kconv), len(kbn))
    for name, kl in zip(convs, kconv):
        w = sd[name + ".weight"]
        kl.set_weights([w.transpose(2, 3, 1, 0) if w.ndim == 4 else w.transpose(2, 1, 0)]
                       + ([sd[name + ".bias"]] if name + ".bias" in sd else []))
    for name, kl in zip([k[:-13] for k in sd if k.endswith(".running_mean")], kbn):
        kl.set_weights([sd[name + ".weight"], sd[name + ".bias"], sd[name + ".running_mean"], sd[name + ".running_var"]])


def tflite_predict(path, imgs):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    outs = []
    for g in imgs:
        it.set_tensor(i, g[None, :, :, None].astype(np.float32))
        it.invoke()
        outs.append(it.get_tensor(o)[0].copy())
    return np.stack(outs)  # N,T,NCLS


def main():
    ckpt = Path(sys.argv[1])
    out = ckpt.parent
    net = T.Net()
    net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    net.eval()
    km = build_keras()
    copy_weights(net, km)

    test = T.read_real("test")
    ds = T.PlateDS([(p, l) for p, l, _ in test], False)
    imgs = [T.letterbox(g if g is not None else np.full((T.H, T.W), 128, np.uint8)) for g in ds.cache]
    labels = [l for _, l, _ in test]

    with torch.no_grad():
        tl = net(torch.from_numpy((np.stack(imgs).astype(np.float32) - 127.5) / 127.5)[:, None])  # N,NCLS,T
    kl = km(np.stack(imgs)[..., None].astype(np.float32)).numpy()
    print(f"selisih logit maks PyTorch vs Keras: {np.abs(tl.permute(0, 2, 1).numpy() - kl).max():.2e}")

    ref = T.decode(tl)
    print(f"PyTorch benar persis: {100 * np.mean([a == b for a, b in zip(ref, labels)]):.1f}%")

    def conv(opt=False, fp16=False):
        c = tf.lite.TFLiteConverter.from_keras_model(km)
        if opt:
            c.optimizations = [tf.lite.Optimize.DEFAULT]
        if fp16:
            c.target_spec.supported_types = [tf.float16]
        return c

    for name, c in {"fp32": conv(), "fp16": conv(True, True), "int8dinamis": conv(True)}.items():
        p = out / f"ocr_{name}.tflite"
        p.write_bytes(c.convert())
        pr = T.decode(torch.from_numpy(tflite_predict(p, imgs)).permute(0, 2, 1))
        t0 = time.time()
        tflite_predict(p, imgs[:100])
        ms = (time.time() - t0) / 100 * 1000
        print(f"{name:12s} {p.stat().st_size / 1e6:5.2f} MB | sama dengan PyTorch "
              f"{100 * np.mean([a == b for a, b in zip(pr, ref)]):5.1f}% | benar persis "
              f"{100 * np.mean([a == b for a, b in zip(pr, labels)]):.1f}% | {ms:.1f} ms/gambar (CPU PC, 1 thread)")


if __name__ == "__main__":
    main()
