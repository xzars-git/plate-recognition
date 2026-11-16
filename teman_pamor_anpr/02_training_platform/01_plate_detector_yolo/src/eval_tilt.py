#!/usr/bin/env python
"""
Ukur ketahanan detektor plat terhadap kemiringan, pada soal uji dari rotate_dataset.py (mode test).

Per sudut: recall = persen plat yang ketemu (IoU kotak tegak >= 0,5 dan >= 0,3) dan rata-rata deteksi
palsu per gambar. Model OBB dibandingkan lewat kotak tegak dari kotak miringnya, supaya angkanya setara.

  python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/eval_tilt.py \
      --model teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/checkpoints/best.pt
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ANGLE = re.compile(r"__a([+-]\d+)$")


def gt_boxes(label_path, w, h):
    out = []
    for line in label_path.read_text().split("\n"):
        v = line.split()
        if len(v) == 9:
            q = np.array(v[1:], float).reshape(4, 2) * [w, h]
            out.append([*q.min(0), *q.max(0)])
    return np.array(out).reshape(-1, 4)


def iou(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M]."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(gt, pred, thr):
    """Jumlah GT yang tertangkap (satu prediksi hanya boleh dipakai sekali) dan jumlah prediksi tak terpakai."""
    if len(gt) == 0 or len(pred) == 0:
        return 0, len(pred)
    m = iou(gt, pred)
    used, hit = set(), 0
    for g in np.argsort(-m.max(1)):
        for p in np.argsort(-m[g]):
            if m[g, p] < thr:
                break
            if p not in used:
                used.add(p)
                hit += 1
                break
    return hit, len(pred) - len(used)


def pred_boxes(r):
    if getattr(r, "obb", None) is not None:
        q = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        return np.concatenate([q.min(1), q.max(1)], 1)
    return r.boxes.xyxy.cpu().numpy()


def evaluate(model_path, test_dir, conf, imgsz):
    from ultralytics import YOLO

    model = YOLO(model_path)
    imgs = sorted((test_dir / "images").glob("*.jpg"))
    st = defaultdict(lambda: np.zeros(5))  # n_gt, hit50, hit30, fp50, n_img
    bs = 32 if str(model_path).endswith(".pt") else 1  # TFLite ekspor app: batch tetap 1
    for i in range(0, len(imgs), bs):
        batch = imgs[i:i + bs]
        for p, r in zip(batch, model.predict([str(x) for x in batch], conf=conf, imgsz=imgsz, verbose=False)):
            a = int(ANGLE.search(p.stem).group(1))
            h, w = r.orig_shape
            gt, pr = gt_boxes(test_dir / "labels" / (p.stem + ".txt"), w, h), pred_boxes(r)
            h50, fp50 = match(gt, pr, 0.5)
            h30, _ = match(gt, pr, 0.3)
            st[a] += [len(gt), h50, h30, fp50, 1]
    return st


def report(name, st):
    print(f"\n{name}")
    print(f"{'sudut':>6} {'plat':>6} {'recall@0.5':>11} {'recall@0.3':>11} {'palsu/gbr':>10}")
    for a in sorted(st, key=lambda x: (abs(x), x)):
        n, h50, h30, fp, ni = st[a]
        print(f"{a:>+6d} {int(n):>6d} {h50 / n:>10.1%} {h30 / n:>10.1%} {fp / ni:>10.2f}")


def selfcheck():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    assert match(gt, np.array([[0, 0, 10, 10]], float), 0.5) == (1, 0)
    assert match(gt, np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float), 0.5) == (1, 1)  # satu GT, dua prediksi
    assert match(gt, np.array([[5, 5, 15, 15]], float), 0.5) == (0, 1)  # IoU 0,14
    assert match(gt, np.zeros((0, 4)), 0.5) == (0, 0)
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", help="satu atau lebih model (.pt / .tflite / .onnx)")
    ap.add_argument("--test", default="dataset/plate_tilt_test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    for m in a.model or []:
        report(m, evaluate(m, Path(a.test), a.conf, a.imgsz))
