#!/usr/bin/env python
"""
Putar gambar plat dengan sudut bebas dan hitung ulang labelnya, tanpa melabel ulang.

Dasar: hampir semua plat di dataset difoto tegak, jadi kotak YOLO-nya = 4 sudut plat. Memutar gambar
sebesar theta dan memutar ke-4 sudut kotak dengan matriks yang sama memberi posisi PASTI plat miring
(kotak miring / OBB). Kanvas diperlebar supaya plat tidak terpotong; tepi diisi BORDER_REPLICATE
(bukan refleksi, yang akan memunculkan salinan plat tanpa label).

Label keluaran = format YOLO-OBB: `kelas x1 y1 x2 y2 x3 y3 x4 y4` (ternormalisasi).

Mode:
  test : soal uji plat miring dari gambar VALIDASI asli (tanpa varian _rot90/180/270, tanpa yang
         basisnya juga ada di train), tiap gambar diputar ke setiap sudut di --angles.
         -> dataset/plate_tilt_test/{images,labels}
  obb  : dataset latih OBB. Tiap gambar train/val disalin (theta=0) + --k varian sudut acak
         dalam [-max, max]. -> dataset/plate_detection_obb/{images,labels}/{train,val} + data.yaml

Jalankan dari root repo:
  python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/rotate_dataset.py test
  python teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/src/rotate_dataset.py obb --k 2 --max 45
"""
import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np

SRC = Path("dataset/plate_detection_augmented")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROT_SUFFIX = re.compile(r"_rot(90|180|270)$")


def read_quads(label_path, w, h):
    """Label YOLO (kotak) -> list quad piksel [4,2], urutan sudut: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    quads = []
    if not label_path.exists():
        return quads
    for line in label_path.read_text().split("\n"):
        v = line.split()
        if len(v) != 5:
            continue
        cx, cy, bw, bh = (float(x) for x in v[1:])
        x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        quads.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64))
    return quads


def rotate(img, quads, theta):
    """Putar img sebesar theta derajat (positif = berlawanan arah jarum jam), kanvas diperlebar."""
    if theta == 0:
        return img, quads
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, [q @ m[:, :2].T + m[:, 2] for q in quads]


def load(img_path, label_path, max_side):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    return img, read_quads(label_path, w, h)


def write(img, quads, out_img, out_lbl):
    h, w = img.shape[:2]
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    lines = []
    for q in quads:
        q = q / [w, h]
        lines.append("0 " + " ".join(f"{v:.6f}" for v in np.clip(q, 0, 1).ravel()))
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))


def base_name(stem):
    return ROT_SUFFIX.sub("", stem)


def images(split):
    return sorted(p for p in (SRC / "images" / split).iterdir() if p.suffix.lower() in IMG_EXT)


def make_test(args):
    out = Path(args.out or "dataset/plate_tilt_test")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    train_bases = {base_name(p.stem) for p in images("train")}
    srcs = [p for p in images("val") if not ROT_SUFFIX.search(p.stem) and p.stem not in train_bases]
    n = 0
    for p in srcs:
        img, quads = load(p, SRC / "labels" / "val" / (p.stem + ".txt"), args.max_side)
        if img is None or not quads:
            continue
        for a in args.angles:
            r, q = rotate(img, quads, a)
            tag = f"{p.stem}__a{a:+d}"
            write(r, q, out / "images" / f"{tag}.jpg", out / "labels" / f"{tag}.txt")
            n += 1
    print(f"soal uji miring: {len(srcs)} gambar validasi x {len(args.angles)} sudut = {n} gambar -> {out}")


def make_obb(args):
    out = Path(args.out or "dataset/plate_detection_obb")
    rng = random.Random(args.seed)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for p in images(split):
            img, quads = load(p, SRC / "labels" / split / (p.stem + ".txt"), args.max_side)
            if img is None:
                continue
            for i, a in enumerate([0] + [rng.uniform(-args.max, args.max) for _ in range(args.k)]):
                r, q = rotate(img, quads, a)
                tag = p.stem if i == 0 else f"{p.stem}__r{i}"
                write(r, q, out / "images" / split / f"{tag}.jpg", out / "labels" / split / f"{tag}.txt")
                n += 1
        print(f"{split}: {n} gambar")
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: license_plate\n")
    print(f"-> {out}/data.yaml")


def selfcheck():
    """Kotak yang diputar 90 derajat harus sama dengan rotasi 90 derajat eksak; sisi kotak tetap sama panjang."""
    img = np.zeros((100, 200, 3), np.uint8)
    q = [np.array([[20, 30], [80, 30], [80, 50], [20, 50]], np.float64)]
    r, rq = rotate(img, q, 90)
    assert r.shape[:2] == (200, 100), r.shape
    exp = np.array([[30, 180], [30, 120], [50, 120], [50, 180]], np.float64)  # (x,y) -> (y, W-x)
    assert np.allclose(rq[0], exp, atol=1e-6), rq[0]
    _, rq = rotate(img, q, 30)
    sides = np.linalg.norm(np.roll(rq[0], -1, 0) - rq[0], axis=1)
    assert np.allclose(sides, [60, 20, 60, 20]), sides
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "obb", "selfcheck"])
    ap.add_argument("--angles", type=int, nargs="+", default=[0, 15, -15, 30, -30, 45, -45])
    ap.add_argument("--k", type=int, default=2, help="obb: jumlah varian sudut acak per gambar")
    ap.add_argument("--max", type=float, default=45.0, help="obb: sudut acak maksimum (derajat)")
    ap.add_argument("--max-side", dest="max_side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"test": make_test, "obb": make_obb, "selfcheck": lambda _: selfcheck()}[a.mode](a)
