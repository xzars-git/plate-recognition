#!/usr/bin/env python
"""
Tambah data warna dari dataset Roboflow (YOLOv8) berlabel warna, mis. JMTO "Plat Nomor" v22.

- Kelas dipetakan lewat nama: Hitam-* -> black, Putih-* -> white, Kuning-* -> yellow, Merah-* -> red.
- Split mengikuti split Roboflow: train -> train, valid+test -> val.
- --cap membatasi crop per warna dari dataset ini (putih ~3000 akan membanjiri dataset).
- --synth-red N: SEMUA plat merah asli (dataset kita + eksternal) dipindah ke val/red, lalu train/red
  diisi N plat merah sintetis dari plat hitam (latar gelap -> merah, teks putih).
  Val red jadi uji nyata: kalau recall merah asli tinggi, sintetis terbukti berlaku.

Lisensi dataset JMTO di Roboflow: CC BY 4.0 -> wajib atribusi (lihat README modul).
"""

import argparse
import random
import shutil
import zlib
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

COLOR_PREFIX = {'hitam': 'black', 'putih': 'white', 'kuning': 'yellow', 'merah': 'red'}
COLORS = ['white', 'black', 'red', 'yellow']
SPLIT_MAP = {'train': 'train', 'valid': 'val', 'test': 'val'}


def import_external(src, out, size, cap, min_w, min_h):
    names = yaml.safe_load((src / 'data.yaml').read_text())['names']
    cls_color = {i: COLOR_PREFIX.get(n.split('-')[0].lower()) for i, n in enumerate(names)}
    rng = random.Random(42)

    items = []  # (color, split, img_path, idx, box)
    for split_dir, split in SPLIT_MAP.items():
        for lbl in (src / split_dir / 'labels').glob('*.txt'):
            img = next((p for e in ('.jpg', '.jpeg', '.png') if (p := src / split_dir / 'images' / (lbl.stem + e)).exists()), None)
            if img is None:
                continue
            for idx, line in enumerate(lbl.read_text().splitlines()):
                parts = line.split()
                if len(parts) >= 5 and cls_color.get(int(parts[0])):
                    items.append((cls_color[int(parts[0])], split, img, idx, list(map(float, parts[1:5]))))

    rng.shuffle(items)
    per_color = {c: 0 for c in COLORS}
    saved = {c: 0 for c in COLORS}
    for color, split, img_path, idx, (xc, yc, w, h) in items:
        if per_color[color] >= cap:
            continue
        im = Image.open(img_path).convert('RGB')
        W, H = im.size
        x1, y1 = max(0, int((xc - w / 2) * W)), max(0, int((yc - h / 2) * H))
        x2, y2 = min(W, int((xc + w / 2) * W)), min(H, int((yc + h / 2) * H))
        if x2 - x1 < min_w or y2 - y1 < min_h:
            continue
        per_color[color] += 1
        dest = out / split / color
        dest.mkdir(parents=True, exist_ok=True)
        im.crop((x1, y1, x2, y2)).resize((size, size), Image.Resampling.LANCZOS).save(
            dest / f'jmto_{img_path.stem}_{idx}.jpg', quality=95)
        saved[color] += 1
    print(f"Eksternal ditambahkan: {saved}")


def synth_red(path):
    """Plat hitam (latar gelap, teks putih) -> plat merah (latar merah, teks putih)."""
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    lum = a.mean(2)
    lo, hi = np.percentile(lum, 5), np.percentile(lum, 95)
    t = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)[..., None]
    # plat merah asli bervariasi: merah terang, pink, sampai maroon gelap saat malam
    bg = np.array([random.uniform(110, 220), random.uniform(15, 110), random.uniform(15, 110)]) / 255
    fg = np.array([random.uniform(150, 255)] * 3) / 255
    out = bg * (1 - t) + fg * t
    out = out * random.uniform(0.2, 1.15) + np.random.normal(0, 0.01, out.shape)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def make_synth_red(out, n, real_to_train=0, web_dir=None):
    random.seed(42)
    np.random.seed(42)
    (out / 'val' / 'red').mkdir(parents=True, exist_ok=True)
    (out / 'train' / 'red').mkdir(parents=True, exist_ok=True)
    for p in (out / 'train' / 'red').glob('*.jpg'):
        if p.name.startswith('real_0_'):  # kembalikan dulu ke val agar idempoten
            shutil.copy(p, out / 'val' / 'red' / p.name[len('real_0_'):])
        elif not p.name.startswith(('synth_', 'real_')):
            shutil.move(str(p), out / 'val' / 'red' / p.name)
    for p in list((out / 'train' / 'red').glob('*.jpg')):
        if p.name.startswith(('synth_', 'real_')):
            p.unlink()
    # sebagian plat merah asli ikut training (diulang 5x agar berbobot); sisanya tetap jadi uji
    for i, p in enumerate(sorted((out / 'val' / 'red').glob('*.jpg'))[::2][:real_to_train]):
        for k in range(5):
            shutil.copy(p, out / 'train' / 'red' / f'real_{k}_{p.name}')
        (out / 'val' / 'red' / p.name).unlink()
    black = sorted((out / 'train' / 'black').glob('*.jpg'))
    for i in range(n):
        synth_red(random.choice(black)).save(out / 'train' / 'red' / f'synth_{i}.jpg', quality=95)
    for p in list((out / 'train' / 'red').glob('web_*.jpg')) + list((out / 'val' / 'red').glob('web_*.jpg')):
        p.unlink()
    web = sorted(Path(web_dir).glob('*.jpg')) if web_dir else []
    for p in web:  # ~25% ke val; sisanya train
        split = 'val' if zlib.crc32(p.stem.encode()) % 4 == 0 else 'train'
        shutil.copy(p, out / split / 'red' / p.name)
    real = len(list((out / 'val' / 'red').glob('*.jpg')))
    print(f"red: train={n} sintetis + {real_to_train} asli(x5) + web, val={real} ASLI (uji); web total={len(web)}")


def dark_aug(out, per_image):
    """Versi 'foto malam' dari plat merah ASLI di train (digelapkan, desaturasi, noise, blur, color cast).

    Hanya dari train; uji (val/red) tidak disentuh. Hasil berprefix dark_ dan dibuat ulang tiap run.
    """
    import cv2
    rng = np.random.default_rng(42)
    red_dir = out / 'train' / 'red'
    for p in red_dir.glob('dark_*.jpg'):
        p.unlink()
    srcs = [p for p in sorted(red_dir.glob('*.jpg')) if not p.name.startswith(('synth_', 'dark_'))
            and not p.name.startswith('real_') or p.name.startswith('real_0_')]
    n = 0
    for p in srcs:
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255
        for k in range(per_image):
            x = (a ** rng.uniform(1.6, 2.8)) * rng.uniform(0.45, 0.95)      # gelap
            hsv = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= rng.uniform(0.5, 1.0)                              # warna memudar
            x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255
            x = x * rng.uniform(0.9, 1.1, 3)                                  # color cast
            r = rng.uniform(0, 1.2)
            if r > 0.3:
                x = cv2.GaussianBlur(x, (0, 0), r)
            x = np.clip(x + rng.normal(0, rng.uniform(0.01, 0.04), x.shape), 0, 1)
            Image.fromarray((x * 255).astype(np.uint8)).save(red_dir / f'dark_{p.stem}_{k}.jpg', quality=int(rng.integers(70, 96)))
            n += 1
    print(f"dark_aug: {n} gambar merah gelap dari {len(srcs)} sumber asli (train saja)")


def add_yellow(out, yellow_dir):
    """Crop kuning hasil pindai dataset (scanY_*.jpg), 80% train / 20% val."""
    for s in ('train', 'val'):
        for p in (out / s / 'yellow').glob('scanY_*.jpg'):
            p.unlink()
    files = sorted(Path(yellow_dir).glob('*.jpg'))
    for p in files:
        split = 'val' if zlib.crc32(p.stem.encode()) % 5 == 0 else 'train'
        shutil.copy(p, out / split / 'yellow' / p.name)
    print(f"yellow tambahan: {len(files)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder hasil extract Roboflow YOLOv8')
    ap.add_argument('--out', default='dataset/plate_colors')
    ap.add_argument('--size', type=int, default=96)
    ap.add_argument('--cap', type=int, default=400, help='maks crop per warna dari dataset eksternal')
    ap.add_argument('--min-w', type=int, default=40)
    ap.add_argument('--min-h', type=int, default=12)
    ap.add_argument('--synth-red', type=int, default=0, help='jumlah plat merah sintetis untuk train')
    ap.add_argument('--real-red-to-train', type=int, default=0, help='jumlah plat merah asli yang ikut training')
    ap.add_argument('--web-red-dir', default=None, help='folder crop plat merah asli hasil kumpulan web (96x96 jpg)')
    ap.add_argument('--yellow-dir', default=None, help='folder crop kuning tambahan (scanY_*.jpg)')
    ap.add_argument('--dark-aug', type=int, default=0, help='jumlah versi "foto malam" per plat merah asli di train')
    args = ap.parse_args()

    out = Path(args.out)
    import_external(Path(args.src), out, args.size, args.cap, args.min_w, args.min_h)
    if args.yellow_dir:
        add_yellow(out, args.yellow_dir)
    if args.synth_red:
        make_synth_red(out, args.synth_red, args.real_red_to_train, args.web_red_dir)
    if args.dark_aug:
        dark_aug(out, args.dark_aug)
    for s in ('train', 'val'):
        print(s, {c: len(list((out / s / c).glob('*.jpg'))) for c in COLORS})


if __name__ == '__main__':
    main()
