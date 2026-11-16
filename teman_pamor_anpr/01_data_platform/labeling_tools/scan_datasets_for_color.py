#!/usr/bin/env python
"""
Pindai dataset plat lokal (tanpa label warna) untuk mencari plat MERAH dan KUNING.

Per gambar: detektor plat repo memotong plat; kalau tidak ada deteksi dan gambar berbentuk plat
(rasio 1.6-4.5, lebar <= 600) seluruh gambar dianggap crop plat. Crop disaring warna (HSV),
didedup (dHash, juga terhadap --existing), disimpan 96x96 + contact sheet untuk dicek mata.
Folder karakter (angka/huruf/textplat) dilewati.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from harvest_web_plates import color_frac, dhash

SKIP = ('angka', 'huruf', 'textplat', 'sheet_', 'thumbs')
EXT = {'.jpg', '.jpeg', '.png'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out', default='dataset/plate_colors_extra')
    ap.add_argument('--detector', default='teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/checkpoints/best.pt')
    ap.add_argument('--min-frac', type=float, default=0.40)
    ap.add_argument('--existing', nargs='*', default=[])
    ap.add_argument('--limit', type=int, default=0, help='batasi jumlah gambar (uji cepat)')
    args = ap.parse_args()

    out = {c: Path(args.out) / f'{c}_scan' for c in ('red', 'yellow')}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    seen = {c: [] for c in out}
    for d in args.existing:
        for p in Path(d).glob('*.jpg'):
            h = dhash(Image.open(p))
            for c in out:
                seen[c].append(h)
    kept = {c: [] for c in out}
    det = YOLO(args.detector)

    files = [p for d in args.dirs for p in Path(d).rglob('*')
             if p.suffix.lower() in EXT and not any(s in str(p).lower() for s in SKIP)]
    if args.limit:
        files = files[:args.limit]
    print("gambar dipindai:", len(files), flush=True)

    for n, p in enumerate(files):
        try:
            im = np.asarray(Image.open(p).convert('RGB'))
        except Exception:
            continue
        H, W = im.shape[:2]
        crops = [im[max(0, y1):y2, max(0, x1):x2] for x1, y1, x2, y2 in
                 det.predict(im, conf=0.35, verbose=False)[0].boxes.xyxy.cpu().numpy().astype(int)]
        if not crops and 1.6 <= W / H <= 4.5 and W <= 600:
            crops = [im]
        for crop in crops:
            if crop.shape[1] < 36 or crop.shape[0] < 10:
                continue
            for c in out:
                if color_frac(crop, c) < args.min_frac:
                    continue
                pil = Image.fromarray(crop).resize((96, 96))
                h = dhash(pil)
                if any((h != s).sum() <= 10 for s in seen[c]):
                    continue
                seen[c].append(h)
                f = out[c] / f"scan_{len(kept[c]) + 1:04d}.jpg"
                pil.save(f, quality=95)
                kept[c].append(f)
        if n % 500 == 0:
            print(n, {c: len(v) for c, v in kept.items()}, flush=True)

    for c, fl in kept.items():
        cols = 15
        for pg in range(0, len(fl), 60):
            ch = fl[pg:pg + 60]
            sh = Image.new("RGB", (cols * 100, ((len(ch) + cols - 1) // cols) * 100), "gray")
            for i, f in enumerate(ch):
                sh.paste(Image.open(f), ((i % cols) * 100, (i // cols) * 100))
            sh.save(out[c] / f"sheet_{pg // 60}.png")
    print("selesai:", {c: len(v) for c, v in kept.items()})


if __name__ == '__main__':
    main()
