#!/usr/bin/env python
"""
Kumpulkan crop plat dari hasil pencarian gambar web untuk satu warna (red | yellow).

Alur: query Bing Images -> unduh thumbnail (toko online dibuang) -> detektor plat repo memotong plat
-> saring yang dominan warna target (HSV) -> dedup (dHash, juga terhadap --existing) -> simpan 96x96.
Hasil WAJIB dicek mata (contact sheet sheet_*.png di folder output) sebelum dipakai training.

Catatan: gambar web punya hak cipta pemiliknya; pakai untuk training internal saja, jangan dipublikasikan.
"""

import argparse
import concurrent.futures as cf
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "id-ID,id;q=0.9"}
SHOP = re.compile(r"shopee|tokopedia|lazada|bukalapak|blibli|olx|alibaba|aliexpress|tiktok|pinterest|youtube|facebook|instagram|carousell|jakmall|ebay|amazon", re.I)

QUERIES = {
    'red': ["mobil dinas plat merah", "kendaraan dinas pemerintah plat merah di jalan", "plat merah mobil pemda",
            "motor dinas plat merah", "plat nomor merah close up kendaraan", "mobil dinas plat merah malam hari",
            "mobil dinas DPRD plat merah", "plat merah kendaraan dinas parkir", "mobil dinas walikota plat merah",
            "mobil dinas bupati plat merah", "ambulans dinas plat merah", "truk dinas pemerintah plat merah",
            "plat merah Innova dinas", "plat merah Pajero dinas", "plat merah Satpol PP motor dinas",
            "plat merah mobil dinas kecelakaan", "plat merah kantor pemerintah parkiran"],
    'yellow': ["angkot plat kuning", "taksi plat kuning", "bus plat kuning", "truk plat kuning",
               "mobil rental plat kuning", "bus pariwisata plat kuning", "plat nomor kuning close up kendaraan",
               "plat kuning malam hari", "pickup plat kuning angkutan umum", "plat kuning ojek online mobil",
               "plat kuning bajaj", "plat kuning colt L300 angkutan"],
}


def color_frac(rgb, color):
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    if color == 'red':
        m = ((h < 10) | (h > 168)) & (s > 80) & (v > 45)
    else:
        m = (h >= 15) & (h <= 35) & (s > 90) & (v > 90)
    return float(m.mean())


def dhash(img):
    g = np.asarray(img.convert("L").resize((9, 8)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def search(queries, pages):
    items = {}
    for q in queries:
        for first in range(1, 35 * pages, 35):
            try:
                u = f"https://www.bing.com/images/async?q={urllib.parse.quote(q)}&first={first}&count=35&mmasync=1"
                h = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read().decode("utf8", "ignore")
            except Exception as e:
                print("err", q, first, e)
                continue
            for m in re.findall(r'\sm="(\{.*?\})"', h):
                try:
                    d = json.loads(html.unescape(m))
                except Exception:
                    continue
                i = (re.search(r"id=([^&]+)", d.get("turl", "")) or [None, None])[1]
                if i and i not in items and not SHOP.search(d.get("purl", "")):
                    items[i] = d
    return items


def download(i, dest):
    p = dest / (re.sub(r"[^A-Za-z0-9_-]", "_", i) + ".jpg")
    if not p.exists():
        try:
            p.write_bytes(urllib.request.urlopen(urllib.request.Request(
                f"https://tse1.mm.bing.net/th?id={urllib.parse.quote(i)}&pid=15.1", headers=UA), timeout=30).read())
        except Exception:
            return None
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--color', choices=['red', 'yellow'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--detector', default='teman_pamor_anpr/02_training_platform/01_plate_detector_yolo/checkpoints/best.pt')
    ap.add_argument('--pages', type=int, default=5, help='halaman hasil per query (35 gambar/halaman)')
    ap.add_argument('--min-frac', type=float, default=0.40)
    ap.add_argument('--existing', nargs='*', default=[], help='folder crop yang sudah ada (untuk dedup)')
    ap.add_argument('--prefix', default='web')
    args = ap.parse_args()

    out = Path(args.out)
    thumbs = out / 'thumbs'
    thumbs.mkdir(parents=True, exist_ok=True)
    items = search(QUERIES[args.color], args.pages)
    print("gambar unik non-toko:", len(items))
    with cf.ThreadPoolExecutor(8) as ex:
        paths = [p for p in ex.map(lambda i: download(i, thumbs), items) if p]

    seen = [dhash(Image.open(p)) for d in args.existing for p in Path(d).glob('*.jpg')]
    det = YOLO(args.detector)
    start = len(list(out.glob(f'{args.prefix}_*.jpg')))
    kept = []
    for p in sorted(paths):
        im = np.asarray(Image.open(p).convert("RGB"))
        for b in det.predict(im, conf=0.35, verbose=False)[0].boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = b
            crop = im[max(0, y1):y2, max(0, x1):x2]
            if crop.shape[1] < 36 or crop.shape[0] < 10 or color_frac(crop, args.color) < args.min_frac:
                continue
            pil = Image.fromarray(crop).resize((96, 96))
            h = dhash(pil)
            if any((h != s).sum() <= 16 for s in seen):
                continue
            seen.append(h)
            f = out / f"{args.prefix}_{start + len(kept) + 1:03d}.jpg"
            pil.save(f, quality=95)
            kept.append(f)
    print(f"{args.color}: {len(kept)} crop baru unik -> {out}")

    cols = 15
    for pg in range(0, len(kept), 60):
        ch = kept[pg:pg + 60]
        sh = Image.new("RGB", (cols * 100, ((len(ch) + cols - 1) // cols) * 100), "gray")
        for i, f in enumerate(ch):
            sh.paste(Image.open(f), ((i % cols) * 100, (i // cols) * 100))
        sh.save(out / f"sheet_{pg // 60}.png")


if __name__ == '__main__':
    main()
