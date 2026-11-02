#!/usr/bin/env python
"""
Pembangkit plat nomor sintetis untuk melatih OCR kustom (hanya untuk data LATIH).
Lihat teman_pamor_anpr/docs/OCR_CUSTOM_BLUEPRINT.md, bagian "Sintetis dan augmentasi".

- Nomor diacak mengikuti distribusi plat LATIH (kode wilayah, panjang angka/seri, frekuensi huruf).
  Nomor yang sama dengan plat uji/validasi/dikeluarkan DITOLAK (tidak boleh ada kebocoran).
- Render: latar hitam/putih/kuning/merah, garis tepi, nomor dengan font plat, baris masa berlaku kecil,
  baut, lalu "tempel" ke latar kendaraan dengan padding acak (meniru crop detektor, kadang terpotong).
- Degradasi kamera: perspektif, rotasi, perkecil lalu perbesar, blur biasa/gerak, JPEG, noise,
  terang/gelap/gamma, warna, silau, bayangan, kotoran.

Pakai:  python .../synth_plates.py --n 20000 --out dataset/ocr_corpus/synth
        python .../synth_plates.py --preview      (lembar pratinjau untuk pemeriksaan visual)
"""

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CORPUS = Path("dataset/ocr_corpus")
FONTS = [(CORPUS / "fonts/PlatNomor.ttf", 0.5), (CORPUS / "fonts/FE-FONT.TTF", 0.25),
         (CORPUS / "fonts/IndoLicensePlate.ttf", 0.25)]


def load_stats():
    rows = list(csv.DictReader(open(CORPUS / "manifest.csv", encoding="utf-8")))
    train = {r["plate"] for r in rows if r["split"] == "train"}
    banned = {r["plate"] for r in rows if r["split"] != "train"}  # uji, validasi, dikeluarkan
    p1, n2, n3, suf = Counter(), Counter(), Counter(), Counter()
    for p in train:
        a, b, c = re.match(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$", p).groups()
        p1[a] += 1
        n2[len(b)] += 1
        n3[len(c)] += 1
        suf.update(c)
    return p1, n2, n3, suf, banned


def sample_plate(rng, st):
    p1, n2, n3, suf, banned = st
    while True:
        a = rng.choices(list(p1), weights=p1.values())[0]
        nd = rng.choices(list(n2), weights=n2.values())[0]
        b = str(rng.randint(10 ** (nd - 1), 10 ** nd - 1)) if nd > 1 else str(rng.randint(1, 9))
        c = "".join(rng.choices(list(suf), weights=suf.values(), k=rng.choices(list(n3), weights=n3.values())[0]))
        if a + b + c not in banned:
            return a, b, c


def render_plate(parts, rng):
    kind = rng.choices(["black", "white", "yellow", "red"], weights=[0.45, 0.40, 0.10, 0.05])[0]
    j = rng.randint
    if kind == "black":
        bg, fg = (j(15, 45),) * 3, (j(215, 250),) * 3
    elif kind == "white":
        bg, fg = (j(215, 250),) * 3, (j(5, 40),) * 3
    elif kind == "yellow":
        bg, fg = (255, j(165, 205), j(0, 40)), (j(0, 30),) * 3
    else:
        bg, fg = (j(170, 215), j(20, 60), j(20, 60)), (j(225, 255),) * 3
    car = rng.random() < 0.6
    W, H = (j(380, 520), j(105, 150)) if car else (j(250, 330), j(105, 150))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    bw = j(2, 4)
    d.rectangle([bw * 2, bw * 2, W - bw * 2, H - bw * 2], outline=fg, width=bw)

    font_path = rng.choices([f for f, _ in FONTS], weights=[w for _, w in FONTS])[0]
    fs = int(H * rng.uniform(0.50, 0.62))
    while True:
        font = ImageFont.truetype(str(font_path), fs)
        gap = font.getlength("0") * rng.uniform(0.3, 0.7)
        widths = [font.getlength(p) for p in parts]
        total = sum(widths) + gap * 2
        if total < W * 0.86 or fs < 20:
            break
        fs -= 2
    x = (W - total) / 2 + rng.uniform(-6, 6)
    y = H * rng.uniform(0.10, 0.22)
    sw = rng.choice([0, 0, 1, 1, 2]) if fs >= 40 else rng.choice([0, 0, 1])  # plat asli timbul/tebal
    for p, w in zip(parts, widths):
        d.text((x, y), p, font=font, fill=fg, stroke_width=sw, stroke_fill=fg)
        x += w + gap + 2 * sw
    if rng.random() < 0.85:  # baris masa berlaku kecil
        small = ImageFont.truetype(str(font_path), int(H * rng.uniform(0.14, 0.20)))
        t = f"{rng.randint(1, 12):02d}{rng.choice('.-')}{rng.randint(22, 31):02d}"  # font tak punya titik tengah
        d.text(((W - small.getlength(t)) / 2, H * rng.uniform(0.74, 0.80)), t, font=small, fill=fg)
    if rng.random() < 0.4:  # baut
        shade = -60 if sum(bg) > 400 else 60
        col = tuple(max(0, min(255, v + shade)) for v in bg)
        for cx in (bw * 6, W - bw * 6):
            d.ellipse([cx - 4, H / 2 - 4, cx + 4, H / 2 + 4], fill=col)
    return img


def degrade(plate_img, rng, nprng):
    W, H = plate_img.size
    # latar kendaraan + padding acak seperti crop detektor (negatif = terpotong di tepi)
    padx, pady = rng.uniform(-0.04, 0.22) * W, rng.uniform(-0.04, 0.30) * H
    # latar kendaraan realistis: gelap/abu paling sering, kadang terang, kadang warna kusam (bukan neon)
    g = rng.choices([rng.randint(10, 90), rng.randint(110, 230)], weights=[0.7, 0.3])[0]
    bgc = tuple(int(max(0, min(255, g + nprng.integers(-35, 36)))) for _ in range(3)) if rng.random() < 0.35 else (g,) * 3
    Wc, Hc = int(W + 2 * max(padx, 0)), int(H + 2 * max(pady, 0))
    canvas = Image.new("RGB", (Wc, Hc), bgc)
    canvas.paste(plate_img, (int(max(padx, 0) + min(padx, 0) / 2), int(max(pady, 0) + min(pady, 0) / 2)))
    a = np.asarray(canvas).astype(np.float32)

    if rng.random() < 0.3:  # silau / bayangan
        gx = np.linspace(0, 1, Wc)[None, :, None]
        a = a + rng.uniform(40, 120) * np.exp(-((gx - rng.random()) ** 2) / 0.02) * rng.choice([1, 1, -1])
    if rng.random() < 0.3:  # kotoran
        for _ in range(rng.randint(3, 15)):
            cv2.circle(a, (rng.randint(0, Wc), rng.randint(0, Hc)), rng.randint(1, 4), (rng.randint(0, 255),) * 3, -1)
    a = np.clip(a, 0, 255).astype(np.uint8)

    # perspektif + rotasi
    s = rng.uniform(0, 0.07)
    src = np.float32([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]])
    dst = src + nprng.uniform(-s, s, (4, 2)).astype(np.float32) * np.float32([Wc, Hc])
    a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (Wc, Hc), borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(a, cv2.getRotationMatrix2D((Wc / 2, Hc / 2), rng.uniform(-5, 5), 1.0), (Wc, Hc),
                       borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:  # crop tol: baris bawah / sisi terpotong, huruf seri akhir separuh hilang
        top = max(pady, 0) + min(pady, 0) / 2  # huruf berada di ~0,1-0,85 tinggi plat dari tepi atas plat
        cut = min(Hc, int(top + H * rng.uniform(0.62, 0.92))) if rng.random() < 0.8 else Hc
        side = int(Wc * rng.uniform(0.0, 0.05))
        a = a[:cut, side: Wc - (side if rng.random() < 0.5 else 0)]
        Hc, Wc = a.shape[:2]

    # resolusi rendah (plat jauh): perkecil sampai tinggi 14-220 px
    # minimum 22 px: di bawah itu tulisan tak terbaca dan hanya jadi noise label
    th = rng.choices([rng.randint(22, 40), rng.randint(40, 100), rng.randint(100, 220)], weights=[0.3, 0.4, 0.3])[0]
    a = cv2.resize(a, (max(8, int(Wc * th / Hc)), max(8, th)), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:  # blur biasa
        k = rng.choice([3, 5, 7])
        # blur diskalakan resolusi: gambar kecil sudah kabur, blur berat membuatnya tak terbaca
        a = cv2.GaussianBlur(a, (k, k), max(0.4, rng.uniform(0.5, 2.0) * min(1.0, th / 80)))
    if rng.random() < 0.25:  # blur gerak
        k = rng.choice([5, 7, 9])
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(-20, 20), 1), (k, k))
        a = cv2.filter2D(a, -1, ker / max(float(ker.sum()), 1e-6))

    f = a.astype(np.float32)
    f = 255 * (f / 255) ** rng.uniform(0.6, 1.8)  # gamma
    f = (f - 128) * rng.uniform(0.6, 1.3) + 128 + rng.uniform(-30, 30)  # kontras / terang
    f = f * nprng.uniform(0.94, 1.06, 3)  # pergeseran warna ringan
    f += nprng.normal(0, rng.uniform(0, 12), f.shape)  # noise
    a = np.clip(f, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 95)])
    return Image.fromarray(cv2.imdecode(enc, cv2.IMREAD_COLOR)).convert("RGB")


def make(rng, nprng, st):
    parts = sample_plate(rng, st)
    return "".join(parts), degrade(render_plate(parts, rng), rng, nprng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(CORPUS / "synth"))
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    st = load_stats()
    rng, nprng = random.Random(args.seed), np.random.default_rng(args.seed)

    if args.preview:
        out = CORPUS / "synth_preview.png"
        sh = Image.new("RGB", (6 * 330, 6 * 110), "gray")
        for k in range(36):
            plate, im = make(rng, nprng, st)
            im.thumbnail((320, 90))
            x, y = (k % 6) * 330, (k // 6) * 110
            sh.paste(im, (x, y))
            ImageDraw.Draw(sh).text((x + 2, y + 94), plate, fill="yellow")
        sh.save(out)
        print("pratinjau:", out)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "synth_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "plate", "path"])
        for k in range(args.n):
            plate, im = make(rng, nprng, st)
            rel = f"s{k:06d}.jpg"
            im.save(out / rel, quality=92)
            w.writerow([f"s{k:06d}", plate, rel])
            if (k + 1) % 2000 == 0:
                print("sintetis:", k + 1, flush=True)
    print("selesai:", args.n, "->", out)


if __name__ == "__main__":
    main()
