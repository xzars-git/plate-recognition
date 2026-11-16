#!/usr/bin/env python
"""
Bangun korpus OCR berlabel + split tetap (uji / validasi / latih) untuk OCR kustom.
Lihat teman_pamor_anpr/docs/OCR_CUSTOM_BLUEPRINT.md.

Sumber (label plat -> crop seperti di aplikasi: padding 0.15, lebar maks 800 px):
- kaggle : label.csv + foto; foto utuh dipotong detektor aplikasi, crop rapat dipakai apa adanya
- jmto   : label dari NAMA BERKAS + kotak YOLO (hanya gambar dengan tepat 1 kotak)
- own    : label dari NAMA BERKAS + kotak di JSON label data sendiri (hanya gambar dengan tepat 1 kotak)

Split dikunci (seed tetap) PER NOMOR PLAT dan terstratifikasi per sumber:
- uji     : 250 plat kaggle (gaya HP) + 250 plat jmto (kamera tol), SATU gambar per plat
- validasi: 150 + 150 plat
- latih   : sisanya (semua gambar plat tersebut). Gambar lain dari plat uji/validasi DIBUANG, bukan
            dimasukkan ke latih (mencegah kebocoran).
Keluaran: dataset/ocr_corpus/{crops,manifest.csv}  (dataset/ di-ignore git: data plat = data pribadi)

Jalankan dari root repo:  python teman_pamor_anpr/02_training_platform/03_ocr_custom/scripts/build_corpus.py
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("dataset")
PLATE = re.compile(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$")
NAME = re.compile(r"(?<![A-Z0-9])([A-Z]{1,2})[-_ ]?(\d{1,4})[-_ ]?([A-Z]{1,3})(?![A-Z0-9])")
PAD = 0.15
MAXW = 800


def plate_from_name(stem):
    s = re.sub(r"\.rf\.[0-9a-f]+|_jpe?g|_png", "", stem.upper())
    m = NAME.search(s)
    return "".join(m.groups()) if m else None


def shrink(im):
    return im if im.width <= MAXW else im.resize((MAXW, int(im.height * MAXW / im.width)), Image.LANCZOS)


def crop(im, box, pad=PAD):
    W, H = im.size
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return shrink(im.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(W, int(x2 + px)), min(H, int(y2 + py)))))


def kaggle(det):
    base = ROOT / "archive/plate_text_dataset/plate_text_dataset"
    for r in csv.DictReader(open(base / "label.csv", encoding="utf-8")):
        lab = r["label"].upper()
        if not PLATE.match(lab):
            continue
        im = Image.open(base / "dataset" / r["filename"]).convert("RGB")
        W, H = im.size
        tight = W <= 800 and W / H >= 2.0
        res = det.predict(np.asarray(im), conf=0.25, iou=0.45, verbose=False)[0]
        if len(res.boxes):
            b = res.boxes.xyxy.cpu().numpy()[int(res.boxes.conf.cpu().numpy().argmax())]
            yield "kaggle", lab, crop(im, b), "rapat" if tight else "adegan"
        elif tight:
            yield "kaggle", lab, shrink(im), "rapat_apa_adanya"


def jmto():
    base = ROOT / "plate_colors/Plat Nomor.v22i.yolov8"
    for sp in ("train", "valid", "test"):
        for lbl in (base / sp / "labels").glob("*.txt"):
            lab = plate_from_name(lbl.stem)
            if not lab or not PLATE.match(lab):
                continue
            rows = [l.split() for l in lbl.read_text().splitlines() if len(l.split()) >= 5]
            if len(rows) != 1:
                continue
            img = next((p for e in (".jpg", ".jpeg", ".png") if (p := base / sp / "images" / (lbl.stem + e)).exists()), None)
            if img is None:
                continue
            im = Image.open(img).convert("RGB")
            W, H = im.size
            xc, yc, w, h = map(float, rows[0][1:5])
            yield "jmto", lab, crop(im, ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)), "tol"


def own():
    base = ROOT / "plate_detection_color/train"
    for img in (base / "images").iterdir():
        lab = plate_from_name(img.stem)
        js = base / "labels" / (img.stem + ".json")
        if not lab or not PLATE.match(lab) or not js.exists():
            continue
        boxes = json.loads(js.read_text(encoding="utf-8")).get("boxes", [])
        if len(boxes) != 1:
            continue
        b = boxes[0]
        if b.get("type") == "polygon":
            xs, ys = [p[0] for p in b["points"]], [p[1] for p in b["points"]]
            nx1, ny1, nx2, ny2 = min(xs), min(ys), max(xs), max(ys)
        else:
            _, xc, yc, w, h = b.get("yolo", b)
            nx1, ny1, nx2, ny2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        im = Image.open(img).convert("RGB")
        W, H = im.size
        yield "own", lab, crop(im, (nx1 * W, ny1 * H, nx2 * W, ny2 * H)), "sendiri"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="D:/Bapenda New/Panah Pasopati/panah-pasopati/android/app/src/main/assets/plat_recognation.tflite")
    ap.add_argument("--out", default=str(ROOT / "ocr_corpus"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(args.detector, task="detect")

    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    counter = Counter()
    for name, gen in (("kaggle", kaggle(det)), ("jmto", jmto()), ("own", own())):
        for source, plate, im, kind in gen:
            counter[(source, plate)] += 1
            rel = f"crops/{source}_{plate}_{counter[(source, plate)]}.jpg"
            im.save(out / rel, quality=95)
            rows.append({"id": Path(rel).stem, "source": source, "plate": plate, "kind": kind,
                         "path": rel, "w": im.width, "h": im.height})
        print("selesai", name, "| total gambar:", len(rows), flush=True)

    plates = defaultdict(list)
    for r in rows:
        plates[r["plate"]].append(r)
    main_src = {p: rs[0]["source"] for p, rs in plates.items()}

    rng = random.Random(args.seed)
    split = {}
    for s in ("kaggle", "jmto"):
        pool = sorted(p for p, ms in main_src.items() if ms == s)
        rng.shuffle(pool)
        for p in pool[:250]:
            split[p] = "test"
        for p in pool[250:400]:
            split[p] = "val"
    for p in plates:
        split.setdefault(p, "train")

    final = []
    for p, rs in plates.items():
        if split[p] == "test":
            rs = rs[:1]  # satu gambar per plat uji; sisanya dibuang (bukan ke latih)
        for r in rs:
            final.append({**r, "split": split[p]})
    final.sort(key=lambda r: (r["split"], r["source"], r["id"]))

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "source", "plate", "kind", "path", "w", "h"])
        w.writeheader()
        w.writerows(final)

    print("\nplat unik:", len(plates), "| gambar total:", len(rows))
    for sp in ("test", "val", "train"):
        sub = [r for r in final if r["split"] == sp]
        print(f"{sp:>5}: {len({r['plate'] for r in sub}):>5} plat, {len(sub):>5} gambar, per sumber:",
              dict(Counter(r["source"] for r in sub)))
    multi = sum(1 for rs in plates.values() if len({r["source"] for r in rs}) > 1)
    print("plat yang muncul di lebih dari satu sumber:", multi)


if __name__ == "__main__":
    main()
