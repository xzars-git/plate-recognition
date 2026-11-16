#!/usr/bin/env python
"""
OCR plat nomor kustom: CNN penuh + CTC (tanpa RNN, mudah diekspor ke TFLite).
Lihat teman_pamor_anpr/docs/OCR_CUSTOM_BLUEPRINT.md.

  Latih  : python .../src/train.py --epochs 60
  Uji    : python .../src/train.py --eval-test models/ocr_custom/best.pt     (SOAL UJI TERKUNCI)

Aturan: model dipilih dengan VALIDASI asli. Soal uji hanya dilihat sekali per versi besar.
Data (dataset/ocr_corpus, tidak di-commit): manifest.csv (split train/val/test) + synth/synth_manifest.csv.
"""

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

CORPUS = Path("dataset/ocr_corpus")
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
C2I = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = blank (CTC)
NCLS = len(CHARS) + 1
H, W = 48, 192
PLATE = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$")
MLKIT_BEST = 40.1  # benar persis, ML Kit terbaik pada soal uji (blueprint)
GATE = MLKIT_BEST + 15


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]

        self.cnn = nn.Sequential(
            *blk(1, 32), nn.MaxPool2d(2),                                   # 24x96
            *blk(32, 64), nn.MaxPool2d(2),                                  # 12x48
            *blk(64, 128), *blk(128, 128), nn.MaxPool2d((2, 1)),            # 6x48
            *blk(128, 192), *blk(192, 192), nn.MaxPool2d((2, 1)),           # 3x48
        )
        self.collapse = nn.Sequential(nn.Conv2d(192, 192, (3, 1), bias=False), nn.BatchNorm2d(192), nn.ReLU(True))
        self.seq = nn.Sequential(
            nn.Conv1d(192, 192, 3, padding=1, bias=False), nn.BatchNorm1d(192), nn.ReLU(True), nn.Dropout(0.2),
            nn.Conv1d(192, 192, 3, padding=1, bias=False), nn.BatchNorm1d(192), nn.ReLU(True),
        )
        self.out = nn.Conv1d(192, NCLS, 1)

    def forward(self, x):  # x: B,1,H,W -> B,NCLS,T
        return self.out(self.seq(self.collapse(self.cnn(x)).squeeze(2)))


def augment(g):
    """Augmentasi ringan pada gambar abu-abu (data asli sudah bervariasi; sintetis sudah didegradasi)."""
    h, w = g.shape
    if random.random() < 0.5:  # geser tepi / terpotong
        dx, dy = int(w * random.uniform(-0.06, 0.06)), int(h * random.uniform(-0.12, 0.12))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        g = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.4:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-3, 3), 1.0)
        g = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.35:  # perkecil lalu perbesar
        s = random.uniform(0.35, 0.8)
        g = cv2.resize(cv2.resize(g, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA), (w, h))
    if random.random() < 0.3:
        k = random.choice([3, 5])
        g = cv2.GaussianBlur(g, (k, k), random.uniform(0.4, 1.4))
    f = g.astype(np.float32)
    f = (f - 128) * random.uniform(0.7, 1.3) + 128 + random.uniform(-25, 25)
    if random.random() < 0.3:
        f += np.random.normal(0, random.uniform(2, 10), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def letterbox(g):
    """Resize mempertahankan rasio ke H x W; sisa diisi median gambar. Crop asli (~2,2:1) tidak lagi
    diperas ke 4:1 seperti sebelumnya, jadi bentuk huruf konsisten dengan data sintetis."""
    h, w = g.shape
    s = min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.full((H, W), int(np.median(g)), np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


class PlateDS(Dataset):
    """Semua gambar dimuat ke memori SEKALI (paralel): baca berkas dingin ~20 ms/gambar di mesin ini,
    sedangkan satu sampel dari memori hanya ~0,4 ms. Praproses model tidak berubah."""

    def __init__(self, items, train):
        self.items, self.train = items, train
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16) as ex:
            self.cache = list(ex.map(lambda it: cv2.imread(str(it[0]), cv2.IMREAD_GRAYSCALE), items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        label = self.items[i][1]
        g = self.cache[i]
        if g is None:
            g = np.full((H, W), 128, np.uint8)
        if self.train:
            g = augment(g)
        g = letterbox(g)
        x =torch.from_numpy((g.astype(np.float32) - 127.5) / 127.5)[None]
        y = [C2I[c] for c in label]
        return x, torch.tensor(y, dtype=torch.long), len(y), label


def collate(b):
    xs, ys, ls, labs = zip(*b)
    return torch.stack(xs), torch.cat(ys), torch.tensor(ls, dtype=torch.long), list(labs)


def read_real(split):
    return [(CORPUS / r["path"], r["plate"], r["source"]) for r in csv.DictReader(open(CORPUS / "manifest.csv", encoding="utf-8"))
            if r["split"] == split]


def read_synth():
    d = CORPUS / "synth"
    return [(d / r["path"], r["plate"]) for r in csv.DictReader(open(d / "synth_manifest.csv", encoding="utf-8"))]


def decode(logits):  # B,NCLS,T -> list[str] (CTC greedy)
    out = []
    for seq in logits.argmax(1).cpu().numpy():
        s, prev = [], 0
        for c in seq:
            if c != prev and c != 0:
                s.append(CHARS[c - 1])
            prev = c
        out.append("".join(s))
    return out


def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]


@torch.no_grad()
def predict(model, items, dev, bs=256):
    model.eval()
    ds = items if isinstance(items, PlateDS) else PlateDS([(p, l) for p, l, *_ in items], False)
    dl = DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=0)
    preds = []
    for x, _, _, _ in dl:
        preds += decode(model(x.to(dev)))
    return preds


def score(preds, items):
    n = len(items)
    exact = sum(p == it[1] for p, it in zip(preds, items)) / n * 100
    cer = float(np.mean([lev(p, it[1]) / len(it[1]) for p, it in zip(preds, items)])) * 100
    wrong_auto = sum(1 for p, it in zip(preds, items) if PLATE.match(p) and p != it[1]) / n * 100
    return exact, cer, wrong_auto


def train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    real = [(p, l) for p, l, _ in read_real("train")]
    synth = read_synth()
    val = read_real("val")
    print(f"real latih {len(real)} | sintetis {len(synth)} | validasi {len(val)} | perangkat {dev}", flush=True)

    t_load = time.time()
    val_ds = PlateDS([(p, l) for p, l, _ in val], False)
    ds = ConcatDataset([PlateDS(real, True), PlateDS(synth, True)])
    print(f"data dimuat ke memori dalam {time.time() - t_load:.0f} detik", flush=True)
    w = [a.real_share / len(real)] * len(real) + [(1 - a.real_share) / len(synth)] * len(synth)
    dl = DataLoader(ds, batch_size=a.bs, sampler=WeightedRandomSampler(w, a.per_epoch, replacement=True),
                    collate_fn=collate, num_workers=a.workers, persistent_workers=a.workers > 0, drop_last=True)

    model = Net().to(dev)
    print("parameter: %.2f juta" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best, log = (-1, 0), open(out / "train_log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "loss", "val_exact", "val_cer", "val_wrong_auto", "detik"])
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y, ls, _ in dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                lp = model(x).log_softmax(1).permute(2, 0, 1)  # T,B,C
            loss = ctc(lp.float(), y.to(dev), torch.full((x.size(0),), lp.size(0), dtype=torch.long), ls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        ex, cer, wa = score(predict(model, val_ds, dev), val)
        wr.writerow([ep, round(tot / len(dl), 4), round(ex, 2), round(cer, 2), round(wa, 2), int(time.time() - t0)])
        log.flush()
        mark = ""
        if (ex, -cer) > best:
            best = (ex, -cer)
            torch.save({"model": model.state_dict(), "chars": CHARS, "hw": (H, W)}, out / "best.pt")
            mark = "  <- terbaik"
        print(f"ep {ep:>3}  loss {tot / len(dl):.3f}  val benar {ex:5.1f}%  salah-karakter {cer:5.1f}%{mark}", flush=True)
    print("selesai. validasi terbaik:", best[0], "%")


def eval_test(ckpt, out):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=dev)
    model = Net().to(dev)
    model.load_state_dict(ck["model"])
    items = read_real("test")
    preds = predict(model, items, dev)
    ex, cer, wa = score(preds, items)
    print(f"\n=== SOAL UJI TERKUNCI (n={len(items)}) ===")
    print(f"OCR kustom: benar persis {ex:.1f}% | salah per karakter {cer:.1f}% | otomatis-salah {wa:.1f}%")
    for src in ("kaggle", "jmto"):
        sub = [(it, p) for it, p in zip(items, preds) if it[2] == src]
        e, c, w_ = score([p for _, p in sub], [it for it, _ in sub])
        print(f"  {'HP/Kaggle' if src == 'kaggle' else 'Tol/JMTO':<10} n={len(sub)}  benar {e:.1f}%  salah-karakter {c:.1f}%  otomatis-salah {w_:.1f}%")
    print(f"ML Kit terbaik: {MLKIT_BEST}% | TARGET GERBANG: >= {GATE:.1f}%  ->", "LULUS" if ex >= GATE else "BELUM LULUS")
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump([{"plate": it[1], "pred": p, "source": it[2]} for it, p in zip(items, preds)],
              open(Path(out) / "test_predictions.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/ocr_custom")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--per-epoch", dest="per_epoch", type=int, default=24000)
    ap.add_argument("--real-share", dest="real_share", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=0, help="0 = augmentasi di proses utama (data sudah di memori)")
    ap.add_argument("--eval-test", dest="eval_test", default=None)
    a = ap.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if a.eval_test:
        eval_test(a.eval_test, a.out)
    else:
        train(a)
