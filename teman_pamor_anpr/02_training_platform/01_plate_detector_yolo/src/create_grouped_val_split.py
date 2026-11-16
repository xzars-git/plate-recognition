#!/usr/bin/env python
"""
Create a grouped validation split from an augmented YOLO dataset.

Grouping rule:
- Files with suffix _rot90/_rot180/_rot270 are grouped with their original stem.
- Entire groups are moved together to avoid data leakage between train and val.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

ROT_SUFFIX_RE = re.compile(r"_rot(?:90|180|270)$")
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def base_stem(stem: str) -> str:
    return ROT_SUFFIX_RE.sub("", stem)


def build_groups(train_img_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for img in train_img_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in VALID_EXTS:
            continue
        key = base_stem(img.stem)
        groups.setdefault(key, []).append(img)
    return groups


def move_group_files(
    selected_groups: list[str],
    groups: dict[str, list[Path]],
    train_img_dir: Path,
    val_img_dir: Path,
    train_lbl_dir: Path,
    val_lbl_dir: Path,
) -> tuple[int, int, int]:
    moved_images = 0
    moved_labels = 0
    missing_labels = 0

    for group in selected_groups:
        for src_img in groups[group]:
            dst_img = val_img_dir / src_img.name
            shutil.move(str(src_img), str(dst_img))
            moved_images += 1

            src_lbl = train_lbl_dir / f"{src_img.stem}.txt"
            dst_lbl = val_lbl_dir / f"{src_img.stem}.txt"
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
                moved_labels += 1
            else:
                missing_labels += 1

    return moved_images, moved_labels, missing_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Create grouped val split from train")
    parser.add_argument(
        "--dataset",
        default="dataset/plate_detection_augmented",
        help="Dataset root containing images/train and labels/train",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (0 < r < 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--allow-non-empty-val",
        action="store_true",
        help="Allow split even if val folders already contain files",
    )
    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1")

    dataset = Path(args.dataset).resolve()
    train_img_dir = dataset / "images" / "train"
    val_img_dir = dataset / "images" / "val"
    train_lbl_dir = dataset / "labels" / "train"
    val_lbl_dir = dataset / "labels" / "val"

    for path in [train_img_dir, train_lbl_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    existing_val_images = sum(1 for p in val_img_dir.iterdir() if p.is_file())
    if existing_val_images > 0 and not args.allow_non_empty_val:
        raise RuntimeError(
            "Validation folder is not empty. Use --allow-non-empty-val if you really want to continue."
        )

    groups = build_groups(train_img_dir)
    group_keys = sorted(groups.keys())
    total_groups = len(group_keys)
    if total_groups == 0:
        raise RuntimeError("No training images found to split.")

    val_group_count = max(1, int(total_groups * args.val_ratio))

    rng = random.Random(args.seed)
    rng.shuffle(group_keys)
    selected_groups = group_keys[:val_group_count]

    moved_images, moved_labels, missing_labels = move_group_files(
        selected_groups,
        groups,
        train_img_dir,
        val_img_dir,
        train_lbl_dir,
        val_lbl_dir,
    )

    final_train_images = sum(1 for p in train_img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
    final_val_images = sum(1 for p in val_img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)

    print("=" * 70)
    print("GROUPED VAL SPLIT COMPLETE")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Seed: {args.seed}")
    print(f"Val ratio: {args.val_ratio:.2f}")
    print(f"Total groups: {total_groups}")
    print(f"Selected val groups: {val_group_count}")
    print(f"Moved images: {moved_images}")
    print(f"Moved labels: {moved_labels}")
    print(f"Missing labels: {missing_labels}")
    print(f"Final images train: {final_train_images}")
    print(f"Final images val: {final_val_images}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
