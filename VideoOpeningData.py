# VideoOpeningData.py
# -------------------------------------------------------------
# Equivalent de OpeningData.py mais pour des VIDEOS (.mp4, .avi, ...)
#
# Format attendu :
#   data/Dataset/Videos/<class_name>/*.mp4
#
# On peut :
#  - scanner le dataset
#  - créer un manifest.csv (path,label)
#  - split train/val/test (70/15/15) si tu veux
# -------------------------------------------------------------

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd

from utils import is_video_file, list_class_folders, ensure_dir


@dataclass
class VideoSample:
    path: str
    label_id: int
    label_name: str


def collect_samples(root_dir: str) -> Tuple[List[VideoSample], List[str]]:
    """
    Scan root_dir with subfolders = class names, and return samples + class_names.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

    class_names = list_class_folders(root_dir)
    if not class_names:
        raise ValueError(f"No class subfolders found in: {root_dir}")

    name_to_id = {name: i for i, name in enumerate(class_names)}
    samples: List[VideoSample] = []

    for cname in class_names:
        cdir = os.path.join(root_dir, cname)
        for fn in os.listdir(cdir):
            if not is_video_file(fn):
                continue
            path = os.path.join(cdir, fn)
            samples.append(VideoSample(path=path, label_id=name_to_id[cname], label_name=cname))

    if not samples:
        raise ValueError(f"No video files found under: {root_dir}")

    return samples, class_names


def split_samples(
    samples: List[VideoSample],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[VideoSample], List[VideoSample], List[VideoSample]]:
    """
    Stratified-ish split by shuffling within each class. Good enough for small projects.
    """
    random.seed(seed)
    by_class: dict[int, List[VideoSample]] = {}
    for s in samples:
        by_class.setdefault(s.label_id, []).append(s)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def write_manifest_csv(samples: List[VideoSample], out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv) or ".")
    df = pd.DataFrame([{"path": s.path, "label": s.label_name} for s in samples])
    df.to_csv(out_csv, index=False, encoding="utf-8")
