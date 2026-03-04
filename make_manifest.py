# make_manifest.py
from __future__ import annotations

import os
import pandas as pd

from VideoOpeningData import collect_samples, write_manifest_csv


def build_manifest_from_folders(
    dataset_root: str = os.path.join("data", "Dataset", "Videos"),
    out_csv: str = os.path.join("data", "Dataset", "manifest.csv"),
) -> str:
    samples, class_names = collect_samples(dataset_root)
    write_manifest_csv(samples, out_csv)
    print(f"[OK] Found {len(samples)} videos, {len(class_names)} classes")
    print(f"[OK] Saved manifest -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    build_manifest_from_folders()
