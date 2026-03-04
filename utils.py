# utils.py
from __future__ import annotations

import os
from typing import Iterable, List


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def is_video_file(filename: str) -> bool:
    return filename.lower().endswith(VIDEO_EXTS)


def list_class_folders(root_dir: str) -> List[str]:
    """Return sorted list of subdirectories (class folders)."""
    dirs = []
    for d in os.listdir(root_dir):
        p = os.path.join(root_dir, d)
        if os.path.isdir(p):
            dirs.append(d)
    return sorted(dirs)


def safe_relpath(path: str) -> str:
    """Pretty path for printing."""
    try:
        return os.path.relpath(path)
    except Exception:
        return path
