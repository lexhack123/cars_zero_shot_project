# video_io.py
from __future__ import annotations

import numpy as np


def sample_frames_uniform(video_path: str, num_frames: int = 16) -> np.ndarray:
    """
    Return frames as uint8 numpy array (T, H, W, C).
    Tries decord first, falls back to OpenCV.
    """
    # --- Try decord ---
    try:
        from decord import VideoReader, cpu  # type: ignore
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise ValueError("empty video")
        idx = np.linspace(0, total - 1, num_frames).astype(int)
        frames = vr.get_batch(idx).asnumpy()
        # ensure shape (T,H,W,C)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("unexpected frame shape from decord")
        return frames
    except Exception:
        pass

    # --- Fallback OpenCV ---
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError(
            "Unable to read videos. Install either 'decord' or 'opencv-python'. "
            "Example: pip install decord opencv-python"
        ) from e

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # unknown length: just read sequentially and sample later
        frames_all = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_all.append(frame)
        cap.release()
        if not frames_all:
            raise ValueError(f"Empty/invalid video: {video_path}")
        frames_all = np.stack(frames_all, axis=0)
        total = frames_all.shape[0]
        idx = np.linspace(0, total - 1, num_frames).astype(int)
        return frames_all[idx]

    idx = np.linspace(0, total - 1, num_frames).astype(int).tolist()
    frames = []
    pos = 0
    target_i = 0
    next_target = idx[target_i] if idx else 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if pos == next_target:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            target_i += 1
            if target_i >= len(idx):
                break
            next_target = idx[target_i]
        pos += 1

    cap.release()

    if not frames:
        raise ValueError(f"Failed to sample frames: {video_path}")

    frames = np.stack(frames, axis=0).astype(np.uint8)

    # if video too short: pad by repeating last frame
    if frames.shape[0] < num_frames:
        last = frames[-1:]
        pad = np.repeat(last, num_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    return frames
