# predict_zero_shot.py
from __future__ import annotations

import os
import traceback
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# NEW: PNG summary plots
import matplotlib.pyplot as plt

from prompts import PROMPTS_BY_CLASS
from utils import ensure_dir, safe_relpath
from video_io import sample_frames_uniform
from xclip_backend import load_xclip


def _flatten_prompts(prompts_by_class: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    flat_prompts: List[str] = []
    prompt_to_class: List[str] = []
    for cls, prompts in prompts_by_class.items():
        for p in prompts:
            flat_prompts.append(p)
            prompt_to_class.append(cls)
    return flat_prompts, prompt_to_class


def _get_pixel_values_from_frames(processor: Any, frames: np.ndarray) -> torch.Tensor:
    """
    frames: (T,H,W,C) uint8 RGB
    Retour: pixel_values (1,T,3,H',W') float32
    """
    if frames is None or not isinstance(frames, np.ndarray):
        raise TypeError(f"frames must be numpy.ndarray, got {type(frames)}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3), got shape={getattr(frames,'shape',None)}")

    # list of frames (H,W,C) numpy
    frame_list = [np.ascontiguousarray(frames[i]) for i in range(frames.shape[0])]

    img_proc = getattr(processor, "image_processor", None) or getattr(processor, "feature_extractor", None)
    if img_proc is None:
        raise AttributeError("Processor has no image_processor/feature_extractor (cannot build pixel_values).")

    out = img_proc(images=frame_list, return_tensors="pt")
    pixel_values = out.get("pixel_values", None)
    if pixel_values is None:
        raise ValueError("image_processor did not return 'pixel_values'")

    # pixel_values: (T,3,H',W') -> (1,T,3,H',W')
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)

    if pixel_values.ndim != 5:
        raise ValueError(f"pixel_values must be 5D (1,T,3,H,W), got shape={tuple(pixel_values.shape)}")

    return pixel_values


@torch.inference_mode()
def predict_one_xclip(
    video_path: str,
    model: Any,
    processor: Any,
    device: str,
    prompts_by_class: Dict[str, List[str]],
    num_frames: int = 8,
) -> Tuple[str, Dict[str, float]]:
    frames = sample_frames_uniform(video_path, num_frames=num_frames)  # (T,H,W,C) uint8 RGB

    class_names = list(prompts_by_class.keys())
    flat_prompts, prompt_to_class = _flatten_prompts(prompts_by_class)

    # 1) tokens texte
    text_inputs = processor(text=flat_prompts, return_tensors="pt", padding=True)
    # 2) pixel_values vidéo (on ne dépend PAS de processor(videos=...))
    pixel_values = _get_pixel_values_from_frames(processor, frames)

    # move to device
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    pixel_values = pixel_values.to(device)

    # forward
    outputs = model(**text_inputs, pixel_values=pixel_values)

    if not hasattr(outputs, "logits_per_video"):
        raise AttributeError("Model output has no 'logits_per_video' (wrong model/processor pairing?).")

    logits = outputs.logits_per_video[0]  # (nb_prompts,)

    # moyenne des prompts par classe
    scores = []
    for cls in class_names:
        idxs = [i for i, c in enumerate(prompt_to_class) if c == cls]
        scores.append(logits[idxs].mean())
    scores = torch.stack(scores, dim=0)

    probs = torch.softmax(scores, dim=0).detach().cpu().numpy()
    probs_by_class = {c: float(p) for c, p in zip(class_names, probs)}
    pred = class_names[int(np.argmax(probs))]
    return pred, probs_by_class


def save_predictions_png(out_df: pd.DataFrame, out_png: str) -> None:
    """
    Enregistre des graphiques PNG pendant le "Predict" (sans labels requis) :
      1) Distribution des classes prédites
      2) Confiance moyenne (max proba) par classe
    """
    df_ok = out_df[out_df["pred"] != "__ERROR__"].copy()
    if df_ok.empty:
        print("[WARN] No successful predictions -> PNG not generated.")
        return

    # 1) comptage par classe
    counts = df_ok["pred"].value_counts()
    # pour avoir un ordre stable (alphabétique)
    counts = counts.reindex(sorted(counts.index), fill_value=0)

    # 2) confiance = max(p_classe) par vidéo
    p_cols = [c for c in df_ok.columns if c.startswith("p_")]
    if p_cols:
        df_ok["confidence"] = df_ok[p_cols].max(axis=1)
        conf_mean = df_ok.groupby("pred")["confidence"].mean().reindex(counts.index)
    else:
        conf_mean = pd.Series(index=counts.index, data=np.nan)

    # dossier sortie
    ensure_dir(os.path.dirname(out_png) or ".")

    # --- Plot 1: distribution ---
    plt.figure(figsize=(10, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Predicted class distribution (X-CLIP zero-shot)")
    plt.xlabel("Class")
    plt.ylabel("Number of videos")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # --- Plot 2: confiance moyenne ---
    out_png2 = os.path.splitext(out_png)[0] + "_confidence.png"
    plt.figure(figsize=(10, 5))
    plt.bar(conf_mean.index.astype(str), conf_mean.values)
    plt.title("Mean confidence by predicted class")
    plt.xlabel("Class")
    plt.ylabel("Mean max probability")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png2, dpi=200)
    plt.close()

    print(f"[OK] Saved PNG -> {safe_relpath(out_png)}")
    print(f"[OK] Saved PNG -> {safe_relpath(out_png2)}")


def predict_manifest(
    manifest_csv: str,
    out_csv: str = os.path.join("CSV", "predictions_xclip.csv"),
    num_frames: int = 8,
    model_name: str = "microsoft/xclip-base-patch32",
    show_cli: bool = True,
    out_png: str = os.path.join("CSV", "predictions_summary.png"),
) -> str:
    df = pd.read_csv(manifest_csv)
    if "path" not in df.columns:
        raise ValueError("manifest.csv must contain a 'path' column")
    has_label = "label" in df.columns

    backend = load_xclip(model_name=model_name)
    model, processor, device = backend.model, backend.processor, backend.device

    ensure_dir(os.path.dirname(out_csv) or ".")

    rows = []
    for i, r in tqdm(df.iterrows(), total=len(df), desc="Predict"):
        path = os.path.normpath(str(r["path"]))
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video not found: {path} (cwd={os.getcwd()})")

            pred, probs = predict_one_xclip(
                path,
                model,
                processor,
                device,
                prompts_by_class=PROMPTS_BY_CLASS,
                num_frames=num_frames,
            )

            row = {"path": path, "pred": pred}
            if has_label:
                row["label"] = str(r["label"])
            for cls, p in probs.items():
                row[f"p_{cls}"] = p
            rows.append(row)

        except Exception:
            row = {"path": path, "pred": "__ERROR__", "error": traceback.format_exc()}
            if has_label:
                row["label"] = str(r["label"])
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    # ---- AFFICHAGE CLI ----
    if show_cli:
        df_ok = out_df[out_df["pred"] != "__ERROR__"].copy()
        n_err = int((out_df["pred"] == "__ERROR__").sum())

        print("\n=== Prediction summary ===")
        print(f"Total videos: {len(out_df)} | OK: {len(df_ok)} | Errors: {n_err}")

        if not df_ok.empty:
            print("\nClass counts:")
            print(df_ok["pred"].value_counts().to_string())

            # petit tableau (10 premières prédictions)
            cols = ["path", "pred"] + (["label"] if "label" in out_df.columns else [])
            print("\nSample results (first 10):")
            print(df_ok[cols].head(10).to_string(index=False))

            # accuracy rapide si label existe
            if "label" in df_ok.columns:
                acc = float((df_ok["pred"].astype(str) == df_ok["label"].astype(str)).mean())
                print(f"\nQuick accuracy (pred == label): {acc:.4f}")

        if n_err:
            print("\n[WARN] First error sample:")
            print(out_df.loc[out_df["pred"] == "__ERROR__", ["path", "error"]].head(1).to_string(index=False))

    # ---- PNG ----
    try:
        save_predictions_png(out_df, out_png)
    except Exception as e:
        print("[WARN] Could not save PNG:", e)

    print(f"\n[OK] Saved predictions -> {safe_relpath(out_csv)}")
    return out_csv