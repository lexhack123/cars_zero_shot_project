# evaluate_zero_shot.py
# -------------------------------------------------------------
# Evaluation pour zero-shot :
# - lit le CSV de predictions (doit contenir: label, pred, p_<classe>*)
# - calcule accuracy / precision / recall / f1 (macro)
# - (NOUVEAU) génère un PNG "history" (courbes) comme un training curve,
#   mais sur l'axe X = nombre de vidéos traitées (cumulatif)
# - (optionnel) confusion matrix + PNG
# -------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from utils import ensure_dir, safe_relpath


def _safe_prob(val: float, eps: float = 1e-12) -> float:
    try:
        v = float(val)
    except Exception:
        return eps
    if not np.isfinite(v):
        return eps
    return float(max(v, eps))


def _macro_precision_recall_from_cm(cm: np.ndarray) -> Tuple[float, float]:
    """
    cm shape: (C,C), rows=true, cols=pred
    precision_k = TP/(TP+FP)
    recall_k    = TP/(TP+FN)
    macro = mean over classes (zero_division=0)
    """
    cm = cm.astype(np.float64, copy=False)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)

    return float(np.mean(prec)), float(np.mean(rec))


def _save_eval_history_png(
    df: pd.DataFrame,
    labels: List[str],
    out_png: str,
    title: str = "Evaluation history (cumulative)",
) -> None:
    """
    Produit 4 courbes (2x2) sur l'axe X = nombre de vidéos traitées:
      - Accuracy (cumulative)
      - Loss (mean -log(p_true)) si p_<label> existe
      - Precision macro (cumulative, via confusion matrix cumulée)
      - Recall macro (cumulative, via confusion matrix cumulée)

    NOTE: en zero-shot il n’y a pas d’époques => l’axe X = vidéos traitées.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[WARN] matplotlib not available -> no history PNG:", e)
        return

    ensure_dir(os.path.dirname(out_png) or ".")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    C = len(labels)
    cm = np.zeros((C, C), dtype=np.int64)

    acc_hist: List[float] = []
    loss_hist: List[float] = []
    prec_hist: List[float] = []
    rec_hist: List[float] = []

    correct = 0
    loss_sum = 0.0
    n = 0

    p_cols = {c: c for c in df.columns if c.startswith("p_")}

    for _, row in df.iterrows():
        y_t = str(row["label"])
        y_p = str(row["pred"])
        if y_t not in label_to_idx or y_p not in label_to_idx:
            continue

        i = label_to_idx[y_t]
        j = label_to_idx[y_p]
        cm[i, j] += 1

        n += 1
        if i == j:
            correct += 1

        # Loss = -log(p_true) if available
        col = f"p_{y_t}"
        if col in p_cols:
            p_true = _safe_prob(row[col])
            loss_sum += -float(np.log(p_true))
            loss_hist.append(loss_sum / n)
        else:
            loss_hist.append(np.nan)

        acc_hist.append(correct / n)
        prec, rec = _macro_precision_recall_from_cm(cm)
        prec_hist.append(prec)
        rec_hist.append(rec)

    if n == 0:
        print("[WARN] No samples to plot -> no history PNG.")
        return

    x = np.arange(1, n + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title)

    # Accuracy
    axs[0, 0].plot(x, acc_hist, marker="o", linewidth=1)
    axs[0, 0].set_title("Accuracy (cumulative)")
    axs[0, 0].set_xlabel("Videos processed")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].set_ylim(0, 1)

    # Loss
    axs[0, 1].plot(x, loss_hist, marker="o", linewidth=1)
    axs[0, 1].set_title("Loss (mean -log(p_true))")
    axs[0, 1].set_xlabel("Videos processed")
    axs[0, 1].set_ylabel("Loss")

    # Precision
    axs[1, 0].plot(x, prec_hist, marker="o", linewidth=1)
    axs[1, 0].set_title("Precision (macro, cumulative)")
    axs[1, 0].set_xlabel("Videos processed")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].set_ylim(0, 1)

    # Recall
    axs[1, 1].plot(x, rec_hist, marker="o", linewidth=1)
    axs[1, 1].set_title("Recall (macro, cumulative)")
    axs[1, 1].set_xlabel("Videos processed")
    axs[1, 1].set_ylabel("Recall")
    axs[1, 1].set_ylim(0, 1)

    for ax in axs.ravel():
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[OK] Eval history PNG -> {safe_relpath(out_png)}")


def evaluate_predictions(
    predictions_csv: str,
    out_json: str = os.path.join("CSV", "metrics_xclip.json"),
    out_history_png: str = os.path.join("PNG", "evaluation_history_xclip.png"),
    out_cm_png: str = os.path.join("PNG", "confusion_matrix_xclip.png"),
    save_confusion_matrix: bool = False,
) -> dict:
    df_all = pd.read_csv(predictions_csv)

    if "label" not in df_all.columns:
        raise ValueError("No 'label' column found. For evaluation, your manifest must include labels.")

    # Remove errors
    df = df_all[df_all["pred"] != "__ERROR__"].copy()
    if df.empty:
        raise ValueError("No valid predictions to evaluate (all errors).")

    y_true = df["label"].astype(str).tolist()
    y_pred = df["pred"].astype(str).tolist()

    labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p, r, f1 = float(p), float(r), float(f1)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": acc,
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_samples": int(len(df)),
        "n_errors_skipped": int((df_all["pred"] == "__ERROR__").sum()),
    }

    ensure_dir(os.path.dirname(out_json) or ".")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Print summary in CLI
    print("\n=== Evaluation summary ===")
    print(f"Samples: {metrics['n_samples']} | Errors skipped: {metrics['n_errors_skipped']}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {p:.4f}")
    print(f"Recall (macro): {r:.4f}")
    print(f"F1 (macro): {f1:.4f}")

    # NEW: Save "history" curves PNG (not a matrix)
    _save_eval_history_png(
        df=df,
        labels=labels,
        out_png=out_history_png,
        title=f"Evaluation history: {os.path.basename(predictions_csv)}",
    )

    # OPTIONAL: Save confusion matrix figure
    if save_confusion_matrix:
        try:
            import matplotlib.pyplot as plt

            ensure_dir(os.path.dirname(out_cm_png) or ".")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title("Confusion Matrix (X-CLIP)")
            fig.colorbar(im)

            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(labels)

            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")

            # annotate cells
            thresh = cm.max() * 0.6 if cm.size else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )

            fig.tight_layout()
            fig.savefig(out_cm_png, dpi=200)
            plt.close(fig)
            print(f"[OK] Confusion matrix -> {safe_relpath(out_cm_png)}")
        except Exception as e:
            print("[WARN] Could not save confusion matrix PNG:", e)

    print(f"[OK] Metrics -> {safe_relpath(out_json)}")
    return metrics