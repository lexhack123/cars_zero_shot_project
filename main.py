# main_zero_shot.py
# -------------------------------------------------------------
# Menu CLI pour Zero-Shot X-CLIP
# -------------------------------------------------------------

from __future__ import annotations

import os

from make_manifest import build_manifest_from_folders
from predict_zero_shot import predict_manifest
from evaluate_zero_shot import evaluate_predictions


DATASET_ROOT = os.path.join("data", "Dataset", "Videos")
MANIFEST_CSV = os.path.join("data", "Dataset", "manifest.csv")
PRED_CSV = os.path.join("CSV", "predictions_xclip.csv")


def menu():
    while True:
        print(
            "\n=== Zero-Shot Video Classification (X-CLIP) ===\n"
            f"Dataset root: {DATASET_ROOT}\n"
            "1) Build manifest.csv from folder dataset\n"
            "2) Predict (manifest.csv -> predictions CSV + console + PNG)\n"
            "3) Evaluate (scores + PNG curves, needs labels)\n"
            "0) Quit\n"
        )

        choice = input("Choose: ").strip()

        if choice == "1":
            build_manifest_from_folders(DATASET_ROOT, MANIFEST_CSV)

        elif choice == "2":
            if not os.path.exists(MANIFEST_CSV):
                print("manifest.csv not found. Run option 1 first.")
                continue

            predict_manifest(
                MANIFEST_CSV,
                out_csv=PRED_CSV,
                num_frames=8,
                show_cli=True,
                out_png=os.path.join("CSV", "predictions_summary.png"),
            )

        elif choice == "3":
            if not os.path.exists(PRED_CSV):
                print("predictions CSV not found. Run option 2 first.")
                continue

            try:
                evaluate_predictions(
                    PRED_CSV,
                    out_json=os.path.join("CSV", "metrics_xclip.json"),
                    out_history_png=os.path.join("PNG", "evaluation_history_xclip.png"),
                    # mets True si tu veux AUSSI la matrice
                    save_confusion_matrix=False,
                )
            except Exception as e:
                print("Evaluation error:", e)

        elif choice == "0":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    menu()