"""Comprehensive Evaluation Module.

Satisfies PRD Functional Requirements:
- FR-08: Evaluate classification on clean, noisy, and denoised test images.
- Section 8.3: Controlled 3-condition experiment on held-out test data.
- Section 10: Metrics for CNN (Accuracy, Macro F1, Precision, Recall, Confusion Matrix)
  and DAE (MSE, MAE, PSNR).
"""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow import keras

from src import config
from src.prepare_data import load_dataset_arrays, add_gaussian_noise


def calculate_psnr_batch(clean: np.ndarray, recon: np.ndarray) -> float:
    """Calculates mean PSNR across a batch of images."""
    psnrs = []
    for c, r in zip(clean, recon):
        mse = np.mean((c - r) ** 2)
        if mse == 0:
            psnrs.append(100.0)
        else:
            psnrs.append(20 * np.log10(1.0 / np.sqrt(mse)))
    return float(np.mean(psnrs))


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes standard classification metrics with macro averaging."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    
    per_class = {}
    class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    for i, cls_name in enumerate(config.CLASSES):
        per_class[cls_name] = {
            "precision": float(class_prec[i]),
            "recall": float(class_rec[i]),
            "f1": float(class_f1[i]),
            "support": int(class_supp[i])
        }
        
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


def plot_evaluation_summary(results: Dict[str, Any], save_path=config.REPORTS_DIR / "evaluation_comparison.png"):
    """Generates comparison charts and 3 confusion matrices."""
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    
    # Row 1: Confusion Matrices for Clean, Noisy, Denoised
    conditions = [
        ("Clean Test Images", results["clean"]["confusion_matrix"], "#1e3a8a"),
        ("Noisy Test Images (Corrupted)", results["noisy"]["confusion_matrix"], "#7f1d1d"),
        ("Denoised by DAE -> CNN", results["denoised"]["confusion_matrix"], "#064e3b")
    ]
    
    for idx, (title, cm_data, title_color) in enumerate(conditions):
        ax = fig.add_subplot(gs[0, idx])
        cm_arr = np.array(cm_data)
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else ("Reds" if idx == 1 else "Greens"),
            xticklabels=config.CLASSES,
            yticklabels=config.CLASSES,
            cbar=False,
            ax=ax
        )
        ax.set_title(title, fontsize=12, fontweight="bold", color=title_color)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
    # Row 2: Left: Metric Bar Chart; Right: Per-Class F1 comparison
    ax_bar = fig.add_subplot(gs[1, 0:2])
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    x = np.arange(len(metrics))
    width = 0.25
    
    clean_vals = [results["clean"]["accuracy"], results["clean"]["macro_precision"], results["clean"]["macro_recall"], results["clean"]["macro_f1"]]
    noisy_vals = [results["noisy"]["accuracy"], results["noisy"]["macro_precision"], results["noisy"]["macro_recall"], results["noisy"]["macro_f1"]]
    denoised_vals = [results["denoised"]["accuracy"], results["denoised"]["macro_precision"], results["denoised"]["macro_recall"], results["denoised"]["macro_f1"]]
    
    rects1 = ax_bar.bar(x - width, clean_vals, width, label="Clean", color="#2563eb")
    rects2 = ax_bar.bar(x, noisy_vals, width, label="Noisy (Unprocessed)", color="#ef4444")
    rects3 = ax_bar.bar(x + width, denoised_vals, width, label="Denoised (DAE + CNN)", color="#10b981")
    
    ax_bar.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax_bar.set_title("Classification Performance Comparison Under Noise & Denoising", fontsize=13, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontsize=11)
    ax_bar.set_ylim([0, 1.05])
    ax_bar.legend(loc="upper right", fontsize=10)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
    
    # Auto-label bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax_bar.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
                            
    # Row 2: Right: DAE Reconstruction Summary Card
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis("off")
    dae_metrics = results["dae_reconstruction"]
    summary_text = (
        f"DAE RECONSTRUCTION BENCHMARK\n"
        f"-----------------------------------------\n"
        f"* Gaussian Noise Std:  {results['noise_factor']:.2f}\n"
        f"* Mean Squared Error:  {dae_metrics['mse']:.5f}\n"
        f"* Mean Abs Error (MAE):{dae_metrics['mae']:.5f}\n"
        f"* Corrupted Input PSNR: {dae_metrics['noisy_psnr']:.2f} dB\n"
        f"* Reconstructed PSNR:  {dae_metrics['reconstructed_psnr']:.2f} dB\n"
        f"* PSNR Improvement:    +{dae_metrics['psnr_gain']:.2f} dB\n\n"
        f"CONCLUSION:\n"
        f"Denoising recovers image fidelity and\n"
        f"restores classifier confidence on\n"
        f"noisy visual inputs."
    )
    ax_info.text(0.05, 0.5, summary_text, fontsize=10, family="monospace",
                 bbox=dict(boxstyle="round,pad=1", facecolor="#f8fafc", edgecolor="#cbd5e1"))
                 
    plt.suptitle("Smart Waste Classification: Comprehensive 3-Way Experimental Evaluation", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Evaluation comparison summary saved to: {save_path}")


def run_evaluation(noise_factor: float = config.NOISE_FACTOR) -> Dict[str, Any]:
    """Runs the 3-way evaluation experiment on unseen test data."""
    print("=== [Phase 4] Held-Out Test Evaluation Experiment (FR-08, Sec 8.3) ===")
    
    if not config.CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained CNN model not found at {config.CNN_MODEL_PATH}. Train CNN first.")
    if not config.DAE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained DAE model not found at {config.DAE_MODEL_PATH}. Train DAE first.")
        
    # Load Models
    print("Loading trained models...")
    cnn_model = keras.models.load_model(config.CNN_MODEL_PATH)
    dae_model = keras.models.load_model(config.DAE_MODEL_PATH)
    
    # Load clean test set
    X_test_clean, y_test = load_dataset_arrays("test")
    print(f"Loaded {len(X_test_clean)} held-out test images.")
    
    # Generate corrupted noisy test set
    X_test_noisy = add_gaussian_noise(X_test_clean, noise_factor=noise_factor, seed=config.RANDOM_SEED)
    
    # Generate DAE-denoised test set
    print("Denoising noisy test images through Convolutional DAE...")
    X_test_denoised = dae_model.predict(X_test_noisy, batch_size=config.BATCH_SIZE, verbose=0)
    X_test_denoised = np.clip(X_test_denoised, 0.0, 1.0)
    
    # 1. Clean condition
    print("Evaluating CNN on Clean Images...")
    preds_clean = cnn_model.predict(X_test_clean, batch_size=config.BATCH_SIZE, verbose=0)
    y_pred_clean = np.argmax(preds_clean, axis=1)
    clean_metrics = compute_classification_metrics(y_test, y_pred_clean)
    
    # 2. Noisy condition
    print("Evaluating CNN on Corrupted Noisy Images...")
    preds_noisy = cnn_model.predict(X_test_noisy, batch_size=config.BATCH_SIZE, verbose=0)
    y_pred_noisy = np.argmax(preds_noisy, axis=1)
    noisy_metrics = compute_classification_metrics(y_test, y_pred_noisy)
    
    # 3. Denoised condition
    print("Evaluating CNN on DAE-Denoised Images...")
    preds_denoised = cnn_model.predict(X_test_denoised, batch_size=config.BATCH_SIZE, verbose=0)
    y_pred_denoised = np.argmax(preds_denoised, axis=1)
    denoised_metrics = compute_classification_metrics(y_test, y_pred_denoised)
    
    # Autoencoder reconstruction metrics
    dae_mse = float(np.mean((X_test_clean - X_test_denoised) ** 2))
    dae_mae = float(np.mean(np.abs(X_test_clean - X_test_denoised)))
    noisy_psnr = calculate_psnr_batch(X_test_clean, X_test_noisy)
    recon_psnr = calculate_psnr_batch(X_test_clean, X_test_denoised)
    
    results = {
        "noise_factor": noise_factor,
        "clean": clean_metrics,
        "noisy": noisy_metrics,
        "denoised": denoised_metrics,
        "dae_reconstruction": {
            "mse": dae_mse,
            "mae": dae_mae,
            "noisy_psnr": noisy_psnr,
            "reconstructed_psnr": recon_psnr,
            "psnr_gain": recon_psnr - noisy_psnr
        }
    }
    
    # Save results to JSON
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EVALUATION_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[OK] Evaluation report saved to: {config.EVALUATION_REPORT_FILE}")
    
    # Generate visualization
    plot_evaluation_summary(results)
    
    # Print formatted markdown table
    print("\n" + "="*70)
    print("EXPERIMENTAL EVALUATION SUMMARY (Held-out Test Set)")
    print("="*70)
    print(f"{'Condition':<25} | {'Accuracy':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Macro F1':<10}")
    print("-" * 75)
    print(f"{'1. Clean Images':<25} | {clean_metrics['accuracy']*100:6.2f}%    | {clean_metrics['macro_precision']:9.3f}  | {clean_metrics['macro_recall']:9.3f}  | {clean_metrics['macro_f1']:9.3f}")
    print(f"{'2. Noisy Images':<25} | {noisy_metrics['accuracy']*100:6.2f}%    | {noisy_metrics['macro_precision']:9.3f}  | {noisy_metrics['macro_recall']:9.3f}  | {noisy_metrics['macro_f1']:9.3f}")
    print(f"{'3. DAE-Denoised Images':<25} | {denoised_metrics['accuracy']*100:6.2f}%    | {denoised_metrics['macro_precision']:9.3f}  | {denoised_metrics['macro_recall']:9.3f}  | {denoised_metrics['macro_f1']:9.3f}")
    print("-" * 75)
    print(f"DAE Reconstruction: MSE = {dae_mse:.5f}, PSNR = {recon_psnr:.2f} dB (Gain: +{recon_psnr - noisy_psnr:.2f} dB)")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    run_evaluation()
