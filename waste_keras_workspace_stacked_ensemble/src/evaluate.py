from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)


def collect_predictions(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.append(np.argmax(y.numpy(), axis=1))
        y_prob.append(p)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob, class_names):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
    }
    try:
        if len(class_names) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def save_evaluation(model_name, model, test_ds, class_names, out_dir):
    out = Path(out_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, y_prob = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{model_name} — Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics


def save_training_curves(model_name, history_dict, out_dir):
    out = Path(out_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    h = history_dict
    epochs = range(1, len(h.get("loss", [])) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, h.get("loss", []), label="train loss")
    ax.plot(epochs, h.get("val_loss", []), label="validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"{model_name} — Loss"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / "loss_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, h.get("accuracy", []), label="train accuracy")
    ax.plot(epochs, h.get("val_accuracy", []), label="validation accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title(f"{model_name} — Accuracy"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / "accuracy_curve.png", dpi=160); plt.close(fig)


def save_comparison(results_df: pd.DataFrame, out_dir):
    out = Path(out_dir)
    results_df.to_csv(out / "model_comparison.csv", index=False)
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]
    usable = [m for m in metrics if m in results_df.columns]
    ax = results_df.set_index("model")[usable].plot(kind="bar", figsize=(12, 6))
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Transfer-learning model comparison")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=.25)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=170); plt.close(fig)


def save_best_model_summary(best_name, history, train_metrics, val_metrics, test_metrics, out_dir):
    out = Path(out_dir)
    # One figure showing training/validation curves plus final train/val/test accuracy bars.
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history["accuracy"]) + 1)
    ax.plot(epochs, history["accuracy"], label="train accuracy")
    ax.plot(epochs, history["val_accuracy"], label="validation accuracy")
    ax.axhline(test_metrics["accuracy"], linestyle="--", label=f"test accuracy = {test_metrics['accuracy']:.3f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Best classifier: {best_name} — train / validation / test")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "best_classifier_train_val_test_curve.png", dpi=180)
    plt.close(fig)

    summary = {
        "best_model": best_name,
        "selection_metric": "validation_f1_weighted",
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(out / "best_classifier_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
