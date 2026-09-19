from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from .evaluate import compute_metrics


def collect_base_probabilities(models, ds):
    """Return labels plus one probability matrix per base learner, in identical sample order."""
    y_true, chunks = [], {name: [] for name in models}
    for x, y in ds:
        y_true.append(np.argmax(y.numpy(), axis=1))
        for name, model in models.items():
            chunks[name].append(model.predict(x, verbose=0))
    y_true = np.concatenate(y_true)
    probs = {name: np.concatenate(parts) for name, parts in chunks.items()}
    return y_true, probs


def stack_features(probabilities, model_order):
    """Concatenate class probabilities: [model1 probs | ... | model8 probs]."""
    return np.concatenate([probabilities[name] for name in model_order], axis=1)


def fit_meta_ensemble(val_probabilities, y_val, model_order, seed=42, C=1.0):
    """Fit multinomial logistic regression only on validation predictions.

    The held-out test set is never used to fit or choose the stacker.
    """
    X_val = stack_features(val_probabilities, model_order)
    meta = LogisticRegression(max_iter=3000, C=C, random_state=seed)
    meta.fit(X_val, y_val)
    return meta


def evaluate_meta_ensemble(meta, probabilities, y_true, model_order, class_names):
    X = stack_features(probabilities, model_order)
    y_prob = meta.predict_proba(X)
    y_pred = meta.predict(X)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    return metrics, y_pred, y_prob


def save_meta_ensemble(meta, model_order, class_names, val_result, test_result, out_dir):
    out = Path(out_dir) / "StackedMetaEnsemble"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta, out / "meta_classifier.joblib")
    (out / "ensemble_manifest.json").write_text(json.dumps({
        "type": "stacked_meta_ensemble",
        "base_models": model_order,
        "feature_definition": "concatenated softmax probabilities from all base models",
        "meta_classifier": "scikit-learn LogisticRegression",
        "meta_training_split": "validation",
        "classes": class_names,
    }, indent=2))

    val_metrics, _, _ = val_result
    test_metrics, y_test_pred, _ = test_result
    (out / "metrics.json").write_text(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))
    report = classification_report(test_result[3] if len(test_result) > 3 else [], y_test_pred) if False else None
    return out


def save_ensemble_evaluation(meta, model_order, class_names, y_val, val_probs, y_test, test_probs, out_dir):
    out = Path(out_dir) / "StackedMetaEnsemble"
    out.mkdir(parents=True, exist_ok=True)
    val_metrics, val_pred, val_meta_prob = evaluate_meta_ensemble(meta, val_probs, y_val, model_order, class_names)
    test_metrics, test_pred, test_meta_prob = evaluate_meta_ensemble(meta, test_probs, y_test, model_order, class_names)

    joblib.dump(meta, out / "meta_classifier.joblib")
    (out / "ensemble_manifest.json").write_text(json.dumps({
        "type": "stacked_meta_ensemble",
        "base_models": model_order,
        "feature_definition": "concatenated softmax probabilities from all base models",
        "meta_classifier": "LogisticRegression",
        "meta_training_split": "validation",
        "classes": class_names,
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))
    pd.DataFrame(classification_report(y_test, test_pred, target_names=class_names,
                                       output_dict=True, zero_division=0)).T.to_csv(out / "classification_report.csv")

    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Stacked Meta-Ensemble — Test Confusion Matrix")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=170); plt.close(fig)

    # Coefficient magnitude gives an interpretable indication of each base model's contribution.
    coef = np.abs(meta.coef_)
    n_classes = len(class_names)
    contribution = []
    for i, name in enumerate(model_order):
        contribution.append((name, float(coef[:, i*n_classes:(i+1)*n_classes].mean())))
    cdf = pd.DataFrame(contribution, columns=["model", "mean_abs_meta_coefficient"]).sort_values(
        "mean_abs_meta_coefficient", ascending=False)
    cdf.to_csv(out / "base_model_contributions.csv", index=False)
    ax = cdf.set_index("model").plot(kind="bar", figsize=(10, 5), legend=False)
    ax.set_ylabel("Mean |meta coefficient|"); ax.set_title("Stacked ensemble — base-model contribution")
    ax.grid(axis="y", alpha=.25); fig = ax.get_figure(); fig.tight_layout()
    fig.savefig(out / "base_model_contributions.png", dpi=170); plt.close(fig)
    return val_metrics, test_metrics
