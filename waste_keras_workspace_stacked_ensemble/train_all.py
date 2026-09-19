from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data import make_datasets, set_seed
from src.models import build_model, compile_model
from src.ensemble import collect_base_probabilities, fit_meta_ensemble, save_ensemble_evaluation
from src.evaluate import (
    collect_predictions, compute_metrics, save_evaluation, save_training_curves,
    save_comparison, save_best_model_summary
)


def merge_histories(*histories):
    keys = set().union(*(h.history.keys() for h in histories))
    return {k: sum([h.history.get(k, []) for h in histories], []) for k in keys}


def evaluate_split(model, ds, class_names):
    yt, yp, yprob = collect_predictions(model, ds)
    return compute_metrics(yt, yp, yprob, class_names)


def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, detected_root = make_datasets(
        cfg["data_dir"], tuple(cfg["image_size"]), cfg["batch_size"],
        cfg["validation_split"], cfg["test_split"], cfg["seed"]
    )
    (out_dir / "dataset_info.json").write_text(json.dumps({
        "detected_root": str(detected_root), "classes": class_names,
        "num_classes": len(class_names), "image_size": cfg["image_size"]
    }, indent=2))

    rows, histories = [], {}
    best = None
    trained_models = {}

    for model_name in cfg["models"]:
        print(f"\n=== {model_name} ===")
        tf.keras.backend.clear_session()
        model, base = build_model(model_name, len(class_names), tuple(cfg["image_size"]))
        compile_model(model, cfg["learning_rate_head"])

        model_out = out_dir / model_name; model_out.mkdir(parents=True, exist_ok=True)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg["patience"], restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=max(1, cfg["patience"] // 2), factor=0.2, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_out / "best.keras", monitor="val_loss", save_best_only=True, verbose=0
            ),
        ]

        h1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_head"], callbacks=callbacks, verbose=1)

        # Fine-tune only the last fraction of the pretrained backbone.
        base.trainable = True
        cut = int(len(base.layers) * (1.0 - cfg["fine_tune_fraction"]))
        for layer in base.layers[:cut]:
            layer.trainable = False
        compile_model(model, cfg["learning_rate_finetune"])
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_finetune"], callbacks=callbacks, verbose=1)

        history = merge_histories(h1, h2)
        histories[model_name] = history
        pd.DataFrame(history).to_csv(model_out / "history.csv", index=False)
        save_training_curves(model_name, history, out_dir)

        test_metrics = save_evaluation(model_name, model, test_ds, class_names, out_dir)
        train_metrics = evaluate_split(model, train_ds, class_names)
        val_metrics = evaluate_split(model, val_ds, class_names)
        row = {"model": model_name, **test_metrics,
               "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"]}
        rows.append(row)
        # Reload the best checkpoint so every ensemble member uses its best validation-loss weights.
        trained_models[model_name] = keras.models.load_model(model_out / "best.keras")

        candidate = (val_metrics["f1_weighted"], model_name, train_metrics, val_metrics, test_metrics)
        if best is None or candidate[0] > best[0]:
            best = candidate
            model.save(out_dir / "best_classifier.keras")

    # Optional stacked meta-ensemble: concatenate the 8 base softmax vectors and learn
    # a logistic-regression meta classifier on validation predictions only.
    ensemble_cfg = cfg.get("ensemble", {})
    if ensemble_cfg.get("enabled", True):
        model_order = list(cfg["models"])
        y_val, val_probs = collect_base_probabilities(trained_models, val_ds)
        y_test, test_probs = collect_base_probabilities(trained_models, test_ds)
        meta = fit_meta_ensemble(val_probs, y_val, model_order, seed=cfg["seed"], C=ensemble_cfg.get("C", 1.0))
        ensemble_val, ensemble_test = save_ensemble_evaluation(
            meta, model_order, class_names, y_val, val_probs, y_test, test_probs, out_dir
        )
        rows.append({
            "model": "StackedMetaEnsemble", **ensemble_test,
            "train_accuracy": None, "val_accuracy": ensemble_val["accuracy"],
            "val_f1_weighted": ensemble_val["f1_weighted"]
        })

    df = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    save_comparison(df, out_dir)
    best_val_f1, best_name, train_m, val_m, test_m = best
    save_best_model_summary(best_name, histories[best_name], train_m, val_m, test_m, out_dir)

    md = ["# Waste classification experiment summary", "",
          f"**Best classifier:** {best_name}",
          f"**Selection criterion:** weighted validation F1 = {best_val_f1:.4f}", "",
          "## Test-set comparison", "", df.to_markdown(index=False), "",
          "Artifacts: `model_comparison.png`, per-model confusion matrices and learning curves, "
          "`best_classifier_train_val_test_curve.png`, `classification_report.csv`, and `best_classifier.keras`.",
          "", "## Stacked meta-ensemble",
          "When enabled, `StackedMetaEnsemble/` contains a logistic-regression stacker trained on the "
          "validation softmax outputs of all eight transfer-learning models, plus test metrics, confusion "
          "matrix, contribution plot, and an inference manifest."]
    (out_dir / "SUMMARY.md").write_text("\n".join(md))
    print(df.to_string(index=False))
    print(f"\nBest classifier: {best_name} (selected by validation F1)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
