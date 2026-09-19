import argparse, json
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

ap = argparse.ArgumentParser(description="Predict with the stacked 8-model waste classifier")
ap.add_argument("image")
ap.add_argument("--outputs", default="outputs")
args = ap.parse_args()

out = Path(args.outputs)
info = json.loads((out / "dataset_info.json").read_text())
manifest = json.loads((out / "StackedMetaEnsemble" / "ensemble_manifest.json").read_text())
classes, size = info["classes"], tuple(info["image_size"])
img = tf.keras.utils.load_img(args.image, target_size=size)
x = tf.keras.utils.img_to_array(img)[None, ...]
features = []
for name in manifest["base_models"]:
    model = keras.models.load_model(out / name / "best.keras")
    features.append(model.predict(x, verbose=0))
meta = joblib.load(out / "StackedMetaEnsemble" / "meta_classifier.joblib")
p = meta.predict_proba(np.concatenate(features, axis=1))[0]
for i in np.argsort(p)[::-1][:5]:
    print(f"{classes[i]:25s} {p[i]:.4f}")
