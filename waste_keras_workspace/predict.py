import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--model", default="outputs/best_classifier.keras")
ap.add_argument("--dataset-info", default="outputs/dataset_info.json")
args = ap.parse_args()

info = json.loads(Path(args.dataset_info).read_text())
classes = info["classes"]; size = tuple(info["image_size"])
model = keras.models.load_model(args.model)
img = tf.keras.utils.load_img(args.image, target_size=size)
x = tf.keras.utils.img_to_array(img)[None, ...]
p = model.predict(x, verbose=0)[0]
for i in np.argsort(p)[::-1][:5]:
    print(f"{classes[i]:25s} {p[i]:.4f}")
