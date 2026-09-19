from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .data import augmentation_layer

MODEL_REGISTRY = {
    "MobileNetV2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "EfficientNetB0": (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input),
    "ResNet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    "DenseNet121": (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input),
    "InceptionV3": (keras.applications.InceptionV3, keras.applications.inception_v3.preprocess_input),
    "Xception": (keras.applications.Xception, keras.applications.xception.preprocess_input),
    "NASNetMobile": (keras.applications.NASNetMobile, keras.applications.nasnet.preprocess_input),
    "VGG16": (keras.applications.VGG16, keras.applications.vgg16.preprocess_input),
}


def build_model(name: str, num_classes: int, image_size=(224, 224), dropout=0.30):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name}. Choices: {list(MODEL_REGISTRY)}")
    base_cls, preprocess_input = MODEL_REGISTRY[name]
    base = base_cls(include_top=False, weights="imagenet", input_shape=(*image_size, 3))
    base.trainable = False

    inputs = keras.Input(shape=(*image_size, 3), name="image")
    x = augmentation_layer()(inputs)
    # Match each ImageNet backbone to its official Keras preprocessing function.
    x = layers.Lambda(preprocess_input, name=f"{name}_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name=f"{name}_waste_classifier")
    return model, base


def compile_model(model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )
