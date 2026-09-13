# Waste Image Classification with Keras — Transfer Learning Benchmark

This workspace benchmarks **8 pretrained CNN backbones** on a waste-image classification dataset using one consistent Keras pipeline:

- MobileNetV2
- EfficientNetB0
- ResNet50
- DenseNet121
- InceptionV3
- Xception
- NASNetMobile
- VGG16

It performs:

1. automatic dataset-folder discovery and RGB resize/batching;
2. training-time augmentation (flip, rotation, zoom, translation, contrast);
3. transfer learning with frozen ImageNet backbones;
4. controlled fine-tuning of the final part of each backbone;
5. EarlyStopping + ReduceLROnPlateau + best-checkpoint saving;
6. classification metrics: accuracy, weighted precision, recall, F1, ROC-AUC (when defined), top-2 accuracy during training;
7. per-model classification reports, confusion matrices, and learning curves;
8. cross-model comparison plot and CSV;
9. automatic best-classifier selection by weighted validation F1;
10. a train/validation/test performance curve for the selected best classifier.

## Dataset

The project is designed for datasets arranged as class folders:

```text
data/waste_dataset/
├── organic/
│   ├── image1.jpg
│   └── ...
└── recyclable/
    ├── image2.jpg
    └── ...
```

It also handles a parent directory that contains such a class-folder directory one or more levels below it.

The Mendeley **Waste Classification Dataset V3** (DOI `10.17632/n3gtgm9jxj.3`) is a suitable example. It contains 24,705 household-waste images in two categories: organic and recyclable. Download/extract the dataset yourself and set `data_dir` in `configs/config.yaml` to the extracted folder.

The same code works with multiclass datasets (for example paper/glass/plastic/metal/cardboard/etc.) as long as each class has its own folder.

## Setup

```bash
python3.10 -m venv ~/venvs/testcv
source ~/venvs/testcv/bin/activate            
pip install -r requirements.txt
```

GPU is strongly recommended because this workspace trains eight models.

## Configure

Edit `configs/config.yaml`:

```yaml
data_dir: data/waste_dataset
image_size: [224, 224]
batch_size: 32
epochs_head: 12
epochs_finetune: 12
```

For a 2-hour bachelor practical, start with 3-5 epochs for each phase and 2-3 backbones, then run all eight as a take-home experiment.

## Run the full benchmark

```bash
python train_all.py --config configs/config.yaml
or
nohup python -u train_all.py --config configs/config.yaml > training.log 2>&1 &
```

Outputs are written to `outputs/`.

## Main outputs

```text
outputs/
├── model_comparison.csv
├── model_comparison.png
├── best_classifier.keras
├── best_classifier_summary.json
├── best_classifier_train_val_test_curve.png
├── SUMMARY.md
├── dataset_info.json
├── MobileNetV2/
│   ├── best.keras
│   ├── history.csv
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── classification_report.csv
│   └── metrics.json
└── ... one directory per model
```

## Predict one image

```bash
python predict.py path/to/photo.jpg
```

## Notes for teaching / methodology

- **Augmentation is applied only during training** because it is embedded as Keras preprocessing layers in the model.
- The first stage trains only the new classification head.
- The second stage unfreezes the final fraction of the backbone and fine-tunes at a much smaller learning rate.
- **Early stopping** monitors validation loss and restores the best weights.
- The best model is selected using **weighted validation F1**. The test set is then reported separately, avoiding test-set leakage during model selection.
- Each architecture uses its official Keras `preprocess_input` function so the ImageNet weights receive inputs in the form they were trained to expect.

## Reproducibility

The config uses a fixed random seed. Exact GPU results can still vary slightly because some GPU kernels are nondeterministic.

## Citation / dataset provenance

Example dataset: Nnamoko, N., Barrowclough, J., & Procter, J. (2023), *Waste Classification Dataset*, Mendeley Data, V3, DOI: 10.17632/n3gtgm9jxj.3, CC BY 4.0.
