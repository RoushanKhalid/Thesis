# Thesis Cats-vs-Dogs Pipeline (`tcontext`)

End-to-end deep learning and retrieval experiment pipeline for binary image classification.
This version is tuned for smaller-data robustness using **25% class-balanced sampling** and explicit anti-overfitting controls.

## Project Structure

- `quick500_experiment.py`: full pipeline (sampling, EDA, training, evaluation, plots, report export)
- `query_demo.py`: CLI demo for incremental CLIP+vector memory update and retrieval
- `web_app.py`: Streamlit web app for visual classifier vs retrieval comparison
- `dataset_utils.py`: local dataset discovery, extraction, cleaning, and caching helpers
- `data/`: raw cache, cleaned dataset, and sampled subsets
- `artifacts/`: machine-readable outputs (`.json`, `.csv`) plus visual plots
- `reports/`: thesis-ready markdown reports
- `model/`: trained model export (`.keras`)

## Dataset Collection and Preparation

### 1) Data source

- Dataset: Microsoft Cats vs Dogs (Kaggle PetImages archive), accessed locally from TensorFlow Datasets cache.
- The pipeline expects a previously downloaded local zip in the TensorFlow datasets download cache.

### 2) Cleaning and validation

- Corrupted or unreadable files are skipped.
- All valid images are converted to RGB and re-saved as JPEG.
- A clean cache is created with up to `2500` images per class (`cats`, `dogs`).

### 3) Controlled sampling for experiments

- For this run, the script creates a **25% stratified sample per class**:
  - `625 cats + 625 dogs = 1250 total`
- Train/validation split is 80/20:
  - `1000 train`, `250 validation`

## Modeling Approach

### Classifier

- Backbone: `EfficientNetB0` (`imagenet` pretrained, frozen)
- Head: `GlobalAveragePooling2D -> Dropout(0.45) -> Dense(1, sigmoid, L2=1e-4)`
- Optimizer/Loss: `Adam(3e-4)` + `BinaryCrossentropy(label_smoothing=0.05)`
- Input size: `224x224`, batch size: `32`
- Data augmentation (train only): random horizontal flip, rotation, zoom
- Callbacks:
  - Early stopping (`patience=3`, monitor `val_loss`)
  - ReduceLROnPlateau (`factor=0.35`, `patience=1`)

### Retrieval baseline

- Feature extractor: `CLIP` image encoder (`openai/clip-vit-base-patch32`)
- Vector database: persistent `Chroma` collection on disk
- Similarity search: cosine top-`k=5` with majority-vote label decision

## Why the Previous 99% Was Overfitting-Prone

Small data plus a light head can memorize quickly and report unrealistically high train accuracy.
This version reduces that risk by combining:

- more data exposure (25% sample instead of 10%)
- augmentation
- dropout + L2 regularization
- label smoothing
- validation-driven stopping and LR scheduling

## Run

From workspace root:

- Install web demo dependency once:
  - `python -m pip install streamlit`
- One-click launcher (run experiment, capture log, start web demo):
  - `.\run_demo.ps1`

- Fixed seed:
  - `python tcontext/quick500_experiment.py --seed 777`
- Fresh random seed:
  - `python tcontext/quick500_experiment.py`
- Run with a retrieval query image (cat/dog prediction from vector DB):
  - `python tcontext/quick500_experiment.py --seed 777 --query-image "path/to/image.jpg"`

To capture full terminal output (including epoch logs):

- `python tcontext/quick500_experiment.py --seed 777 2>&1 | Tee-Object -FilePath logs/run_sampled25_terminal.log`

### Incremental Retrieval Demo (CLI)

- Add a new labeled image to vector memory:
  - `python tcontext/query_demo.py --add-image "path/to/new_image.jpg" --label cats`
- Query from vector memory:
  - `python tcontext/query_demo.py --query-image "path/to/test_image.jpg" --top-k 5`

### Web Demo for Thesis Presentation

- Launch:
  - `streamlit run tcontext/web_app.py`
- Web app gives:
  - model-vs-retrieval metric overview
  - computational cost profile (GPU availability, model size/params, retrieval index size)
  - incremental update-cost estimate for +5 / +10 images
  - EDA and training artifacts for visual explanation
  - live image upload and side-by-side predictions (EfficientNet vs CLIP+VectorDB)
  - **incremental retrieval memory update** (upload 1-10 new images with label, add instantly without retraining)

## Latest Experiment Snapshot (Seed 777)

- Sample size: `1250` (balanced)
- Epochs completed: `12`
- Best epoch by validation loss: `12`
- Potential overfitting epoch (gap > 0.08): `None`

### Classifier metrics

- Accuracy: `0.9800`
- Precision: `1.0000`
- Recall: `0.9600`
- F1: `0.9796`
- ROC-AUC: `0.9996`

### Retrieval metrics

- Accuracy: `0.9880`
- Precision: `1.0000`
- Recall: `0.9760`
- F1: `0.9879`

## Key Output Files

- `tcontext/reports/sampled25_report.md`
- `tcontext/reports/run_sampled25_log.md`
- `tcontext/reports/thesis_summary.md`
- `tcontext/artifacts/sampled25_summary.json`
- `tcontext/artifacts/sampled25_training_history.csv`
- `tcontext/artifacts/sampled25_comparison_metrics.csv`
- `logs/run_sampled25_terminal.log`
- `tcontext/artifacts/training_curves.png`
- `tcontext/artifacts/classifier_roc_curve.png`
- `tcontext/artifacts/comparison_confusion_matrices.png`

## Reproducibility Notes

- Set `--seed` for deterministic subset sampling and split behavior.
- The cleaned dataset cache is reused unless removed.
- If no local TFDS zip is available, run a one-time TFDS download first.
