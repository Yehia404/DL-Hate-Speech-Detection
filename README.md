# Hate Speech Detection with Deep Learning

A deep learning project that classifies tweets into three categories — **Hate**, **Offensive**, or **Neither** — using a Bidirectional LSTM network built with TensorFlow/Keras.

Developed as a team project for a Deep Learning course (ASU, Semester 9).

## What It Does

The model takes raw tweet text and predicts which of three classes it belongs to:

| Label | Meaning |
|-------|---------|
| `0` | Hate speech |
| `1` | Offensive language |
| `2` | Neither |

The pipeline covers the full workflow: data exploration and visualization, text cleaning and preprocessing (tokenization, padding), model training with imbalance handling, training-curve visualization, and evaluation with confusion matrices and per-class metrics. There is also a small interactive demo where you can type a sentence and get a live prediction.

## Model Architecture

```
Embedding (vocab 5,000, dim 128)
  → SpatialDropout1D (0.3)
  → Bidirectional LSTM (64 units, dropout 0.3, recurrent dropout 0.3)
  → Dense (32, ReLU) → Dropout (0.3)
  → Dense (3, Softmax)
```

Preprocessing uses a Keras `Tokenizer` capped at 5,000 words (with an `<OOV>` token) and sequences padded/truncated to a maximum length of 100.

## Notebooks

| Notebook | Experiment |
|----------|-----------|
| `main_weights.ipynb` | Baseline pipeline with **class weighting** (`class_weight='balanced'`) to counter class imbalance |
| `main_augment.ipynb` | **Data augmentation**: oversamples minority classes by synonym-replacing random words (WordNet-based) in training tweets |
| `main_crossval_weights.ipynb` | **Stratified k-fold cross-validation** of the class-weighted model for more reliable performance estimates, followed by final training on the full dev set |

Each notebook is self-contained and can be run top-to-bottom.

## Dataset

The dataset is loaded directly from a public CSV (hate-speech / offensive-language annotated tweets) hosted online, so no local download is needed — the notebooks fetch it at startup.

## Getting Started

```bash
pip install tensorflow pandas numpy scikit-learn nltk matplotlib seaborn
```

Then open any notebook in Jupyter and run the cells in order:

```bash
jupyter notebook main_weights.ipynb
```

NLTK resources (WordNet) are downloaded inside the augmentation notebook on first use.

## Project Report

`Team 1 DL Project Report.pdf` contains the full write-up: methodology, experiments, and results.

## Results

Trained with early stopping (patience 5, best weights restored) and `ReduceLROnPlateau`. Evaluation uses accuracy, precision/recall/F1 per class, and confusion matrices — see the notebooks and report for detailed numbers per experiment.
