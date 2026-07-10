# Machine Translation

Neural machine translation project built in PyTorch. The repository compares two sequence-to-sequence families on a parallel corpus:

- an RNN encoder-decoder with Bahdanau attention
- a Transformer encoder-decoder with positional encoding

The codebase is notebook-driven, with reusable model and data utilities in `src/` and prepared datasets / checkpoints stored under `data/` and `models/`.

## Overview

The pipeline is intentionally simple:

1. Read aligned source and target sentences from raw text files.
2. Tokenize, lowercase, and build separate vocabularies for source and target languages.
3. Encode sentence pairs into token IDs and filter long examples.
4. Split the processed corpus into train, validation, and test sets.
5. Train and compare the RNN and Transformer models.
6. Decode translations with greedy inference.

## Repository Layout

```text
Machine_Translation/
├── README.md
├── requirements.txt
├── checkpoints/
├── data/
│   ├── raw/
│   │   ├── src.txt
│   │   └── tgt.txt
│   └── processed/
│       ├── train.pt
│       ├── val.pt
│       ├── test.pt
│       ├── vocab.src.json
│       └── vocab.tgt.json
├── models/
│   ├── rnn_emb128_hid256_lr0.001.pt
│   └── transformer_d256_h8_l3_lr0.0005.pt
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training_rnn_and_transformer.ipynb
│   └── 03_results_analysis.ipynb
└── src/
	├── attention.py
	├── dataset.py
	├── decoder.py
	├── encoder.py
	├── seq2seq.py
	├── transformer.py
	└── vocabulary.py
```

## Requirements

Install the Python dependencies listed in `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The project depends on:

- `torch`
- `numpy`
- `nltk`
- `jupyter`
- `ipykernel`
- `unbabel-comet`

## Data

The repository expects a parallel corpus in `data/raw/`:

- `data/raw/src.txt`: source-language sentences, one per line
- `data/raw/tgt.txt`: target-language sentences, one per line

The files must stay aligned line by line. Empty lines are ignored during preprocessing.

Processed artifacts are written to `data/processed/`:

- `train.pt`, `val.pt`, `test.pt`: tokenized sentence pairs saved with `torch.save`
- `vocab.src.json`, `vocab.tgt.json`: serialized vocabularies

## Preprocessing

All preprocessing logic lives in `src/dataset.py` and `src/vocabulary.py`.

Key behavior:

- tokenization is lightweight and punctuation-aware
- text is lowercased before vocabulary building and encoding
- source and target vocabularies are built independently
- special tokens are reserved for padding, start-of-sequence, end-of-sequence, and unknown tokens
- long pairs are filtered with `max_len`
- the final dataset is split into train / validation / test partitions

You can run preprocessing from the notebook or directly from Python:

```python
from src.dataset import prepare_parallel_data

src_vocab, tgt_vocab, train, val, test = prepare_parallel_data(
	src_path="data/raw/src.txt",
	tgt_path="data/raw/tgt.txt",
	outdir="data/processed",
	min_freq=1,
	ratios=(0.9, 0.05, 0.05),
	seed=42,
	save=True,
	max_len=30,
)
```

## Models

### RNN Seq2Seq

The RNN stack is split across:

- `src/encoder.py`: bidirectional GRU encoder with packed sequences
- `src/attention.py`: Bahdanau attention
- `src/decoder.py`: attention-based decoder
- `src/seq2seq.py`: training forward pass and greedy translation

This model uses teacher forcing during training and greedy decoding during inference.

### Transformer Seq2Seq

The Transformer implementation lives in `src/transformer.py`.

It provides the same high-level interface as the RNN model so the notebooks can switch between architectures with minimal changes. The model includes:

- learned source and target embeddings
- sinusoidal positional encoding
- stacked Transformer encoder and decoder layers
- padding masks and a causal mask for decoding

## Training

The main training workflow is in `notebooks/02_training_rnn_and_transformer.ipynb`.

Typical steps are:

1. Load the processed datasets and vocabularies.
2. Build either the RNN or Transformer model.
3. Train on `train.pt` with validation on `val.pt`.
4. Save the best checkpoints in `checkpoints/` or `models/`.

The repository already contains example trained weights:

- `models/rnn_emb128_hid256_lr0.001.pt`
- `models/transformer_d256_h8_l3_lr0.0005.pt`

## Inference

Both model families expose a `translate(...)` method for greedy decoding.

Example usage:

```python
translations, attentions = model.translate(src_batch, src_lengths, max_len=50)
```

The RNN model returns attention weights that can be visualized later. The Transformer model returns a placeholder attention tensor so it matches the same interface.

## Notebooks

### `notebooks/01_data_preparation.ipynb`

Used to build the processed dataset and vocabularies from the raw parallel corpus.

### `notebooks/02_training_rnn_and_transformer.ipynb`

Used to train and compare the RNN and Transformer approaches.

### `notebooks/03_results_analysis.ipynb`

Reserved for additional result inspection, qualitative examples, and comparison plots. (currently empty because all results and experiments are in the previous notebook)

## Module Guide

### `src/vocabulary.py`

Defines the `Vocabulary` class for building, saving, loading, encoding, and decoding token vocabularies.

### `src/dataset.py`

Contains data loading, tokenization, preprocessing, train/validation/test splitting, and the PyTorch dataset / collate utilities.

### `src/attention.py`

Implements additive Bahdanau attention.

### `src/encoder.py`

Implements the RNN encoder with packed padded sequences.

### `src/decoder.py`

Implements the attention-based decoder for step-by-step generation.

### `src/seq2seq.py`

Wraps the RNN encoder-decoder into a full training and inference model.

### `src/transformer.py`

Implements the Transformer-based translation model with a matching API.

## Reproducing The Project

1. Put a parallel corpus into `data/raw/src.txt` and `data/raw/tgt.txt`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `notebooks/01_data_preparation.ipynb` to create `data/processed/`.
4. Run `notebooks/02_training_rnn_and_transformer.ipynb` to train and save checkpoints + compute metrics.

## Notes

- The project is designed for experimentation in notebooks rather than a standalone CLI.
- `checkpoints/` is available for intermediate training state, while `models/` contains saved model files.
- If you change the corpus, regenerate the processed `.pt` and vocabulary files before retraining.
