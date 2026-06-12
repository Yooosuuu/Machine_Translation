import os
import json
import random
import re
from typing import List, Tuple, Iterable

import torch
from torch.utils.data import Dataset

from vocabulary import Vocabulary


def simple_tokenize(text: str) -> List[str]:
    """Basic tokenization: separate punctuation, split on whitespace."""
    text = text.strip()
    text = re.sub(r"([\.,!?;:\(\)\"])", r" \1 ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def read_parallel_corpus(src_path: str, tgt_path: str) -> List[Tuple[str, str]]:
    """Read two files with parallel sentences (one sentence per line).

    Returns list of (src, tgt) pairs.
    """
    pairs = []
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        for sline, tline in zip(fs, ft):
            s = sline.strip()
            t = tline.strip()
            if s and t:
                pairs.append((s, t))
    return pairs


def build_vocab_from_pairs(pairs: Iterable[Tuple[str, str]], min_freq: int = 1, max_size: int = None) -> Tuple[Vocabulary, Vocabulary]:
    """Build source and target vocabularies from sentence pairs."""
    src_vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    tgt_vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    for s, t in pairs:
        src_vocab.add_sentence(simple_tokenize(s.lower()))
        tgt_vocab.add_sentence(simple_tokenize(t.lower()))
    src_vocab.build()
    tgt_vocab.build()
    return src_vocab, tgt_vocab


def encode_pairs(pairs: Iterable[Tuple[str, str]], src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> List[Tuple[List[int], List[int]]]:
    """Encode sentence pairs into lists of token ids."""
    encoded = []
    for s, t in pairs:
        s_tok = simple_tokenize(s.lower())
        t_tok = simple_tokenize(t.lower())
        s_ids = src_vocab.encode(s_tok, add_sos_eos=True)
        t_ids = tgt_vocab.encode(t_tok, add_sos_eos=True)
        encoded.append((s_ids, t_ids))
    return encoded


def train_val_test_split(data: List, ratios=(0.9, 0.05, 0.05), seed: int = 42):
    """Split data into train, val, test sets according to given ratios."""
    random.seed(seed)
    n = len(data)
    idx = list(range(n))
    random.shuffle(idx)
    n1 = int(ratios[0] * n)
    n2 = n1 + int(ratios[1] * n)
    train = [data[i] for i in idx[:n1]]
    val = [data[i] for i in idx[n1:n2]]
    test = [data[i] for i in idx[n2:]]
    return train, val, test


class TranslationDataset(Dataset):
    """"Dataset for machine translation, holds pairs of (src_ids, tgt_ids) where each is a list of token ids."""
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_id: int = 0):
    """Collate function to pad sequences in a batch."""
    # batch: list of (src_tensor, tgt_tensor)
    src_seqs = [b[0] for b in batch]
    tgt_seqs = [b[1] for b in batch]
    src_lens = [s.size(0) for s in src_seqs]
    tgt_lens = [t.size(0) for t in tgt_seqs]
    src_max = max(src_lens)
    tgt_max = max(tgt_lens)

    src_padded = src_seqs[0].new_full((len(batch), src_max), pad_id)
    tgt_padded = tgt_seqs[0].new_full((len(batch), tgt_max), pad_id)

    for i, s in enumerate(src_seqs):
        src_padded[i, : s.size(0)] = s
    for i, t in enumerate(tgt_seqs):
        tgt_padded[i, : t.size(0)] = t

    return src_padded, torch.tensor(src_lens, dtype=torch.long), tgt_padded, torch.tensor(tgt_lens, dtype=torch.long)


def save_processed(processed: List[Tuple[List[int], List[int]]], path: str):
    """Save processed data using `torch.save` (binary) for consistency.

    The `path` should usually end with `.pt` (e.g. `data/processed/train.pt`).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(processed, path)


def load_processed(path: str) -> List[Tuple[List[int], List[int]]]:
    """Load processed data saved with `torch.save`.

    Returns a list of (src_ids, tgt_ids).
    """
    return torch.load(path)

def prepare_parallel_data(src_path: str,
                          tgt_path: str,
                          outdir: str = "data/processed",
                          min_freq: int = 1,
                          ratios=(0.9, 0.05, 0.05),
                          seed: int = 42,
                          save: bool = True,
                          max_len: int = 100):
    """Run full preprocessing pipeline and optionally save outputs.

    Returns: (src_vocab, tgt_vocab, train, val, test)
    - `train`, `val`, `test` are lists of (src_ids, tgt_ids).
    """
    pairs = read_parallel_corpus(src_path, tgt_path)
    src_vocab, tgt_vocab = build_vocab_from_pairs(pairs, min_freq=min_freq)

    def filter_by_length(encoded, max_len=max_len):
        return [(s, t) for s, t in encoded
                if len(s) <= max_len and len(t) <= max_len]

    # in prepare_parallel_data:
    encoded = encode_pairs(pairs, src_vocab, tgt_vocab)
    encoded = filter_by_length(encoded)  # filter out very long sentences to speed up training and avoid memory issues
    train, val, test = train_val_test_split(encoded, ratios=ratios, seed=seed)

    if save:
        os.makedirs(outdir, exist_ok=True)
        src_vocab.save(os.path.join(outdir, "vocab.src.json"))
        tgt_vocab.save(os.path.join(outdir, "vocab.tgt.json"))
        save_processed(train, os.path.join(outdir, "train.pt"))
        save_processed(val, os.path.join(outdir, "val.pt"))
        save_processed(test, os.path.join(outdir, "test.pt"))

    return src_vocab, tgt_vocab, train, val, test

