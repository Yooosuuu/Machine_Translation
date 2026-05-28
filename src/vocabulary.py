import json
from collections import Counter
from typing import List, Iterable


class Vocabulary:
    """Simple token vocabulary for the Machine Translation model.

    Keeps mappings token -> id and id -> token and handles special tokens.
    """

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self, min_freq: int = 1, max_size: int = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.freqs = Counter()
        self.token2id = {}
        self.id2token = []

    def add_sentence(self, tokens: Iterable[str]):
        self.freqs.update(tokens)

    def build(self):
        """Build the vocabulary after all sentences have been added. Applies min_freq and max_size constraints."""
        # start with specials (PAD for padding, SOS/EOS for start/end of sentence, UNK for unknown tokens)
        specials = [self.PAD, self.SOS, self.EOS, self.UNK]
        tokens = [t for t, f in self.freqs.items() if f >= self.min_freq]
        tokens.sort(key=lambda t: (-self.freqs[t], t))
        if self.max_size:
            tokens = tokens[: self.max_size - len(specials)]

        self.id2token = specials + tokens
        self.token2id = {t: i for i, t in enumerate(self.id2token)}

    def __len__(self):
        return len(self.id2token)

    def token_to_id(self, token: str) -> int:
        return self.token2id.get(token, self.token2id[self.UNK])

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.id2token):
            return self.id2token[idx]
        return self.UNK

    def encode(self, tokens: List[str], add_sos_eos: bool = True) -> List[int]:
        """Encode a list of tokens into a list of token ids, optionally adding SOS and EOS tokens."""
        ids = [self.token_to_id(t) for t in tokens]
        if add_sos_eos:
            ids = [self.token2id[self.SOS]] + ids + [self.token2id[self.EOS]]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        """Decode a list of token ids into a list of tokens."""
        tokens = [self.id_to_token(i) for i in ids]
        return tokens

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "id2token": self.id2token,
                "min_freq": self.min_freq,
                "max_size": self.max_size
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = cls(min_freq=data.get("min_freq", 1), max_size=data.get("max_size"))
        v.id2token = data["id2token"]
        v.token2id = {t: i for i, t in enumerate(v.id2token)}
        return v
