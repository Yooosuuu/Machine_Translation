import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Self-attention is order-agnostic, so we add a fixed per-position vector to
    the embeddings to encode word order:
        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 1 / 10000^(2i/d_model), computed in log-space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model); buffer = moves with .to(device) but not trained
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """Encoder-decoder Transformer with the same API as the RNN Seq2Seq model.

    Exposes forward(src, src_lens, tgt, teacher_forcing_ratio) -> logits and
    translate(src, src_lens, max_len) -> (tokens, attn), so the notebook can
    switch between models via a flag with no other changes. Built from PyTorch's
    nn.TransformerEncoder / nn.TransformerDecoder; only PositionalEncoding is
    hand-written.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 device: torch.device,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 pad_idx: int = 0,
                 sos_idx: int = 1,
                 eos_idx: int = 2):
        """
        Args:
            input_dim: source vocabulary size
            output_dim: target vocabulary size
            device: cpu / cuda (used to build tensors in translate)
            d_model: model/embedding width (must be divisible by n_heads)
            n_heads: number of attention heads
            n_layers: number of encoder and decoder layers
            d_ff: hidden size of the feed-forward sub-layer
            dropout: dropout used throughout
            max_len: max sequence length for the positional encoding
            pad_idx, sos_idx, eos_idx: special token ids
        """
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        # One embedding per language; padding_idx pins the <PAD> row to zero
        self.src_embedding = nn.Embedding(input_dim, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(output_dim, d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(d_model)  # paper scales embeddings by sqrt(d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        # batch_first=True -> tensors are (batch, seq, d_model), as elsewhere here
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.fc_out = nn.Linear(d_model, output_dim)

    def make_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Bool mask, shape (batch, seq_len), True at <PAD> positions to ignore."""
        return seq == self.pad_idx

    def make_causal_mask(self, size: int) -> torch.Tensor:
        """Bool mask, shape (size, size), True above the diagonal (future positions
        a target token must not attend to). Bool matches the padding masks above."""
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=self.device),
            diagonal=1,
        )

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        """Encode the source once.

        Args:
            src: source token ids, shape (batch, src_len)
            src_padding_mask: bool mask, shape (batch, src_len)

        Returns:
            memory: encoder outputs, shape (batch, src_len, d_model)
        """
        embedded = self.pos_encoder(self.src_embedding(src) * self.scale)
        return self.transformer_encoder(embedded, src_key_padding_mask=src_padding_mask)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """Training pass: predicts the whole target in parallel (causal-masked).

        Args:
            src: source token ids, shape (batch, src_len)
            src_lens: unused (kept to match the RNN signature)
            tgt: target token ids incl. <SOS>/<EOS>, shape (batch, tgt_len)
            teacher_forcing_ratio: unused (causal mask gives full teacher forcing)

        Returns:
            outputs: logits, shape (batch, tgt_len, output_dim). outputs[:, 0] is
            left as zeros so the existing loss-slicing output[:, 1:] lines up.
        """
        batch_size, tgt_len = tgt.size()

        src_padding_mask = self.make_padding_mask(src)

        # Decoder reads the target shifted right by one: tokens 0..L-2 predict 1..L-1
        tgt_input = tgt[:, :-1]
        tgt_padding_mask = self.make_padding_mask(tgt_input)
        causal_mask = self.make_causal_mask(tgt_input.size(1))

        memory = self.encode(src, src_padding_mask)

        tgt_embedded = self.pos_encoder(self.tgt_embedding(tgt_input) * self.scale)
        decoded = self.transformer_decoder(
            tgt_embedded, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )  # (batch, tgt_len-1, d_model)
        logits = self.fc_out(decoded)

        outputs = torch.zeros(batch_size, tgt_len, self.output_dim, device=self.device)
        outputs[:, 1:] = logits
        return outputs

    @torch.no_grad()
    def translate(self, src: torch.Tensor, src_lens: torch.Tensor, max_len: int = 50):
        """Greedy decoding: generate the target one token at a time.

        Args:
            src: source token ids, shape (batch, src_len)
            src_lens: unused (kept to match the RNN signature)
            max_len: maximum number of tokens to generate

        Returns:
            translations: generated token ids, shape (batch, generated_len)
            attentions: zeros placeholder, shape (batch, generated_len, src_len),
                        returned only to match the RNN's (tokens, attn) signature
        """
        self.eval()
        batch_size, src_len = src.size()

        src_padding_mask = self.make_padding_mask(src)
        memory = self.encode(src, src_padding_mask)  # encoded once, reused every step

        ys = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            causal_mask = self.make_causal_mask(ys.size(1))
            tgt_embedded = self.pos_encoder(self.tgt_embedding(ys) * self.scale)
            decoded = self.transformer_decoder(
                tgt_embedded, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            next_token = self.fc_out(decoded[:, -1]).argmax(-1)  # last position only

            # Pad sentences that already emitted <EOS> instead of appending junk
            next_token = next_token.masked_fill(finished, self.pad_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == self.eos_idx)
            if finished.all():
                break

        translations = ys[:, 1:]  # drop the leading <SOS>, matching the RNN output
        attentions = torch.zeros(batch_size, translations.size(1), src_len, device=self.device)
        return translations, attentions
