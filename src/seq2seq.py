import random

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Seq2Seq(nn.Module):
    """Full Encoder-Decoder model with attention, including the training loop logic
    for teacher forcing and the inference logic for greedy decoding.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device, pad_idx: int = 0, sos_idx: int = 1, eos_idx: int = 2):
        """
        Args:
            encoder: an Encoder instance
            decoder: a Decoder instance
            device: torch device (cpu / cuda)
            pad_idx, sos_idx, eos_idx: special token ids needed for masking
                                       and for starting/stopping decoding
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create a mask for the source sequence to ignore <PAD> tokens in attention.

        Args:
            src: source token ids, shape (batch, src_len)

        Returns:
            mask: shape (batch, src_len), 1 for real tokens, 0 for <PAD>
        """
        mask = (src != self.pad_idx).int()
        return mask

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        Training forward pass with teacher forcing.

        Args:
            src: source token ids, shape (batch, src_len)
            src_lens: source sequence lengths, shape (batch,)
            tgt: target token ids (including <SOS> and <EOS>), shape (batch, tgt_len)
            teacher_forcing_ratio: probability of using ground-truth token as
                                    next decoder input instead of model's own prediction

        Returns:
            outputs: predicted logits for each target position,
                     shape (batch, tgt_len, output_dim)
                     (note: outputs[:, 0, :] for the <SOS> position is typically
                      left as zeros, since we don't predict <SOS>)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        output_dim = self.decoder.output_dim
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)
        outputs = torch.zeros(batch_size, tgt_len, output_dim).to(self.device)
        input_token = tgt[:, 0]
        for t in range(1, tgt_len):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[:, t, :] = prediction
            top1 = prediction.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else top1
        return outputs

    @torch.no_grad()
    def translate(self, src: torch.Tensor, src_lens: torch.Tensor, max_len: int = 50):
        """
        Greedy decoding for inference (no teacher forcing, no ground truth available).

        Args:
            src: source token ids, shape (batch, src_len)
            src_lens: source sequence lengths, shape (batch,)
            max_len: maximum number of tokens to generate

        Returns:
            translations: generated token ids, shape (batch, <= max_len)
            attentions: attention weights collected at each step,
                        shape (batch, max_len, src_len)
                        -> useful for plotting attention heatmaps
        """
        self.eval()
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)
        input_token = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
        translations = []
        attentions = []
        for _ in range(max_len):
            prediction, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs, mask)
            input_token = prediction.argmax(1)
            translations.append(input_token)
            attentions.append(attn_weights)
        translations = torch.stack(translations, dim=1)
        attentions = torch.stack(attentions, dim=1)
        return translations, attentions

    # TODO (optional, later): implement beam_search() here for improved decoding quality
