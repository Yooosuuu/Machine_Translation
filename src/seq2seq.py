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
        # TODO: create a mask where positions of <PAD> tokens in src are 0, and others are 1
        raise NotImplementedError

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

        # TODO 1: Run the encoder to get encoder_outputs and the initial decoder hidden state

        # TODO 2: Create the source mask for attention

        # TODO 3: Prepare a tensor to store decoder outputs at each timestep

        # TODO 4: First input to the decoder is the <SOS> token (tgt[:, 0])

        # TODO 5: Loop over target timesteps (from 1 to tgt_len - 1):

        # TODO 6: Return outputs

        raise NotImplementedError

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

        # TODO 1: Run the encoder

        # TODO 2: Create source mask

        # TODO 3: Start with <SOS> token for every sequence in the batch

        # TODO 4: Loop up to max_len steps:
        #   - call decoder to get prediction, hidden, attn_weights
        #   - take argmax to get the next token
        #   - store predicted tokens and attention weights

        # TODO 5: Return the generated token ids and attention weights

        raise NotImplementedError

    # TODO (optional, later): implement beam_search() here for improved decoding quality
