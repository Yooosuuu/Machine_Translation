import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau-style) attention mechanism.

    Given the decoder's previous hidden state and all encoder outputs,
    computes attention weights over the source sequence and returns
    a weighted context vector.
    """

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, attn_dim: int):
        """
        Args:
            enc_hidden_dim: dimensionality of encoder hidden states
                (remember: if encoder is bidirectional, this is 2 * hidden_size)
            dec_hidden_dim: dimensionality of decoder hidden state
            attn_dim: internal dimensionality of the attention layer
        """
        super().__init__()

        # Define the layers needed to compute the attention score:
        #   score(s_{t-1}, h_i) = v^T * tanh(W_dec * s_{t-1} + W_enc * h_i)
        #
        #   - a Linear layer to project encoder outputs to attn_dim
        #   - a Linear layer to project decoder hidden state to attn_dim
        #   - a Linear layer (or vector) to map the tanh output to a scalar score
        
        # Defining layers
        self.enc_proj = nn.Linear(enc_hidden_dim, attn_dim) # Project encoder outputs to attn_dim
        self.dec_proj = nn.Linear(dec_hidden_dim, attn_dim) # Project decoder hidden state to attn_dim
        self.v = nn.Linear(attn_dim, 1, bias=False) # Map the tanh output to a scalar score
        

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            decoder_hidden: previous decoder hidden state, shape (batch, dec_hidden_dim)
            encoder_outputs: all encoder hidden states, shape (batch, src_len, enc_hidden_dim)
            mask: optional mask tensor, shape (batch, src_len)
                  used to ignore <PAD> positions in the source sequence
                  (1 for real tokens, 0 for padding)

        Returns:
            context: weighted sum of encoder_outputs, shape (batch, enc_hidden_dim)
            attn_weights: attention weights, shape (batch, src_len)
                          (useful later for visualization / attention heatmaps)
        """

        # 1: Project decoder_hidden and encoder_outputs into the same space
        
        batch_size, src_len, _ = encoder_outputs.size()
        dec_proj = self.dec_proj(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1) # Project decoder hidden state and repeat across src_len
        enc_proj = self.enc_proj(encoder_outputs) # Project encoder outputs

        # 2: Compute energy = tanh(enc_proj(encoder_outputs) + dec_proj(decoder_hidden_expanded))
        #   shape should be (batch, src_len, attn_dim)
        
        energy = torch.tanh(enc_proj + dec_proj) # Compute energy (shape : (batch, src_len, attn_dim))

        # 3: Compute raw attention scores: self.v(energy) -> (batch, src_len, 1) -> squeeze to (batch, src_len)
        
        scores = self.v(energy).squeeze(-1)
        
        # 4: If mask is provided, set scores at padded positions to -inf (or a very large negative number)
        #   before softmax, so they get ~0 weight.
        #   e.g. scores = scores.masked_fill(mask == 0, -1e10)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)        

        # 5: Apply softmax over the src_len dimension to get attn_weights
        
        attn_weights = F.softmax(scores, dim=1)
        
        # 6: Compute context vector as weighted sum:
        #   context = sum_i attn_weights[i] * encoder_outputs[i]
        
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
