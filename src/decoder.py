import torch
import torch.nn as nn

from attention import BahdanauAttention


class Decoder(nn.Module):
    """RNN Decoder with Bahdanau attention.

    At each timestep, the decoder:
      1. Embeds the previous target token
      2. Computes attention over encoder outputs using the previous hidden state
      3. Concatenates embedding + context vector as input to the RNN cell
      4. Produces a prediction over the target vocabulary
    """

    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hidden_dim: int,
                 dec_hidden_dim: int,
                 attn_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 rnn_type: str = "gru",
                 pad_idx: int = 0):
        """
        Args:
            output_dim: size of the target vocabulary
            emb_dim: embedding dimension for target tokens
            enc_hidden_dim: hidden size of the encoder (per direction * num_directions,
                             i.e. the dimension of encoder_outputs)
            dec_hidden_dim: hidden size of the decoder RNN
            attn_dim: internal dimension of the attention mechanism
            num_layers: number of RNN layers
            dropout: dropout probability
            rnn_type: "gru" or "lstm"
            pad_idx: index of the <PAD> token
        """
        super().__init__()

        self.output_dim = output_dim
        self.rnn_type = rnn_type.lower()

        # 1: Define the embedding layer for target tokens
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)

        # 2: Instantiate the attention module
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim)

        # 3: Define the RNN cell.
        input_size = emb_dim + enc_hidden_dim
        self.rnn = nn.GRU(emb_dim + enc_hidden_dim, dec_hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        # 4: Define the output projection layer
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim + emb_dim, output_dim)

        # 5: Define dropout
        self.dropout = nn.Dropout(dropout)

    def _get_attention_hidden(self, hidden: torch.Tensor):
        """Return the last-layer hidden state in shape (batch, dec_hidden_dim)."""
        # hidden is shaped (num_layers, batch, dec_hidden_dim)
        return hidden[-1] if hidden.dim() == 3 else hidden

    def _prepare_rnn_hidden(self, hidden: torch.Tensor):
        """Convert a 2D init state to the shape expected by nn.GRU/nn.LSTM.

        The encoder currently returns a projected hidden state shaped
        (batch, dec_hidden_dim). The decoder RNN expects
        (num_layers, batch, dec_hidden_dim), so we expand/repeat here.

        For LSTM, if only one tensor is provided, we use it for both h and c
        to keep the interface usable during early development.
        """
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        return hidden

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Single decoding step (used in a loop during training/inference).

        Args:
            input_token: previous target token id, shape (batch,) or (batch, 1)
            hidden: previous decoder hidden state, shape (batch, dec_hidden_dim)
                    (for LSTM you might need to handle (h, c) as a tuple - decide
                     on a convention and be consistent)
            encoder_outputs: all encoder hidden states, shape (batch, src_len, enc_hidden_dim)
            mask: optional source mask for attention, shape (batch, src_len)

        Returns:
            prediction: logits over target vocabulary, shape (batch, output_dim)
            hidden: updated decoder hidden state
            attn_weights: attention weights for this step, shape (batch, src_len)
                          (store these across timesteps if you want to plot
                           attention heatmaps later)
        """

        # 1: Embed input_token and apply dropout: input_token shape (batch,) -> embedded shape (batch, 1, emb_dim)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))

        # 2: Compute attention context vector using `hidden` and `encoder_outputs`
        attn_hidden = self._get_attention_hidden(hidden)
        context, attn_weights = self.attention(attn_hidden, encoder_outputs, mask) # context shape: (batch, enc_hidden_dim) -> unsqueeze to (batch, 1, enc_hidden_dim)

        # 3: Concatenate embedded input and context vector along the feature dimension
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2) # shape: (batch, 1, emb_dim + enc_hidden_dim)

        # 4: Pass through the RNN
        #   - GRU expects hidden shape (num_layers, batch, dec_hidden_dim)
        hidden = self._prepare_rnn_hidden(hidden)
        output, hidden = self.rnn(rnn_input, hidden)

        # 5: Compute the final prediction.
        #   Concatenate: output (squeezed), context, embedded (squeezed) then pass through self.fc_out
        prediction = self.fc_out(torch.cat((output.squeeze(1), context, embedded.squeeze(1)), dim=1))

        # 6: Return prediction, updated hidden state, and attn_weights
        return prediction, hidden, attn_weights