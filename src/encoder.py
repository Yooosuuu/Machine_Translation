import torch
import torch.nn as nn


class Encoder(nn.Module):
    """RNN Encoder for the Seq2Seq model.

    Embeds the source sentence and processes it with a (bidirectional)
    LSTM or GRU to produce:
      - outputs: hidden states for every timestep (used by attention)
      - hidden:  final hidden state (used to initialize the decoder)
    """

    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hidden_dim: int,
                 dec_hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 pad_idx: int = 0):
        """
        Args:
            input_dim: size of the source vocabulary
            emb_dim: embedding dimension
            enc_hidden_dim: hidden size of the encoder RNN (per direction)
            dec_hidden_dim: hidden size of the decoder RNN
                (needed because we may project the encoder's final hidden
                 state to initialize the decoder)
            num_layers: number of RNN layers
            dropout: dropout probability (applied to embeddings and between RNN layers)
            rnn_type: "gru" or "lstm"
            bidirectional: whether to use a bidirectional RNN
            pad_idx: index of the <PAD> token (used for embedding padding_idx)
        """
        super().__init__()

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.enc_hidden_dim = enc_hidden_dim

        # 1: Define the embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        # 2: Define the RNN (GRU or LSTM depending on rnn_type)
        #   - input_size = emb_dim
        #   - hidden_size = enc_hidden_dim
        #   - num_layers = num_layers
        #   - batch_first = True
        #   - bidirectional = bidirectional
        #   - dropout = dropout if num_layers > 1 else 0
        
        dropout=dropout if num_layers>1 else 0
                        
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=enc_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # 3: Define a Linear layer to project the encoder's final hidden state
        #   to the decoder's hidden dimension. This is needed because:
        #   - if bidirectional, you need to combine forward + backward final states
        #   - enc_hidden_dim might differ from dec_hidden_dim
        
        fc_input_dim = enc_hidden_dim * 2 if bidirectional else enc_hidden_dim
        self.fc = nn.Linear(fc_input_dim, dec_hidden_dim)
        
        # 4: Define dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        """
        Args:
            src: source token ids, shape (batch, src_len)
            src_lens: actual lengths of each sequence in the batch (before padding),
                      shape (batch,) - used for pack_padded_sequence

        Returns:
            outputs: encoder hidden states for each timestep,
                     shape (batch, src_len, enc_hidden_dim * num_directions)
                     -> this is what the attention mechanism will attend over
            hidden: initial hidden state for the decoder,
                    shape (batch, dec_hidden_dim)
                    -> obtained by projecting/combining the encoder's final states
        """

        # 1: Embed the input and apply dropout
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)

        # 2: Pack the padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        # 3: Pass through the RNN
        packed_outputs, hidden = self.rnn(packed)
        
        # 4: Build the initial decoder hidden state from the encoder's final hidden state(s).
        #   - If bidirectional, concatenate the final forward and backward hidden states
        #     of the last layer: hidden[-2,:,:] and hidden[-1,:,:]
        #   - Apply self.fc and a non-linearity to project to dec_hidden_dim
        
        # GRU returns h_n tensor of shape (num_layers * num_directions, batch, enc_hidden_dim)
        h_n = hidden
        if self.bidirectional:
            fwd = h_n[-2]
            bwd = h_n[-1]
            final_state = torch.cat((fwd, bwd), dim=1)
        else:
            final_state = h_n[-1]
            
        final_state = self.fc(final_state)
        final_state = torch.tanh(final_state)

        # 5: Unpack the sequence back to a padded tensor
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # 6: Return outputs (for attention) and the projected hidden state (for decoder initialization)
        return outputs, final_state
