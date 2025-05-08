import torch
import torch.nn as nn

class LMModel_LSTM(nn.Module):
    """
    LSTM-based language model:
    1) Embedding layer
    2) LSTM network
    3) Output linear layer
    """
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Construct the LSTM layers manually
        self.weight_ih_l = nn.ParameterList([nn.Parameter(torch.randn(4 * hidden_size, dim if i == 0 else hidden_size)) for i in range(num_layers)])
        self.weight_hh_l = nn.ParameterList([nn.Parameter(torch.randn(4 * hidden_size, hidden_size)) for i in range(num_layers)])
        self.bias_ih_l = nn.ParameterList([nn.Parameter(torch.zeros(4 * hidden_size)) for i in range(num_layers)])
        self.bias_hh_l = nn.ParameterList([nn.Parameter(torch.zeros(4 * hidden_size)) for i in range(num_layers)])

        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

        for weight_ih in self.weight_ih_l:
            nn.init.xavier_uniform_(weight_ih)
        for weight_hh in self.weight_hh_l:
            nn.init.orthogonal_(weight_hh)
        for bias_ih in self.bias_ih_l:
            nn.init.zeros_(bias_ih)
        for bias_hh in self.bias_hh_l:
            nn.init.zeros_(bias_hh)

    def lstm_cell(self, input, hx, weight_ih, weight_hh, bias_ih, bias_hh):
        """A single LSTM cell."""
        hx, cx = hx
        gates = torch.matmul(input, weight_ih.t()) + bias_ih + torch.matmul(hx, weight_hh.t()) + bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)
        seq_len, batch_size, _ = embeddings.size()

        if hidden is None:
            h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=embeddings.dtype, device=embeddings.device)
            c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=embeddings.dtype, device=embeddings.device)
        else:
            h_t, c_t = hidden

        output_seq = []
        for t in range(seq_len):
            x_t = embeddings[t]  # (batch, dim)
            for layer in range(self.num_layers):
                h_t_prev = h_t[layer]
                c_t_prev = c_t[layer]
                weight_ih = self.weight_ih_l[layer]
                weight_hh = self.weight_hh_l[layer]
                bias_ih = self.bias_ih_l[layer]
                bias_hh = self.bias_hh_l[layer]

                if layer == 0:
                    h_t_layer, c_t_layer = self.lstm_cell(x_t, (h_t_prev, c_t_prev), weight_ih, weight_hh, bias_ih, bias_hh)
                else:
                    h_t_layer, c_t_layer = self.lstm_cell(h_t_prev, (h_t_prev, c_t_prev), weight_ih, weight_hh, bias_ih, bias_hh)

                h_t[layer] = h_t_layer
                c_t[layer] = c_t_layer
                x_t = h_t_layer  # The output of the current layer becomes the input of the next layer

            output_seq.append(h_t[-1])  # Use the output of the last layer

        output = torch.stack(output_seq)  # (seq_len, batch, hidden_size)
        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), (h_t, c_t)