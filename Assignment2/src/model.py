import math
import torch
import torch.nn as nn


class LMModel_transformer(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, nhead=8, num_layers = 4):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, dim)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you Transformer model here. You can add additional parameters to the function.

        ########################################

        self.decoder = nn.Linear(dim, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        #print(input.device)
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        L = embeddings.size(0)
        src_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(input.device.type)
        src = embeddings * math.sqrt(self.dim)
        output, hidden = self.transformer(src, embeddings, src_mask)
        ########################################
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

class LMModel_RNN(nn.Module):
    """
    RNN-based language model:
    1) Embedding layer
    2) Vanilla RNN network
    3) Output linear layer
    """
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        # Construct your RNN model here.

        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)

        ########################################
        output, hidden = self.rnn(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden


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

        ########################################
        # Construct your LSTM model here.
        # all the matrices right multiply for usage

        # structure of the network
        self.cellScale = hidden_size
        self.numlayers = num_layers
        # forget gate
        self.wf = nn.Linear(hidden_size, self.cellScale, bias = True)
        self.uf = nn.Linear(dim, self.cellScale)
        # input gate
        self.wi = nn.Linear(dim, self.cellScale, bias = True)
        self.ui = nn.Linear(dim, self.cellScale)
        # new cell content
        self.wc = nn.Linear(dim, self.cellScale, bias = True)
        self.uc = nn.Linear(dim, self.cellScale)
        # output gate
        self.wo = nn.Linear(self.cellScale, hidden_size, bias = True)
        self.uo = nn.Linear(self.cellScale, hidden_size)
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)
        seq_len = embeddings.size(0)
        batch_size = embeddings.size(1)
        # With embeddings, you can get your output here.
        ########################################
        # TODO: use your defined LSTM network
        # initialize cell and hidden
        if hidden is None:
            h_tot = torch.zeros(self.numlayers, batch_size, self.hidden_size, device=input.device)
            c_tot = torch.zeros(self.numlayers, batch_size, self.cellScale, device=input.device)
            hidden = (h_tot, c_tot)
        else:
            h_tot, c_tot = hidden

        # loop
        output = []
        for t in range(seq_len):
            x_t = embeddings[t, :, :]
            for layer in range(self.numlayers):
                if layer != 0:
                    h_prev = h_tot[layer]
                    c_prev = c_tot[layer]
                else:
                    h_prev = torch.zeros_like(h_tot[layer], device=input.device)
                    c_prev = torch.zeros_like(c_tot[layer], device=input.device)

                # forget gate
                f_t = torch.sigmoid(self.wf(h_prev) + self.uf(x_t))
                # input gate
                i_t = torch.sigmoid(self.wi(h_prev) + self.ui(x_t))
                # new cell content
                c_ncont = torch.tanh(self.wc(h_prev) + self.uc(x_t))
                # update cell
                c_tot[layer] = f_t * c_prev + i_t * c_ncont
                # output gate
                o_t = torch.sigmoid(self.wo(h_prev) + self.uo(x_t))
                # update hidden
                h_tot[layer] = o_t * torch.tanh(c_tot[layer])
                x_t = h_tot[layer]  # update x


        ########################################

        # Output has the dimension of
        # sequence_length * batch_size * number of classes
        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden
