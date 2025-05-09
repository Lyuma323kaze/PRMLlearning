import math
import torch
import torch.nn as nn
import os


# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class LMModel_transformer(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, nhead=8, num_layers = 4):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, dim)
        self.dim = dim
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
        self.rnn = nn.RNN(input_size=dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=False)
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -init_uniform, init_uniform)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

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
        self.nvoc = nvoc
        ########################################
        # Construct your LSTM model here.
        # all the matrices right multiply for usage

        # structure of the network
        self.cellScale = hidden_size
        self.hidden_size = hidden_size
        self.numlayers = num_layers
        # gates
        # forget gate
        self.wf = nn.Parameter(torch.zeros(hidden_size, self.cellScale))
        self.uf = nn.Parameter(torch.zeros(dim, self.cellScale))
        self.bf = nn.Parameter(torch.zeros(self.cellScale))
        # input gate
        self.wi = nn.Parameter(torch.zeros(hidden_size, self.cellScale))
        self.ui = nn.Parameter(torch.zeros(dim, self.cellScale))
        self.bi = nn.Parameter(torch.zeros(self.cellScale))
        # new cell content
        self.wc = nn.Parameter(torch.zeros(hidden_size, self.cellScale))
        self.uc = nn.Parameter(torch.zeros(dim, self.cellScale))
        self.bc = nn.Parameter(torch.zeros(self.cellScale))
        # output gate
        self.wo = nn.Parameter(torch.zeros(hidden_size, self.cellScale))
        self.uo = nn.Parameter(torch.zeros(dim, self.cellScale))
        self.bo = nn.Parameter(torch.zeros(self.cellScale))
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        for param in [self.wf, self.uf, self.wi, self.ui, self.wc, self.uc, self.wo, self.uo]:
            param.data.uniform_(-init_uniform, init_uniform)
        for bias in [self.bf, self.bi, self.bc, self.bo]:
            bias.data.zero_()

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
            h_tot = h_tot.clone().detach()
            c_tot = c_tot.clone().detach()
        # loop
        # output = torch.zeros(seq_len, batch_size, self.hidden_size, device=input.device)
        output_ = []
        for t in range(seq_len):
            x_t = embeddings[t, :, :]
            # cloning
            new_h_tot = h_tot.clone()
            new_c_tot = c_tot.clone()
            for layer in range(self.numlayers):
                # h and c of last step
                h_prev = h_tot[layer]
                c_prev = c_tot[layer]
                # forget gate
                f_t_layer = torch.sigmoid(h_prev.matmul(self.wf) + x_t.matmul(self.uf) + self.bf)
                # input gate
                i_t_layer = torch.sigmoid(h_prev.matmul(self.wi) + x_t.matmul(self.ui) + self.bi)
                # output gate
                o_t_layer = torch.sigmoid(h_prev.matmul(self.wo) + x_t.matmul(self.uo) + self.bo)
                # new cell content
                c_ncont_layer = torch.tanh(h_prev.matmul(self.wc) + x_t.matmul(self.uc) + self.bc)
                # update cell
                new_c_layer = f_t_layer * c_prev + i_t_layer * c_ncont_layer
                # update hidden
                new_h_layer = o_t_layer * torch.tanh(new_c_layer)
                # to next layer
                new_c_tot = torch.cat([new_c_tot[:layer], new_c_layer.unsqueeze(0), new_c_tot[layer + 1:]])
                new_h_tot = torch.cat([new_h_tot[:layer], new_h_layer.unsqueeze(0), new_h_tot[layer + 1:]])
                x_t = new_h_tot[layer]
            h_tot = new_h_tot
            c_tot = new_c_tot
            output_.append(h_tot[-1])


        ########################################
        # Output has the dimension of
        # sequence_length * batch_size * number of classes
        output_ = torch.stack(output_)
        output_ = self.drop(output_)
        decoded = self.decoder(output_.view(-1, output_.size(2)))
        return decoded.view(output_.size(0), output_.size(1), decoded.size(-1)), hidden
