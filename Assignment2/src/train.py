# coding: utf-8
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn as nn

import data
import model
import os
import os.path as osp

'''
ATTENTION!
working file: ./src
'''

# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=256,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--use_pe', action="store_true")
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)



########################################
# Build LMModel model (build your language model here)
# transformer
model_transformer = model.LMModel_transformer(nvoc = len(data_loader.vocabulary), num_layers = args.num_layers,
                      dim = args.emb_dim, nhead = args.num_heads)
model_transformer = model_transformer.to(device)
optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=1e-3)
# RNN
model_RNN = model.LMModel_RNN(nvoc = len(data_loader.vocabulary), dim = args.emb_dim,
                              num_layers = args.num_layers)
model_RNN = model_RNN.to(device)
optimizer_RNN = optim.Adam(model_RNN.parameters(), lr=1e-3)
# LSTM
model_LSTM = model.LMModel_LSTM(nvoc = len(data_loader.vocabulary), dim = args.emb_dim,
                              num_layers = args.num_layers)
model_LSTM = model_LSTM.to(device)
optimizer_LSTM = optim.Adam(model_LSTM.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(model_):
    data_loader.set_valid()
    data_, target, end_flag = data_loader.get_batch()
    model_.eval()
    idx = 0
    avg_loss = 0
    print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data_, target, end_flag = data_loader.get_batch()
            data_ = data_.to(device)
            target = target.to(device)
            decode = model_(data_)[0]

            # Calculate cross-entropy loss
            loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target)
            avg_loss += loss
            idx += 1
    print(f"The average loss is {avg_loss / idx}")
    return math.exp(avg_loss.item() / idx)


# Train Function
def train(model_, optimizer_):
    torch.autograd.set_detect_anomaly(True)
    data_loader.set_train()
    data_, target, end_flag = data_loader.get_batch()
    model_.train()
    idx = 0
    avg_loss = 0
    while not end_flag:
        data_, target, end_flag = data_loader.get_batch()
        data_ = data_.to(device)
        target = target.to(device)
        decode = model_(data_)[0]
        # Calculate cross-entropy loss
        optimizer_.zero_grad()
        loss = criterion(decode.view(decode.size(0)*decode.size(1), -1), target)
        loss.backward()
        optimizer_.step()
        if (idx+1) % 50 == 0:
            print(f"The loss is {loss}")
        idx += 1
        avg_loss += loss
    return math.exp(avg_loss.item() / idx)


# Loop over epochs for transformer
train_perplexity_transformer = []
valid_perplexity_transformer = []
train_perplexity_RNN = []
valid_perplexity_RNN = []
train_perplexity_LSTM = []
valid_perplexity_LSTM = []

def see_epoch(model_, train_ls, valid_ls, name:str = None, optimizer_ = None):
    for epoch in range(1, args.epochs + 1):
        print(f"Start training epoch ({name}) {epoch}")
        train_perplexity_transformer.append(train(model_, optimizer_))
        valid_perplexity_transformer.append(evaluate(model_))
    print(f"Train Perplexity {name} {train_ls}")
    print(f"Valid Perplexity {name} {valid_ls}")


# see_epoch(model_RNN, train_perplexity_RNN, valid_perplexity_RNN, "RNN", optimizer_RNN)
# see_epoch(model_transformer, train_perplexity_transformer, valid_perplexity_transformer, "transformer", optimizer_transformer)
see_epoch(model_LSTM, train_perplexity_LSTM, valid_perplexity_LSTM, "LSTM", optimizer_LSTM)

