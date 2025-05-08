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
model_LSTM = model.LMModel_LSTM(nvoc = args.emb_dim, dim = args.emb_dim,
                              num_layers = args.num_layers)
model_LSTM = model_LSTM.to(device)
optimizer_LSTM = optim.Adam(model_LSTM.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(model_):
    data_loader.set_valid()
    data, target, end_flag = data_loader.get_batch()
    model_.eval()
    idx = 0
    avg_loss = 0
    print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            decode = model_(data)

            # Calculate cross-entropy loss
            loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target)
            avg_loss += loss
            idx += 1
    print(f"The average loss is {avg_loss / idx}")
    return math.exp(avg_loss.item() / idx)


# Train Function
def train(model_):
    data_loader.set_train()
    data, target, end_flag = data_loader.get_batch()
    model_.train()
    idx = 0
    avg_loss = 0
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        decode = model_(data)

        # Calculate cross-entropy loss
        optimizer_transformer.zero_grad()
        loss = criterion(decode.view(decode.size(0)*decode.size(1), -1), target)
        loss.backward()
        optimizer_transformer.step()
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
for epoch in range(1, args.epochs+1):
    print(f"Start training epoch (transformer) {epoch}")
    train_perplexity_transformer.append(train(model_transformer))
    valid_perplexity_transformer.append(evaluate(model_transformer))

for epoch in range(1, args.epochs+1):
    print(f"Start training epoch (RNN) {epoch}")
    train_perplexity_RNN.append(train(model_RNN))
    valid_perplexity_RNN.append(evaluate(model_RNN))

for epoch in range(1, args.epochs+1):
    print(f"Start training epoch (LSTM) {epoch}")
    train_perplexity_LSTM.append(train(model_LSTM))
    valid_perplexity_LSTM.append(evaluate(model_LSTM))

print(f"Train Perplexity transformer {train_perplexity_transformer}")
print(f"Valid Perplexity transformer {valid_perplexity_transformer}")
print(f'Train Perplexity RNN {train_perplexity_RNN}')
print(f'Valid Perplexity RNN {valid_perplexity_RNN}')
print(f'Train Perplexity LSTM {train_perplexity_LSTM}')
print(f'Valid Perplexity LSTM {valid_perplexity_LSTM}')

