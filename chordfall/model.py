import torch
import torch.nn as nn


class TinyWaveRNN(nn.Module):
    def __init__(self, vocab_size=256, hidden=128, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)
        y, h = self.rnn(e, h)
        logits = self.head(y)
        return logits, h
