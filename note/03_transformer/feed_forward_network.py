import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.input_layer = nn.Linear(hidden_size, hidden_size*4)
        self.output_layer = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
