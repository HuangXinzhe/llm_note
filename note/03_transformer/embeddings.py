import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    """位置编码
    """

    def __init__(self, embed_size, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float()
                    * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: 词表大小
        :param embed_size: embedding维度768
        :param dropout: dropout概率
        """
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=0)
        self.position_embedding = PositionalEmbedding(
            embed_size=embed_size, max_len=512)
        self.token_type_embedding = nn.Embedding(2, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, input_ids, token_type_ids):
        x = self.token_embedding(input_ids) + self.position_embedding(
            input_ids) + self.token_type_embedding(token_type_ids)
        return self.dropout(x)
