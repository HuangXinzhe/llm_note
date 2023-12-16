import torch.nn as nn
from multi_headed_attention import MultiHeadedAttention
from feed_forward_network import FeedForward

class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, head_num, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadedAttention(head_num, hidden_size)
        self.feed_forward = FeedForward(hidden_size, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x0 = x
        # 多头注意力层
        x = self.multi_head_attention(x, mask)

        # 残差和LayerNorm层(1)
        x = self.dropout1(x)
        x = self.layer_norm1(x0+x)

        # 前向网络层
        x1 = x
        x = self.feed_forward(x)

        # 残差和LayerNorm层(2)
        x = self.dropout2(x)
        x = self.layer_norm2(x1+x)
        return x
