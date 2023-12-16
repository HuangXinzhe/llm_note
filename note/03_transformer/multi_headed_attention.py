import torch.nn as nn
from attention import Attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, hidden_size, dropout=0.1):
        """
        :param head_num: 头的个数，必须能被hidden_size整除
        :param hidden_size: 隐层的维度，与embed_size一致
        """
        super().__init__()
        assert hidden_size % head_num == 0

        self.per_head_dim = hidden_size // head_num
        self.head_num = head_num
        self.hidden_size = hidden_size

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)

        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def reshape(self, x, batch_size):
        # 拆成多个头
        return x.view(batch_size, -1, self.head_num, self.per_head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        query = self.reshape(self.query_linear(x))
        key = self.reshape(self.key_linear(x))
        value = self.reshape(self.value_linear(x))

        # 每个头计算attention
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 把每个头的attention*value拼接在一起
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_size)

        # 乘一个线性矩阵
        return self.output_linear(x)
