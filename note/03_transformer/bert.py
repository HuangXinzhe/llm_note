import torch.nn as nn
from embeddings import BERTEmbedding
from transformer import TransformerBlock


class BERT(nn.Module):

    def __init__(self, vocab_size, hidden_size=768, layer_num=12, head_num=12, dropout=0.1):

        super().__init__()
        # Embedding层
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size, embed_size=hidden_size)
        # N层Transformers
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, head_num, dropout)
             for _ in range(layer_num)]
        )

    def forward(self, input_ids, token_type_ids):
        """
        tokenizer(["你好吗","你好"], text_pair=["我很好","我好"], max_length=10, padding='max_length',truncation=True)
        [CLS]你好吗[SEP]我很好[SEP][PAD]
        [CLS]你好[SEP]我好[SEP][PAD][PAD][PAD]  
        input_ids: [
            [101, 872, 1962, 1408, 102, 2769, 2523, 1962, 102, 0],
            [101, 872, 1962, 102, 2769, 1962, 102, 0, 0, 0]
        ]
        token_type_ids：[
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
            ]
        """
        attention_mask = (x > 0).unsqueeze(
            1).repeat(1, x.size(1), 1).unsqueeze(1)

        # 计算embedding
        x = self.embedding(input_ids, token_type_ids)

        # 逐层代入Tranformers
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, attention_mask)

        return x
