import torch.nn as nn
from multi_attention import MultiHeadAttention
from feed_forward import FeedForward


class Block(nn.Module):
  def __init__(self, n_embd, n_head, block_size, dropout):
    super().__init__()
    head_sz = int(n_embd / n_head)
    self.multi_head_attention_block = MultiHeadAttention(n_head, head_sz, n_embd, block_size, dropout)
    self.feed_foward_transformer_block = FeedForward(n_embd, dropout)
    self.LayerNorm_1 = nn.LayerNorm(n_embd)
    self.LayerNorm_2 = nn.LayerNorm(n_embd)

  def forward(self, param):
    param = param + self.multi_head_attention_block(self.LayerNorm_1(param))
    param = param + self.feed_foward_transformer_block(self.LayerNorm_2(param))
    return param