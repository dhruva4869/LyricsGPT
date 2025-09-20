import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
  """
  This is the self attention head
  We want to be able to concat a bunch of these in the same dimension (unlike stack) to produce MultiHeadAttention
  """
  def __init__(self, head_size, n_embd, block_size, dropout):
    # technically head_size = n_embd / n_heads
    super().__init__()
    self.head_size = head_size
    self.n_embd = n_embd
    self.block_size = block_size

    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)

    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, param):
    # param = batch_size * tokens * embedding length [B * T * C]
    # C = count of how many numbers represent each token. ["Hello", "World"] -> [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    B, T, C = param.shape
    key = self.key(param)
    query = self.query(param)
    value = self.value(param)

    # now time for the maths formula
    tmp = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
    tmp = tmp.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    tmp = F.softmax(tmp, dim=-1)
    tmp = self.dropout(tmp)

    output = tmp @ value
    return output