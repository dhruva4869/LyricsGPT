import torch
import torch.nn as nn
from self_attention import SelfAttentionHead


class MultiHeadAttention(nn.Module):
  """
  Multihead attention block, needs params num_heads to define how many
  SelfAttentionHead we actually need.
  Concatenating all of their values
  Finallu projecting them linearly back into the n_embd size so that they can be used to propagate further
  """
  def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
    super().__init__()
    self.num_heads = num_heads
    self.head_size = head_size
    self.n_embd = n_embd
    self.block_size = block_size
    
    self.heads = nn.ModuleList([SelfAttentionHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, param):
    output = torch.cat([h(param) for h in self.heads], dim=-1)
    return self.dropout(self.proj(output))