import torch
import torch.nn as nn


class FeedForward(nn.Module):
  def __init__(self, n_embd, dropout):
    super().__init__()
    self.n_embd = n_embd
    self.dropout = dropout
    
    self.prog = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.LayerNorm(4*n_embd),
        nn.GELU(),
        nn.Linear(4*n_embd, 8*n_embd),
        nn.LayerNorm(8*n_embd),
        nn.GELU(),
        nn.Linear(8*n_embd, 2*n_embd),
        nn.LayerNorm(2*n_embd),
        nn.ReLU(),
        nn.Linear(2*n_embd, n_embd),
        nn.Dropout(dropout),
    )

  def forward(self, param):
    return self.prog(param)