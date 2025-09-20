# Custom GPT
import torch
import torch.nn as nn
# this just makes it easier instead of creating a custom class / same method twice. No need to define a layer before using it, just use it as F.ReLu(embd)
from torch.nn import functional as F
from block import Block

"""
parameters for the GPT
"""
# basic neural network terminology
batch_size = 64 # kitne batch mai training karenge
block_size = 256 # ek transformer block ka kitna size hoga
learning_rate_alpha = 2e-4 # alpha

# checking device for speedup computation
if torch.cuda.is_available():
  device = 'cuda'
elif torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'

# GPT iterations and logging parameters for debugging.
max_iterations = 2000
check_progress = 200

# evaluate the loss function at ? steps
eval_iters = 200
# size of embed "vector"
n_embd = 384
# attention heads to consider noun-adverb, etc
n_head = 4
# number of layers in the model - Actual working blocks in the GPT
n_layer = 3
# dropout to be used after softmax function.
dropout = 0.2

torch.manual_seed(1948)

with open('./datasets/lyrics.txt', 'r', encoding = 'utf-8') as f:
  text = f.read()

# print(text[:100])

chars = sorted(list(set(text)))
# this the number of tokens the model actually understands.

# NAIVE WAY OF TOKENIZATION - IMPROVE ON THIS USING CUSTOM TOKENIZER
vocab_size = len(chars)
enc = {c : i for i, c in enumerate(chars)}
dec = {v : k for k, v in enc.items()}

def encode(s):
  return [enc[c] for c in s]

def decode(d):
  return "".join([dec[c] for c in d])

# train and test data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
test_data = data[n:]


def get_batch(data_type):
  """
  data_type = "test" or "train".
  Generates a batch of input data and output data based off LLM logic output generation
  """
  if data_type.lower() not in ["test", "train"]:
    raise RuntimeError("Data type chosen is neither training data nor testing data")

  data = train_data if data_type.lower() == 'train' else test_data
  random_nums = torch.randint(len(data) - block_size, (batch_size, ))

  input = torch.stack([data[i:i+block_size] for i in random_nums])
  output = torch.stack([data[i+1:i+block_size+1] for i in random_nums])

  input = input.to(device)
  output = output.to(device)

  return input, output
  

class LyricsGPT(nn.Module):
  def __init__(self):
    super().__init__()
    # vocab_size = # of tokens, n_embd = embedding dimension
    # so vocab_size = all the things that we can actually get that is the token
    # n_embd is the dimension / size of the embedding
    # "h" (token) = [0.1, 0.2 .....] (embedding)
    # total tokens like "h" = vocab_size, length of this embedding array = n_embd
    self.token_to_embedding_conversion = nn.Embedding(vocab_size, n_embd)
    """
    t1 - [.....]
    t2 - [.....]
    t3 - [.....]
    t4 - [.....]
    ............
    """
    self.position_to_embedding_conversion = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
    self.final_layer_norm = nn.LayerNorm(n_embd)
    self.linear_head = nn.Linear(n_embd, vocab_size)
    self.dropout = nn.Dropout(dropout)

    self.apply(self._initialize)

  def _initialize(self, module):
      if isinstance(module, nn.Linear):
          fan_in = module.weight.size(1)
          fan_out = module.weight.size(0)
          std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
          torch.nn.init.normal_(module.weight, mean=0.0, std=std)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

  def forward(self, input, targets=None):
    # input will be of 2 dimensional vocab_size * n_embd type
    B, T = input.shape
    token = self.token_to_embedding_conversion(input)
    pos = self.position_to_embedding_conversion(torch.tensor(list(range(T)), device=device))
    param = token + pos
    param = self.blocks(param)
    param = self.final_layer_norm(param)
    pred = self.linear_head(param)

    if targets is None:
      loss = 0
    else:
      B, T, C = pred.shape
      pred = pred.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(pred, targets)

    return pred, loss

  def generate(self, running_input, gpt_tokens=None):
    gpt_tokens = gpt_tokens or 100
    for _ in range(gpt_tokens):
      get_latest_context = running_input[:, -block_size:]
      pred, loss = self.__call__(get_latest_context)
      pred = pred[:, -1, :]
      pred = F.softmax(pred)
      pred = torch.multinomial(pred, num_samples=1) # B * 1
      output = torch.cat((running_input, pred), dim=1) # B * T+1
      running_input = output
    return running_input

model = LyricsGPT()
m = model.to(device)


import math
total_params = sum(math.prod(p.size()) for p in m.parameters())
print(f"{total_params/1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_alpha)


@torch.no_grad()
def estimate_loss(eval_iters=None, splits=None, verbose=False):
  """
  "eval_iters" is the number of iterations to evaluate the loss.
  "splits" is the splits to evaluate the loss on.
  "verbose" is the verbosity of the output.
  returns the loss for the train and test splits.
  """
  if eval_iters is None:
    eval_iters = globals()['eval_iters']
  if splits is None:
    splits = ['train', 'test']
  out = {}
  model.eval()
  try:
    for split in splits:
      if verbose:
        print(f"Evaluating {split} loss...")
      losses = torch.zeros(eval_iters, device=device)
      for k in range(eval_iters):
        try:
          X, Y = get_batch(split)
          _, loss = model(X, Y)
          losses[k] = loss.item()
          if verbose and (k + 1) % (eval_iters // 4) == 0:
            print(f"  {split}: {k + 1}/{eval_iters} batches processed")
        except Exception as e:
          print(f"Warning: Error in {split} batch {k}: {e}")
          losses[k] = float('inf')  # Mark failed batches
          continue
      valid_losses = losses[losses != float('inf')]

      if len(valid_losses) == 0:
        print(f"Error: No valid losses computed for {split}")
        out[split] = {'mean': float('inf'), 'std': 0.0, 'min': float('inf'), 'max': float('inf')}
        continue
      
      out[split] = {
        'mean': valid_losses.mean().item(),
        'std': valid_losses.std().item(),
        'min': valid_losses.min().item(),
        'max': valid_losses.max().item(),
        'valid_samples': len(valid_losses)
      }
      
      if verbose:
        print(f"  {split} loss: {out[split]['mean']:.4f} ± {out[split]['std']:.4f}")
        
  except Exception as e:
    print(f"Error in estimate_loss: {e}")
    return {'train': {'mean': float('inf')}, 'test': {'mean': float('inf')}}
    
  finally:
    model.train()
  
  return out


for iter in range(max_iterations):
    if iter % eval_iters == 0 or iter == max_iterations - 1:
        losses = estimate_loss(verbose=(iter % (eval_iters * 2) == 0))
        train_loss = losses['train']['mean']
        test_loss = losses['test']['mean']
        train_std = losses['train']['std']
        test_std = losses['test']['std']
        
        print(f"step {iter}: train loss {train_loss:.4f} ± {train_std:.4f}, test loss {test_loss:.4f} ± {test_std:.4f}")
        
        if iter > 0 and iter % (eval_iters * 2) == 0:
            if test_loss > prev_val_loss * 1.1:
                print(f"Early stopping at step {iter} due to validation loss increase")
                break
        prev_val_loss = test_loss
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 2000)[0].tolist()))