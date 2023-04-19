import torch
from torch import nn
from torch.nn import functional as F
import copy
from typing import Union, Sequence

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s

def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s

def positional_embedding(input, d_model) -> torch.Tensor:
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32)
    out = positional_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

def clones(module, n):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def generate_padding_mask(sequences: TensorOrNone, padding_idx: int) -> torch.BoolTensor:
    '''
        sequences: (bs, seq_len, dim)
    '''
    if sequences is None:
        return None

    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences

    mask = (torch.sum(__seq, dim=-1) == (padding_idx*__seq.shape[-1])).long() * -10e4 # (b_s, seq_len)
    return mask.unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)

def generate_sequential_mask(seq_len: int) -> torch.BoolTensor:
    '''
        Mask out subsequent positions
    '''
    attn_shape = (seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) * -10e4 # (seq_len, seq_len)

    return subsequent_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

def generate_self_attention_masks(padding_masks: torch.Tensor, sequential_masks: torch.Tensor):
    padding_masks = padding_masks != 0
    sequential_masks = sequential_masks != 0
    self_attention_masks = torch.logical_or(padding_masks, sequential_masks).long() * -10e4
    
    return self_attention_masks

