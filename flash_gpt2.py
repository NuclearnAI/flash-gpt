import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import flash_attn.flash_attn_triton as flash_attn_triton

# @dataclass
# class GPT2Config:


# GPT2BaseConfig = GPT2Config(

# )

# GPT2MediumConfig = GPT2Config(

# )

# GPT2LargeConfig = GPT2Config(

# )

# GPT2XLConfig = GPT2Config(

# )



class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    # noinspection PyMethodMayBeStatic
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3.0))
            ))
        )

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        inner_dim = config.num_heads * config.head_dim
        self.c_attn = Conv1D(3 * inner_dim, config.hidden_dim)
        self.c_proj = Conv1D(config.hidden_dim, inner_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self,)
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

def compute_flash_attention(query_states, key_states, value_states):
    """Flash Attention (Triton version)
    :param query_states: [batch_size, q_seq_len, num_heads, head_size]
    :param key_states: [batch_size, kv_seq_len, num_heads, head_size]
    :param value_states: [batch_size, kv_seq_len, num_heads, head_size]
    :return: attn_out: [batch_size, q_seq_len, num_heads, head_size]
    """
    return flash_attn_triton.flash_attention(
        query_states, 
        key_states, 
        value_states
    )