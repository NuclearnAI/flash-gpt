import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
#import flash_attn.flash_attn_triton as flash_attn_triton

from triton_flash import flash_attn_func

from einops import rearrange, repeat

@dataclass
class GPT2Config:
    num_heads = 8
    head_dim = 64
    hidden_dim = 512
    attn_pdrop = 0.1
    resid_pdrop = 0.1


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

        print(self.weight, self.bias)

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)

        x = x.view(size_out)

        return x

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        inner_dim = config.num_heads * config.head_dim
        self.c_attn = Conv1D(3 * inner_dim, config.hidden_dim)
        self.c_proj = Conv1D(config.hidden_dim, inner_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        num_heads, head_dim = self.config.num_heads, self.config.head_dim

        print(x.shape)


        qkv = self.c_attn(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = num_heads), qkv)

        # batch_size, seq_len, num_heads, head_dim

        print(q.shape, k.shape, v.shape)

        flash_attn_out = flash_attn_func(
            q, 
            k, 
            v,
            None,
            True,
            None
        )

        out = rearrange(flash_attn_out, 'b n h d -> b n (h d)')

        attn_out = self.c_proj(out)

        attn_out = self.resid_dropout(attn_out)

        return attn_out

# Test GPT2Attention
config = GPT2Config()

attention = GPT2Attention(config).to(torch.float16).cuda()

print(attention(torch.randn(1, 512, 512).to(torch.float16).cuda()))

def stabilize_hidden_states(hidden_states):
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    return hidden_states


