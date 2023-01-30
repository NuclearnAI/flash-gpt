import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
#import flash_attn.flash_attn_triton as flash_attn_triton

from triton_flash import flash_attn_func

from einops import rearrange, repeat

@dataclass
class GPT2Config:
    dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    dim_head: int = 64
    max_seq_len: int = 1024
    attn_pdrop: float = 0.1
    dropout: float = 0.1
    vocab_size: int = 50257
    layer_norm_epsilon: float = 1e-5

"""
{
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "vocab_size": 50257
}
"""



# GPT2BaseConfig = GPT2Config(
#     hidden_dim = 512,
#     num_heads = 8,
#     num_layers = 12,
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
        self.c_fc = Conv1D(config.dim, config.hidden_dim)
        self.c_proj = Conv1D(config.hidden_dim, config.dim)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.dropout)

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
        inner_dim = config.num_heads * config.dim_head
        self.c_attn = Conv1D(3 * inner_dim, config.hidden_dim)
        self.c_proj = Conv1D(config.hidden_dim, inner_dim)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        h = self.config.num_heads

        print(x.shape)

        qkv = self.c_attn(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), qkv)

        # batch_size, seq_len, num_heads, head_dim for flash attention

        print(q.shape, k.shape, v.shape)

        flash_attn_out = flash_attn_func(q, k, v, None, True, None)

        out = rearrange(flash_attn_out, 'b n h d -> b n (h d)')

        attn_out = self.c_proj(out)

        attn_out = self.resid_dropout(attn_out)

        return attn_out

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        x = self.attn(x) + residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x) + residual
        return x

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)

        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([])
        for _ in range(config.num_layers):
            self.layers.append(GPT2Block(config))
        
        self.ln_f = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    def forward(self, x):

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.drop(x)

        for block in self.layers:
            x = block(x)

        x = self.ln_f(x)

        return x


# Test GPT2Attention
config = GPT2Config

attention = GPT2Attention(config).to(torch.float16).cuda()

print(attention(torch.randn(1, 512, 512).to(torch.float16).cuda()))

# Test GPT2Block

block = GPT2Block(config).to(torch.float16).cuda()

print(block(torch.randn(1, 512, 512).to(torch.float16).cuda()))

# Test GPT2Model

model = GPT2Model(config).to(torch.float16).cuda()

print(model(
    torch.randint(0, 50257, (1, 512)).to(torch.float16).cuda()
))


def stabilize_hidden_states(hidden_states):
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    return hidden_states

