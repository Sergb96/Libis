import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Generator
import math


def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self,num_heads ,d_model, dropout = 0, mask = False, max_len = 301) -> None:
        super().__init__()
        self.causal = mask
        self.num_heads = num_heads
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.kqv = nn.Linear(d_model, 3 * d_model, bias=False)
        if self.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(1, 1, max_len, max_len))
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)
        
        relative_pos = get_relative_positions(seq_len).to(value.device)
        bias = (self.m * relative_pos).unsqueeze(0)
        # bias.shape == (1, num_heads, seq_len, seq_len)

        score = torch.matmul(query, key) / self.scale + bias
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.dropout(out)

        return out
    
# alibi, jaketae, repo: https://github.com/jaketae/alibi

class GLULayer(nn.Module):
    def __init__(self, dim):
        super(GLULayer, self).__init__()
        self.dim = dim
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out,gate = torch.chunk(x, 2, dim = self.dim)
        return out * self.sig(gate)
    

class SwiGLULayer(nn.Module):
    def __init__(self, dim):
        super(SwiGLULayer, self).__init__()
        self.dim = dim
        self.swish = nn.SiLU() # same as swish

    def forward(self, x):
        out, gate = torch.chunk(x, 2, dim = self.dim)
        return out * self.swish(gate)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, embedding_dim, mult=4, rate = 0.0, use_bias = True):
        super(FeedForwardSwiGLU, self).__init__()
        swiglu_out = int(embedding_dim * mult/2)
        self.layernorm = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.linear1 = nn.Linear(embedding_dim,embedding_dim * mult, bias = use_bias)
        self.swiglulayer = SwiGLULayer(dim = 1)
        self.drop = nn.Dropout(rate)
        self.linear2 = nn.Linear(swiglu_out,embedding_dim, bias = use_bias)

    def forward(self, inputs):
        x = self.layernorm(inputs.transpose(1,2)) # Swap dimensions and make channel dim=2
        x = self.linear1(x) 
        x = self.swiglulayer(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        x = self.drop(x)
        x = self.linear2(x.transpose(1,2)) # Swap dimensions and make channel dim=2
        out = self.drop(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        return out


class ConformerSASwiGLULayer(nn.Module):
    def __init__(self, embedding_dim,  ff_mult = 4, kernel_size = 15, rate = 0.2, num_heads = 4, use_bias = False):
        super(ConformerSASwiGLULayer, self).__init__()
        self.ff1 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)
        self.layernorm1 = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, groups=embedding_dim, padding='same', bias = False),
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, bias = True),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim,eps = 1e-6)    
        self.attn = ALiBiMultiHeadAttention(d_model=embedding_dim, num_heads=num_heads) #Changed
        self.ff2 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)

    def forward(self, x):
        x = x.float()
        x = x + 0.5 * self.ff1(x)
        
        x1 = x.transpose(1,2)
        x1 = self.layernorm1(x1) #channel dim = 2
        x1 = x1.transpose(1, 2)
        x1 = x1 + self.conv(x1)
        
        x = x + x1
        x = x.transpose(1, 2) # output channel dim = 2
        x = self.layernorm2(x)
        x = x + self.attn(x)
        x = x.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        
        return x
    
class SequenceMaskLayer(torch.nn.Module):
    def __init__(self, n_positions, N, M, ratio=0.2):
        super(SequenceMaskLayer, self).__init__()
        self.ratio = ratio
        self.n_positions = n_positions # max length of sequence
        self.N = N # padding token
        self.M = M # mask token

    def forward(self, x):
        if self.ratio > 0:
            m = torch.rand(x.shape) < self.ratio # random mask
            m = m.to(torch.uint8) # convert to uint8
            is_valid = (x != self.N).to(torch.uint8) # avoid masking padding tokens
            m = m * is_valid # avoid masking padding tokens
            x0 = torch.ones(x.shape).to(torch.uint8) * self.M

            x = m * x0 + (1 - m) * x # mask input sequence
            m = m.to(torch.float32) # convert back to float32
        else:
            m = torch.zeros(x.shape) # no mask

        return x, m
