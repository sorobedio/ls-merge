import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

#############################
# GELU Activation
#############################
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

#############################
# Weight Embedding & Decoding
#############################
class WeightEmbed(nn.Module):
    """
    Embeds a flattened weight vector by dividing it into fixed-size chunks (tokens)
    and projecting each chunk into an embedding dimension. If the vector length
    is not divisible by the chunk size, the input is padded with zeros.
    """
    def __init__(self, chunk_size: int, embed_dim: int, conv: bool = True, flatten: bool = True):
        super().__init__()
        self.conv = conv
        self.flatten = flatten
        self.chunk_size = chunk_size

        if conv:
            self.proj = nn.Conv1d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=chunk_size,
                stride=chunk_size
            )
        else:
            self.proj = nn.Linear(chunk_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        rem = L % self.chunk_size
        if rem != 0:
            pad = self.chunk_size - rem
            x = F.pad(x, (0, pad), "constant", 0)
        if self.conv:
            x = x.unsqueeze(1)
            x = self.proj(x)
            x = x.transpose(1,2)
        else:
            num_tokens = x.shape[1] // self.chunk_size
            x = x.view(B, num_tokens, self.chunk_size)
            x = self.proj(x)
        return x  # (B, num_tokens, embed_dim)

class WeightDecode(nn.Module):
    """
    Decodes embedded weight tokens back to the original weight vector.
    """
    def __init__(self, chunk_size: int, embed_dim: int, out_channels: int = 1, conv: bool = True):
        super().__init__()
        self.conv = conv
        self.chunk_size = chunk_size
        self.out_channels = out_channels

        if conv:
            self.proj = nn.ConvTranspose1d(
                in_channels=embed_dim,
                out_channels=out_channels,
                kernel_size=chunk_size,
                stride=chunk_size
            )
        else:
            self.proj = nn.Linear(embed_dim, chunk_size)

    def forward(self, x: torch.Tensor, original_length: int = None) -> torch.Tensor:
        if self.conv:
            x = x.transpose(1,2)
            x = self.proj(x)
            if self.out_channels == 1:
                x = x.squeeze(1)
        else:
            x = self.proj(x)
            x = x.view(x.size(0), -1)
        if original_length is not None and x.shape[1] > original_length:
            x = x[:, :original_length]
        return x

#############################
# FeedForward
#############################
class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dtype: torch.dtype):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.activation = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return self.fc3(self.activation(x1) * x2)

#############################
# RoPE Utilities
#############################
def precompute_rope_params(
        head_dim: int,
        theta_base: int = 10_000,
        context_length: int = 4096,
        original_context_length: int = None,
        low_freq_factor: float = None,
        high_freq_factor: float = None,
        factor: float = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)

def compute_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, T, D = x.shape
    x1, x2 = x[..., :D//2], x[..., D//2:]
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0)
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rot * sin)

class SharedBuffers:
    _buffers = {}
    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, dtype)
        if key not in SharedBuffers._buffers:
            mask = torch.zeros(context_length, context_length)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length)
            SharedBuffers._buffers[key] = (mask, cos.to(dtype), sin.to(dtype))
        return SharedBuffers._buffers[key]

#########################################
# MultiHeadAttention (Self)
#########################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, rope_base=10_000, dtype=torch.float32):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        _, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, dtype)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        w = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(w, v)
        ctx = ctx.transpose(1,2).reshape(B, T, -1)
        return self.out_proj(ctx)

#########################################
# Transformer Block (Self)
#########################################
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, n_heads, rope_base=10_000, dtype=torch.float32):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, emb_dim, context_length, n_heads, rope_base, dtype)
        self.ff = FeedForward(emb_dim, emb_dim*4, dtype)
        self.norm1 = nn.RMSNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(emb_dim, eps=1e-6)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.attn(x) + res
        res = x
        x = self.norm2(x)
        return self.ff(x) + res

#########################################
# Cross-Attention Modules
#########################################
class CrossMultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_kv, d_out, num_heads, context_length_q, context_length_kv, rope_base=10_000, dtype=torch.float32):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_q, d_out, bias=False)
        self.W_k = nn.Linear(d_kv, d_out, bias=False)
        self.W_v = nn.Linear(d_kv, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        _, cos_q, sin_q = SharedBuffers.get_buffers(context_length_q, self.head_dim, rope_base, dtype)
        _, cos_k, sin_k = SharedBuffers.get_buffers(context_length_kv, self.head_dim, rope_base, dtype)
        self.register_buffer("cos_q", cos_q)
        self.register_buffer("sin_q", sin_q)
        self.register_buffer("cos_k", cos_k)
        self.register_buffer("sin_k", sin_k)

    def forward(self, q_in, kv_in):
        B, Tq, _ = q_in.shape
        B, Tk, _ = kv_in.shape
        Q = self.W_q(q_in).view(B, Tq, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(kv_in).view(B, Tk, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_v(kv_in).view(B, Tk, self.num_heads, self.head_dim).transpose(1,2)
        Q = compute_rope(Q, self.cos_q, self.sin_q)
        K = compute_rope(K, self.cos_k, self.sin_k)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        w = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(w, V)
        ctx = ctx.transpose(1,2).reshape(B, Tq, -1)
        return self.out_proj(ctx)

class CrossTransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length_q, context_length_kv, n_heads, rope_base=10_000, dtype=torch.float32):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, emb_dim, context_length_q, n_heads, rope_base, dtype)
        self.cross_attn = CrossMultiHeadAttention(emb_dim, emb_dim, emb_dim, n_heads, context_length_q, context_length_kv, rope_base, dtype)
        self.ff = FeedForward(emb_dim, emb_dim*4, dtype)
        self.norm1 = nn.RMSNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(emb_dim, eps=1e-6)
        self.norm3 = nn.RMSNorm(emb_dim, eps=1e-6)

    def forward(self, x, enc_out):
        r = x
        x = self.norm1(x)
        x = self.self_attn(x) + r
        r = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out) + r
        r = x
        x = self.norm3(x)
        return self.ff(x) + r

#########################################
# Encoder and Decoder
#########################################
class Encoder(nn.Module):
    def __init__(self, length: int, n_layers: int, chunk_size: int, embed_dim: int,
                 n_heads: int, conv: bool=False, flatten: bool=True,
                 rope_base: int=10_000, dtype=torch.float32):
        super().__init__()
        self.length = length
        self.n_layers = n_layers
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.dtype = dtype
        self.conv = conv
        self.flatten = flatten

        num_tokens = math.ceil(length / chunk_size)
        self.embed = WeightEmbed(chunk_size, embed_dim, conv, flatten)
        self.layers = nn.Sequential(*[
            TransformerBlock(embed_dim, num_tokens, n_heads, rope_base, dtype)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.length = length
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.layers(x)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, length: int, n_layers: int, chunk_size: int, embed_dim: int,
                 n_heads: int, conv: bool=False, flatten: bool=True,
                 rope_base: int=10_000, dtype=torch.float32):
        super().__init__()
        self.length = length
        self.n_layers = n_layers
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.rope_base = rope_base
        self.n_heads = n_heads
        self.conv = conv
        self.flatten = flatten



        num_tokens = math.ceil(length / chunk_size)
        self.init_tokens = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.layers = nn.ModuleList([
            CrossTransformerBlock(embed_dim, num_tokens, num_tokens, n_heads, rope_base, dtype)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decode = WeightDecode(chunk_size, embed_dim, conv=conv)
        self.length = length
        self.chunk_size = chunk_size

    def forward(self, enc_out: torch.Tensor, original_length: int=None) -> torch.Tensor:
        B = enc_out.size(0)
        x = self.init_tokens.expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, enc_out)
        x = self.norm(x)
        if original_length is None:
            original_length = self.length
        return self.decode(x, original_length)