import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Positional embeddings
# --------------------------------------------------------------------------- #

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sinusoidal positional embeddings for the given positions."""
    assert embed_dim % 2 == 0, "embed_dim must be even"
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_hw: Tuple[int, int]) -> np.ndarray:
    """2D sinusoidal positional embeddings over an (H, W) grid."""
    H, W = grid_hw
    grid = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))  # w first
    grid = np.stack(grid, axis=0).reshape([2, 1, H, W])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def build_sincos_position_embedding(
    seq_len: int,
    embed_dim: int,
    grid_hw: Optional[Tuple[int, int]] = None,
    cls_token: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a (1, seq_len[+1], embed_dim) positional embedding tensor."""
    if grid_hw is None:
        pos_np = get_1d_sincos_pos_embed_from_grid(
            embed_dim, np.arange(seq_len, dtype=np.float32))
    else:
        assert grid_hw[0] * grid_hw[1] == seq_len, "H*W must equal seq_len"
        pos_np = get_2d_sincos_pos_embed(embed_dim, grid_hw)

    if cls_token:
        pos_np = np.concatenate(
            [np.zeros((1, embed_dim), dtype=np.float32), pos_np], axis=0)

    return torch.from_numpy(pos_np).float().unsqueeze(0).to(device)


# --------------------------------------------------------------------------- #
# Weight tokenization
# --------------------------------------------------------------------------- #

class WeightEmbed(nn.Module):
    """Split a flattened weight vector into fixed-size chunks and embed each."""

    def __init__(self, chunk_size: int, embed_dim: int, conv: bool = True, flatten: bool = True):
        super().__init__()
        self.conv = conv
        self.flatten = flatten
        self.chunk_size = chunk_size

        if conv:
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=chunk_size, stride=chunk_size)
        else:
            self.proj = nn.Linear(chunk_size, embed_dim)
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L) -> (B, num_tokens, embed_dim); input is zero-padded if needed."""
        B, L = x.shape
        remainder = L % self.chunk_size
        if remainder != 0:
            x = F.pad(x, (0, self.chunk_size - remainder), value=0)

        if self.conv:
            x = self.proj(x.unsqueeze(1))      # (B, embed_dim, num_tokens)
            x = x.transpose(1, 2)              # (B, num_tokens, embed_dim)
        else:
            num_tokens = x.shape[1] // self.chunk_size
            x = self.proj(x.view(B, num_tokens, self.chunk_size))
        return x


class WeightDecode(nn.Module):
    """Inverse of WeightEmbed: tokens back to a flat weight vector."""

    def __init__(self, chunk_size: int, embed_dim: int, out_channels: int = 1, conv: bool = True):
        super().__init__()
        self.conv = conv
        self.chunk_size = chunk_size
        self.out_channels = out_channels

        if conv:
            self.proj = nn.ConvTranspose1d(
                embed_dim, out_channels, kernel_size=chunk_size, stride=chunk_size)
        else:
            self.proj = nn.Linear(embed_dim, chunk_size)

    def forward(self, x: torch.Tensor, original_length: Optional[int] = None) -> torch.Tensor:
        """(B, num_tokens, embed_dim) -> (B, L), trimmed to original_length."""
        if self.conv:
            x = self.proj(x.transpose(1, 2))   # (B, out_channels, L)
            if self.out_channels == 1:
                x = x.squeeze(1)
        else:
            x = self.proj(x).view(x.shape[0], -1)

        if original_length is not None and x.shape[1] > original_length:
            x = x[:, :original_length]
        return x


# --------------------------------------------------------------------------- #
# Transformer building blocks
# --------------------------------------------------------------------------- #

class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def precompute_rope_params(
    head_dim: int,
    theta_base: int = 10_000,
    context_length: int = 4096,
    original_context_length: Optional[int] = None,
    low_freq_factor: Optional[float] = None,
    high_freq_factor: Optional[float] = None,
    factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin, with optional Llama-style frequency scaling."""
    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    if all(v is not None for v in (original_context_length, low_freq_factor,
                                   high_freq_factor, factor)):
        low_freq_wavelen = original_context_length / low_freq_factor
        high_freq_wavelen = original_context_length / high_freq_factor
        wavelen = 2 * torch.pi / inv_freq

        inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth = (original_context_length / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
        is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq = torch.where(is_medium, smoothed, inv_freq_scaled)

    angles = torch.arange(context_length)[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def compute_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, num_heads, seq_len, head_dim)."""
    seq_len, head_dim = x.shape[2], x.shape[3]
    assert head_dim % 2 == 0, "head_dim must be even"

    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


class SharedBuffers:
    """Cache of precomputed RoPE buffers keyed by configuration."""

    _buffers: dict = {}

    @staticmethod
    def get_buffers(
        context_length: int,
        head_dim: int,
        rope_base: int,
        original_context_length: Optional[int] = None,
        low_freq_factor: Optional[float] = None,
        high_freq_factor: Optional[float] = None,
        factor: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (context_length, head_dim, rope_base, original_context_length,
               low_freq_factor, high_freq_factor, factor, dtype)

        if key not in SharedBuffers._buffers:
            cos, sin = precompute_rope_params(
                head_dim, rope_base, context_length,
                original_context_length, low_freq_factor, high_freq_factor, factor)
            if dtype is not None:
                cos, sin = cos.to(dtype), sin.to(dtype)
            SharedBuffers._buffers[key] = (cos, sin)

        return SharedBuffers._buffers[key]


class MultiHeadAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        rope_base: int = 10_000,
        original_context_length: Optional[int] = None,
        low_freq_factor: Optional[float] = None,
        high_freq_factor: Optional[float] = None,
        factor: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

        cos, sin = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base,
            original_context_length, low_freq_factor, high_freq_factor, factor, dtype)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_tokens, d_in) -> (B, num_tokens, d_out)."""
        b, num_tokens, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = compute_rope(split_heads(self.W_q(x)), self.cos, self.sin)
        k = compute_rope(split_heads(self.W_k(x)), self.cos, self.sin)
        v = split_heads(self.W_v(x))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + SwiGLU feed-forward."""

    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        n_heads: int,
        rope_base: int,
        original_context_length: Optional[int] = None,
        low_freq_factor: Optional[float] = None,
        high_freq_factor: Optional[float] = None,
        factor: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=emb_dim, d_out=emb_dim, context_length=context_length,
            num_heads=n_heads, rope_base=rope_base,
            original_context_length=original_context_length,
            low_freq_factor=low_freq_factor, high_freq_factor=high_freq_factor,
            factor=factor, dtype=dtype)
        self.ff = FeedForward(emb_dim, emb_dim * 4)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Encoder / Decoder
# --------------------------------------------------------------------------- #

class Encoder(nn.Module):
    """Embed a flat weight vector into transformer token representations.

    latent_dim and flatten are reserved for API compatibility and unused here.
    """

    def __init__(
        self,
        length: int,
        n_layers: int,
        chunk_size: int,
        embed_dim: int,
        n_heads: int,
        latent_dim: int,
        conv: bool = False,
        flatten: bool = True,
        grid_hw: Optional[Tuple[int, int]] = None,
        rope_base: int = 10_000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.length = length
        self.chunk_size = chunk_size
        self.num_tokens = math.ceil(length / chunk_size)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.rope_base = rope_base


        self.embed = WeightEmbed(chunk_size, embed_dim, conv=conv, flatten=flatten)
        self.encoder_transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, self.num_tokens, n_heads, rope_base, dtype=dtype)
            for _ in range(n_layers)
        ])
        self.register_buffer(
            "pos_embed",
            build_sincos_position_embedding(self.num_tokens, embed_dim, grid_hw=grid_hw),
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, length) -> (B, num_tokens, embed_dim)."""
        x = self.embed(x)
        x = x + self.pos_embed[:, :x.size(1)]
        return self.encoder_transformer(x)


class Decoder(nn.Module):
    """Reconstruct a flat weight vector from token representations.

    latent_dim and flatten are reserved for API compatibility and unused here.
    """

    def __init__(
        self,
        length: int,
        n_layers: int,
        chunk_size: int,
        embed_dim: int,
        n_heads: int,
        latent_dim: int,
        conv: bool = False,
        flatten: bool = True,
        grid_hw: Optional[Tuple[int, int]] = None,
        rope_base: int = 10_000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.length = length
        self.chunk_size = chunk_size
        self.num_tokens = math.ceil(length / chunk_size)
        self.original_length = length
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.rope_base = rope_base

        self.decoder_transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, self.num_tokens, n_heads, rope_base, dtype=dtype)
            for _ in range(n_layers)
        ])
        self.decode = WeightDecode(chunk_size, embed_dim, conv=conv)
        self.register_buffer(
            "pos_embed",
            build_sincos_position_embedding(self.num_tokens, embed_dim, grid_hw=grid_hw),
            persistent=False)

    def forward(self, z: torch.Tensor, original_length: Optional[int] = None) -> torch.Tensor:
        """(B, num_tokens, embed_dim) -> (B, length)."""
        z = z + self.pos_embed[:, :z.size(1)]
        x = self.decoder_transformer(z)
        return self.decode(x, original_length=original_length or self.original_length)