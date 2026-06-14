import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from vit_vae.modules.modules import Encoder, Decoder
from vit_vae.modules.cross_attn import CrossMultiHeadAttention


#############################
# Robust Transform Functions
#############################

def log_transform(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithmic transform for heavy-tailed data."""
    return torch.sign(x) * torch.log1p(torch.abs(x) + eps)


def inverse_log_transform(x_transformed: torch.Tensor) -> torch.Tensor:
    """Inverse logarithmic transform."""
    return torch.sign(x_transformed) * torch.expm1(torch.abs(x_transformed))


#############################
# Robust Loss Functions
#############################

def huber_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        delta: float = 1.0,
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Huber loss: quadratic for small errors, linear for large errors.
    More robust to outliers than MSE.
    """
    error = pred - target
    abs_error = torch.abs(error)

    quadratic = torch.where(
        abs_error <= delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    )

    if reduction == 'mean':
        return quadratic.mean()
    elif reduction == 'sum':
        return quadratic.sum()
    else:
        return quadratic


def log_cosh_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Log-cosh loss: smooth approximation of Huber loss.
    """
    error = pred - target
    loss = torch.log(torch.cosh(error + 1e-12))

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) loss between tensors x and y.

    Args:
        x: Tensor of shape (batch_size, feature_dim)
        y: Tensor of shape (batch_size, feature_dim)
        sigma: Gaussian kernel bandwidth

    Returns:
        Scalar MMD loss
    """

    def gaussian_kernel(a, b):
        a = a.unsqueeze(1)  # (batch_size, 1, feature_dim)
        b = b.unsqueeze(0)  # (1, batch_size, feature_dim)
        dist_sq = ((a - b) ** 2).sum(dim=2)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    k_xx = gaussian_kernel(x, x).mean()
    k_yy = gaussian_kernel(y, y).mean()
    k_xy = gaussian_kernel(x, y).mean()

    return k_xx + k_yy - 2 * k_xy


def reconstruction_loss(
        recon: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = 'huber',
        huber_delta: float = 1.0,
        mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-element reconstruction loss, reduced to a scalar (scaled by 1000).

    If `mask` is given (same shape as recon, 1 = real weight, 0 = padding) the
    loss is averaged only over the real elements, so padded positions never
    contribute to the gradient.
    """
    if loss_type == 'mse':
        per = F.mse_loss(recon, target, reduction='none')
    elif loss_type == 'huber':
        per = huber_loss(recon, target, delta=huber_delta, reduction='none')
    elif loss_type == 'log_cosh':
        per = log_cosh_loss(recon, target, reduction='none')
    else:
        per = F.mse_loss(recon, target, reduction='none')

    if mask is not None:
        m = mask.to(per.dtype)
        loss = (per * m).sum() / m.sum().clamp_min(1.0)
    else:
        loss = per.mean()

    return loss * 1000


def kl_divergence_term(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prior_type: str = 'gaussian',
        student_t_df: float = 3.0,
        free_bits: float = 0.0,
) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) for a diagonal-Gaussian posterior q.

    prior_type:
      'gaussian'  -> exact closed-form KL against N(0, I) (default, unchanged
                     behaviour).
      'student_t' -> single-sample Monte-Carlo estimate against an i.i.d.
                     Student-t(df) prior. No closed form exists, so we draw a
                     reparameterized z ~ q and use E_q[log q(z) - log p(z)].
                     Heavier tails than the Gaussian tolerate outlier latents
                     better; df -> inf recovers the Gaussian.

    Returns the mean over all latent elements (same reduction as before).
    Note: with the Student-t prior the free_bits clamp is applied to the
    per-element MC term, so it is an approximation of the usual per-dimension
    free-bits scheme.
    """
    if prior_type == 'gaussian':
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    elif prior_type in ('student_t', 'student-t', 'studentt', 't'):
        logvar_c = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_c)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()  # reparameterized; the KL term has its own expectation
        df = torch.as_tensor(student_t_df, dtype=z.dtype, device=z.device)
        prior = torch.distributions.StudentT(df)  # loc=0, scale=1
        kl = q.log_prob(z) - prior.log_prob(z)
    else:
        raise ValueError(
            f"Unknown prior_type: {prior_type!r} (use 'gaussian' or 'student_t')")

    if free_bits > 0:
        kl = torch.clamp(kl - free_bits, min=0.0)

    return kl.mean()


def vae_loss(
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 0.0001,
        loss_type: str = 'huber',
        huber_delta: float = 1.0,
        mmd_weight: float = 0.1,
        free_bits: float = 0.0,
        prior_type: str = 'gaussian',
        student_t_df: float = 3.0,
        mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust VAE loss with multiple loss options.

    Args:
        recon: Reconstructed tensor
        x: Input tensor
        mu: Latent mean
        logvar: Latent log-variance
        kl_weight: Weight for KL divergence term
        loss_type: 'mse', 'huber', or 'log_cosh'
        huber_delta: Delta parameter for Huber loss
        mmd_weight: Weight for MMD regularization
        free_bits: Minimum KL per dimension (prevents posterior collapse)
        prior_type: Latent prior, 'gaussian' (default) or 'student_t'
        student_t_df: Degrees of freedom for the Student-t prior

    Returns:
        Tuple of (total_loss, recon_loss, weighted_kl_loss)
    """
    # Reconstruction loss based on type (masked over real weights if mask given)
    recon_loss = reconstruction_loss(
        recon, x, loss_type=loss_type, huber_delta=huber_delta, mask=mask)

    # Add MMD regularization (zero out padded positions so they don't count)
    if mmd_weight > 0:
        recon_flat = recon.reshape(recon.size(0), -1)
        x_flat = x.reshape(x.size(0), -1)
        if mask is not None:
            m_flat = mask.reshape(mask.size(0), -1).to(recon_flat.dtype)
            recon_flat = recon_flat * m_flat
            x_flat = x_flat * m_flat
        recon_loss = recon_loss + mmd_weight * mmd_loss(recon_flat, x_flat) * 100

    # KL divergence against the configured prior (with optional free bits)
    kl_loss = kl_divergence_term(
        mu, logvar,
        prior_type=prior_type,
        student_t_df=student_t_df,
        free_bits=free_bits,
    )

    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss * kl_weight


#############################
# Pooling Modules
#############################

class AttentionPool1D(nn.Module):
    """
    Learnable attention-based pooling: T tokens → 1 token → T tokens.
    Uses cross-attention for both compression and expansion.
    """

    def __init__(
            self,
            num_tokens: int,
            embed_dim: int,
            n_heads: int,
            rope_base: int = 10_000,
            dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Learned queries
        self.compress_q = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.expand_q = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # Cross-attention modules
        self.compress_attn = CrossMultiHeadAttention(
            d_q=embed_dim, d_kv=embed_dim, d_out=embed_dim,
            num_heads=n_heads,
            context_length_q=1,
            context_length_kv=num_tokens,
            rope_base=rope_base,
            dtype=dtype
        )
        self.expand_attn = CrossMultiHeadAttention(
            d_q=embed_dim, d_kv=embed_dim, d_out=embed_dim,
            num_heads=n_heads,
            context_length_q=num_tokens,
            context_length_kv=1,
            rope_base=rope_base,
            dtype=dtype
        )

    def down_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Compress T tokens to 1 token: (B, T, E) → (B, 1, E)"""
        B = x.shape[0]
        q = self.compress_q.expand(B, -1, -1)
        return self.compress_attn(q, x)

    def up_sample(self, z: torch.Tensor) -> torch.Tensor:
        """Expand 1 token to T tokens: (B, 1, E) → (B, T, E)"""
        B = z.shape[0]
        q = self.expand_q.expand(B, -1, -1)
        return self.expand_attn(q, z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full round-trip: (B, T, E) → (B, 1, E) → (B, T, E)"""
        z = self.down_sample(x)
        return self.up_sample(z)


class ExpandWithParams(nn.Module):
    """
    Simple expansion with learned scale and bias: (B, 1, E) → (B, T, E).
    Used with mean pooling for compression.
    """

    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_tokens, embed_dim))
        self.bias = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Expand (B, 1, E) to (B, T, E) with learned affine transform"""
        x = z.expand(-1, self.scale.size(1), -1)
        return x * self.scale + self.bias


class DepthwiseConvPool1D(nn.Module):
    """
    Depthwise convolution pooling: T tokens ↔ 1 token.
    Optionally uses tied weights for down/up operations.
    """

    def __init__(
            self,
            num_tokens: int,
            embed_dim: int,
            tied_weight: bool = True
    ):
        super().__init__()
        self.T = num_tokens
        self.E = embed_dim
        self.tied_weight = tied_weight

        if tied_weight:
            self.weight = nn.Parameter(torch.randn(embed_dim, 1, num_tokens))
        else:
            self.weight_down = nn.Parameter(torch.randn(embed_dim, 1, num_tokens))
            self.weight_up = nn.Parameter(torch.randn(embed_dim, 1, num_tokens))

    def _normalize(self, w: torch.Tensor) -> torch.Tensor:
        """L2-normalize each channel's filter"""
        return w / (w.norm(dim=[1, 2], keepdim=True) + 1e-6)

    def _w_down(self) -> torch.Tensor:
        return self._normalize(self.weight if self.tied_weight else self.weight_down)

    def _w_up(self) -> torch.Tensor:
        return self._normalize(self.weight if self.tied_weight else self.weight_up)

    def down_sample(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, E) → (B, 1, E)"""
        B, T, E = x.shape
        assert T == self.T and E == self.E, f"Expected shape (B, {self.T}, {self.E}), got {x.shape}"
        x_t = x.transpose(1, 2)  # (B, E, T)
        z_t = F.conv1d(x_t, self._w_down(), groups=E)  # (B, E, 1)
        return z_t.transpose(1, 2)  # (B, 1, E)

    def up_sample(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 1, E) → (B, T, E)"""
        B, one, E = z.shape
        assert one == 1 and E == self.E, f"Expected shape (B, 1, {self.E}), got {z.shape}"
        z_t = z.transpose(1, 2)  # (B, E, 1)
        rec_t = F.conv_transpose1d(z_t, self._w_up(), groups=E)  # (B, E, T)
        return rec_t.transpose(1, 2)  # (B, T, E)


class FlattenProject1D(nn.Module):
    """
    Flatten + linear projection: (B, T, E) ↔ (B, 1, L).
    Optionally uses tied weights (W_up = W_down^T).
    """

    def __init__(
            self,
            num_tokens: int,
            embed_dim: int,
            latent_dim: int,
            tied: bool = True
    ):
        super().__init__()
        self.T = num_tokens
        self.E = embed_dim
        self.L = latent_dim
        self.in_dim = num_tokens * embed_dim

        # Down-projection: R^(T·E) → R^L
        self.weight_down = nn.Parameter(torch.randn(latent_dim, self.in_dim) * 0.02)

        if tied:
            self.weight_up = None  # Use W_down^T
        else:
            self.weight_up = nn.Parameter(torch.randn(self.in_dim, latent_dim) * 0.02)

    def _w_up(self) -> torch.Tensor:
        return self.weight_down.t() if self.weight_up is None else self.weight_up

    def down_sample(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, E) → (B, 1, L)"""
        B, T, E = x.shape
        assert T == self.T and E == self.E, f"Expected shape (B, {self.T}, {self.E}), got {x.shape}"
        flat = x.reshape(B, -1)  # (B, T·E)
        z = torch.matmul(flat, self.weight_down.t())  # (B, L)
        return z.unsqueeze(1)  # (B, 1, L)

    def up_sample(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 1, L) → (B, T, E)"""
        if len(z.shape) < 3:
            z = z.unsqueeze(1)
        B, one, L = z.shape
        assert one == 1 and L == self.L, f"Expected shape (B, 1, {self.L}), got {z.shape}"
        z = z.squeeze(1)  # (B, L)
        rec = torch.matmul(z, self._w_up().t())  # (B, T·E)
        return rec.reshape(B, self.T, self.E)


class LinearProject(nn.Module):
    """
    Simple linear projection for bottleneck: (B, T, E) ↔ (B, T, L).
    """

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.down = nn.Linear(in_dim, latent_dim)
        self.up = nn.Linear(latent_dim, in_dim)

    def down_sample(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

    def up_sample(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(z)


class SeqPooling(nn.Module):
    """
    Conv1d-based sequence pooling: T tokens ↔ L tokens.
    """

    def __init__(self, num_tokens: int, latent_tokens: int):
        super().__init__()
        self.down = nn.Conv1d(
            num_tokens, latent_tokens,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.up = nn.Conv1d(
            latent_tokens, num_tokens,
            kernel_size=3, stride=1, padding=1, bias=False
        )

    def down_sample(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, E) → (B, L, E)"""
        return self.down(x)

    def up_sample(self, z: torch.Tensor) -> torch.Tensor:
        """(B, L, E) → (B, T, E)"""
        return self.up(z)

class MuLawTransform(nn.Module):
    def __init__(self, mu: float = 255.0):
        super().__init__()
        self.register_buffer('mu', torch.tensor(mu))

    def forward(self, x):
        # "Spreads" data near zero, compresses outliers
        numerator = torch.log1p(self.mu * torch.abs(x))
        denominator = torch.log1p(self.mu)
        return torch.sign(x) * (numerator / denominator)

    def inverse(self, y):
        # Mathematical inverse to restore original scale
        numerator = torch.expm1(torch.abs(y) * torch.log1p(self.mu))
        return torch.sign(y) * (numerator / self.mu)

class MuLawCompander(nn.Module):
    def __init__(self, mu: float = 255.0, gain: float = 20.0):
        super().__init__()
        self.register_buffer('mu', torch.tensor(mu))
        self.register_buffer('gain', torch.tensor(gain))

    def forward(self, x):
        """
        1. Amplify signal by 'gain' (0.04 -> 0.8)
        2. Apply Mu-Law to spread near-zero values
        """
        x = x * self.gain
        numerator = torch.log1p(self.mu * torch.abs(x))
        denominator = torch.log1p(self.mu)
        return torch.sign(x) * (numerator / denominator)

    def inverse(self, y):
        """
        1. Inverse Mu-Law
        2. Divide by 'gain' to return to original scale
        """
        numerator = torch.expm1(torch.abs(y) * torch.log1p(self.mu))
        x_unscaled = torch.sign(y) * (numerator / self.mu)
        return x_unscaled / self.gain

#############################
# Main Autoencoder
#############################

class AutoencoderKL(nn.Module):
    def __init__(
            self,
            learning_rate: float,
            enconfig: Dict[str, Any],
            deconfig: Dict[str, Any],
            embed_dim: int,
            latent_token: int = 0,
            flat_dim: int = 1024,
            mean_pooling: str = None,
            ckpt_path: Optional[str] = None,
            ignore_keys: List[str] = [],
            input_key: str = "weight",
            cond_key: str = "dataset",
            freeze_encoder: bool = False,
            freeze_decoder: bool = False,
            neck_only: bool = False,
            device: str = 'cuda',
            use_vae: bool = False,
            tied_weight: bool = False,
            monitor: Optional[str] = None,
            # Robustness parameters
            loss_type: str = 'huber',
            huber_delta: float = 1.0,
            gradient_clip_val: float = 1.0,
            free_bits: float = 0.0,
            use_log_transform: bool = False,
            # Latent prior
            prior_type: str = 'gaussian',
            student_t_df: float = 3.0,
            ):
        super().__init__()
        self.device = device
        self.cond_key = cond_key
        self.learning_rate = learning_rate
        self.input_key = input_key
        self.use_vae = use_vae
        self.tied_weight = tied_weight
        self.flat_dim = flat_dim
        self.mean_pooling = mean_pooling
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.neck_only = neck_only

        # Robustness parameters
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.gradient_clip_val = gradient_clip_val
        self.free_bits = free_bits
        self.use_log_transform = use_log_transform
        # self.mu_transform = MuLawCompander(mu=1000, gain=20)

        # Latent prior (default: standard Gaussian)
        self.prior_type = prior_type
        self.student_t_df = student_t_df

        # Initialize encoder and decoder
        self.encoder = Encoder(**enconfig).to(device)
        self.decoder = Decoder(**deconfig).to(device)

        # Store dimensions
        self.length = enconfig['length']
        self.num_tokens = math.ceil(enconfig['length'] / enconfig['chunk_size'])
        self.latent_token = latent_token
        self.embed_dim = embed_dim

        # Projection layers
        self.tokens_to_latent = nn.Linear(enconfig['embed_dim'], enconfig['latent_dim'])
        self.latent_to_tokens = nn.Linear(deconfig['latent_dim'], deconfig['embed_dim'])
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, deconfig['embed_dim'], device=self.device) * 0.02
        )

        # Initialize pooling module
        num_tokens = self.num_tokens
        n_heads = self.encoder.n_heads
        rope_base = self.encoder.rope_base

        if mean_pooling == 'attn':
            self.pool = AttentionPool1D(
                num_tokens, enconfig['latent_dim'], n_heads, rope_base=rope_base
            )
        elif mean_pooling == 'mean':
            self.pool = ExpandWithParams(num_tokens, deconfig['latent_dim'])
        elif mean_pooling == 'depth':
            self.pool = DepthwiseConvPool1D(
                num_tokens, deconfig['latent_dim'], tied_weight=tied_weight
            )
        elif mean_pooling == 'proj':
            self.pool = FlattenProject1D(
                num_tokens=num_tokens,
                embed_dim=deconfig['latent_dim'],
                latent_dim=flat_dim,
                tied=tied_weight
            )
        elif mean_pooling == 'tok':
            if latent_token == 0:
                latent_token = num_tokens // 2
            self.latent_token = latent_token
            self.pool = SeqPooling(num_tokens, latent_tokens=latent_token)
        elif mean_pooling == 'linear':
            self.pool = LinearProject(enconfig['latent_dim'], enconfig['latent_dim'])
        else:
            self.pool = None

        # VAE parameters
        if use_vae:
            latent_size = flat_dim if mean_pooling == 'proj' else enconfig['latent_dim']
            self.fc_mu = nn.Linear(latent_size, latent_size)
            self.fc_logvar = nn.Linear(latent_size, latent_size)

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # Apply freezing
        if freeze_encoder:
            self._freeze_module(self.encoder)
        if freeze_decoder:
            self._freeze_module(self.decoder)
        if neck_only:
            self._freeze_module(self.tokens_to_latent)
            self._freeze_module(self.latent_to_tokens)

        self.monitor = monitor

    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in a module"""
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = []):
        """Load model weights from checkpoint"""
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]

        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick with clamping for stability"""
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_prior(self, shape, device: Optional[str] = None) -> torch.Tensor:
        """
        Draw latent samples from the configured prior (for generation).

        shape: full latent shape including batch, e.g. (B, latent_dim) or
               (B, 1, latent_dim) depending on the pooling variant.
        """
        device = device or self.device
        if self.prior_type == 'gaussian':
            return torch.randn(*shape, device=device)
        df = torch.as_tensor(self.student_t_df, dtype=torch.float32, device=device)
        return torch.distributions.StudentT(df).sample(torch.Size(shape)).to(device)

    def encode(self, x):
        # Optional log transform for heavy-tailed data
        # if self.use_log_transform:
        #     x = log_transform(x)
        # x = self.mu_transform(x)
        # Encode to tokens
        h = self.encoder(x)
        z = self.tokens_to_latent(h)

        # Apply pooling
        if self.mean_pooling == 'mean':
            z = z.mean(dim=1)
        elif self.mean_pooling != 'mean' and self.mean_pooling is not None:
            z = self.pool.down_sample(z)

        # VAE latent space
        if self.use_vae:
            mu = self.fc_mu(z)
            logvar = self.fc_logvar(z)
            z = self.reparameterize(mu, logvar)
        else:
            mu = None
            logvar = None

        return z, mu, logvar

    def decode(self, z: torch.Tensor):
        # Apply unpooling
        if self.mean_pooling == 'mean':
            z = z.unsqueeze(1)
            z = self.pool(z)
        elif self.mean_pooling != 'mean' and self.mean_pooling is not None:
            z = self.pool.up_sample(z)

        # Project back to token dimension
        h = self.latent_to_tokens(z)

        h = h + self.decoder_pos_embed
        # Decode to output
        dec = self.decoder(h)

        # Reverse log transform if used
        # if self.use_log_transform:
        #     dec = inverse_log_transform(dec)
        # dec = self.mu_transform.inverse(dec)

        return dec

    def forward(self, batch):
        # Handle dict input
        if isinstance(batch, dict):
            x = batch[self.input_key]
            if not x.is_cuda:
                x = x.cuda()
        else:
            x = batch

        # Encode
        z, mu, logvar = self.encode(x)
        # Decode
        dec = self.decode(z)
        dec = dec.reshape(x.shape)
        return x, dec, mu, logvar

    def compute_loss(
            self,
            recon: torch.Tensor,
            target: torch.Tensor,
            mu: Optional[torch.Tensor] = None,
            logvar: Optional[torch.Tensor] = None,
            kl_weight: float = 0.0001,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute robust loss with the specified loss type.

        mask: optional (same shape as recon) padding mask, 1 = real weight,
              0 = padding. When given, padded positions are excluded from both
              the reconstruction loss and the MMD term.

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if mask is not None:
            mask = mask.to(recon.dtype).reshape(recon.shape)

        loss_dict = {}

        if self.use_vae and mu is not None and logvar is not None:
            total_loss, recon_loss, kl_loss = vae_loss(
                recon, target, mu, logvar,
                kl_weight=kl_weight,
                loss_type=self.loss_type,
                huber_delta=self.huber_delta,
                mmd_weight=0.1,
                free_bits=self.free_bits,
                prior_type=self.prior_type,
                student_t_df=self.student_t_df,
                mask=mask,
            )
            loss_dict['recon_loss'] = recon_loss
            loss_dict['kl_loss'] = kl_loss
        else:
            # Non-VAE: just reconstruction loss (masked over real weights)
            recon_loss = reconstruction_loss(
                recon, target, loss_type=self.loss_type,
                huber_delta=self.huber_delta, mask=mask)

            # Add MMD (zero out padded positions if masked)
            recon_flat = recon.reshape(recon.size(0), -1)
            target_flat = target.reshape(target.size(0), -1)
            if mask is not None:
                m_flat = mask.reshape(mask.size(0), -1)
                recon_flat = recon_flat * m_flat
                target_flat = target_flat * m_flat
            recon_loss = recon_loss + 0.1 * mmd_loss(recon_flat, target_flat) * 100

            total_loss = recon_loss
            loss_dict['recon_loss'] = recon_loss

        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict

    def get_input(self, batch: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        """Extract input from batch dict"""
        x = batch[k].to(self.device)
        return x

    def configure_optimizers(self):
        """Configure optimizer with gradient clipping."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.90, 0.99),
            weight_decay=0.0,
        )

        return optimizer

    def on_before_optimizer_step(self, optimizer):
        """Apply gradient clipping before optimizer step"""
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=self.gradient_clip_val
            )

class IdentityFirstStage(nn.Module):
    """
    Identity first stage for compatibility.
    Passes through input without modification.
    """

    def __init__(self, *args, vq_interface: bool = False, **kwargs):
        super().__init__()
        self.vq_interface = vq_interface

    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x

    def decode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x

    def quantize(self, x: torch.Tensor, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x