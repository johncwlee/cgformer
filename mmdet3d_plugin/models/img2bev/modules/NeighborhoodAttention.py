import math
import torch
from torch import nn
from torch.nn.functional import pad

try:
    # Fused path (preferred) — heads-last tensors and in-kernel scaling supported
    from natten.functional import na2d
except ImportError:
    raise ImportError("Fused NATTEN na2d is required for this refactor.")

# Minimal DropPath (stochastic depth) to approximate attention regularization at the block edge
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        # Per-sample dropping, broadcast over remaining dims
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep_prob)
        return x * rnd.div(keep_prob)

class DepthwiseCPE2d(nn.Module):
    """
    Lightweight convolutional positional encoding (CPE).
    Injects a learnable local positional prior without modifying attention biases.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size >= 1, "CPE kernel_size must be odd and >= 1."
        self.proj = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=dim, bias=True)

    def forward(self, x_bhwc: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C] -> [B, C, H, W] -> CPE -> [B, H, W, C]
        x = x_bhwc.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class NeighborhoodCrossAttention2D(nn.Module):
    """
    Neighborhood Attention 2D with fused NATTEN na2d, interleaved 2D RoPE,
    optional ConvPosEnc (CPE), and output regularization.

    Shapes:
      - Inputs: q, kv as [B, H, W, C]
      - Internal: heads-last [B, H, W, Heads, D] for na2d
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,        # approximated via output dropout
        proj_drop: float = 0.0,
        drop_path: float = 0.0,        # stochastic depth to mimic attention dropout
        rope_base: float = 1000.0,     # reduced base for image grids
        learnable_axis_scale: bool = True,
        cpe_kernel_size: int | None = 3,  # set None/0 to disable
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__()
        assert kernel_size > 1 and kernel_size % 2 == 1, f"Kernel size must be odd and > 1, got {kernel_size}."
        assert dilation is None or dilation >= 1, f"Dilation must be >= 1, got {dilation}."
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        assert (self.head_dim % 2) == 0, f"head_dim ({self.head_dim}) must be even for RoPE pair rotation."
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation or 1)
        # Effective minimal window to avoid degenerate padding when H/W are small
        self.window_size = self.kernel_size * self.dilation
        self.scale = qk_scale or self.head_dim ** -0.5

        # Projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        # Optional local absolute pos. prior (CPE) applied to inputs separately (q and kv) to ensure shared anchoring
        self.cpe = None
        if cpe_kernel_size and cpe_kernel_size > 0:
            self.cpe = DepthwiseCPE2d(dim, kernel_size=cpe_kernel_size)

        # Interleaved 2D RoPE config
        self.rope_base = float(rope_base)
        self.learnable_axis_scale = bool(learnable_axis_scale)
        if self.learnable_axis_scale:
            # Log-scales ensure positivity when exponentiated
            self.log_scale_h = nn.Parameter(torch.zeros(()))  # scalar per axis
            self.log_scale_w = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_scale_h", None)
            self.register_parameter("log_scale_w", None)

        # Regularization
        # Fused na2d has no attention dropout argument, so approximate via output dropout/stochastic depth
        self.attn_out_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # API compatibility flags
        self.is_causal = bool(is_causal)
        self.rpb = None  # RPB removed by fused kernels

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Rotate last-dim pairs: (x_even, x_odd) -> (-x_odd, x_even)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape(x.shape)

    def _rope_2d_interleaved(self, x_hl: torch.Tensor) -> torch.Tensor:
        """
        Apply interleaved 2D RoPE on heads-last tensor.
        x_hl: [B, H, W, Heads, D]
        Rotations are applied to the same 2-D subspaces sequentially for H and W to capture diagonals.
        """
        B, H, W, Nh, D = x_hl.shape
        device, dtype = x_hl.device, x_hl.dtype
        # Frequency grid for D/2 pairs
        pair_dim = D // 2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, D, 2, device=device, dtype=dtype) / D))
        # Optional learnable per-axis scale on positions
        s_h = torch.exp(self.log_scale_h) if self.learnable_axis_scale else torch.tensor(1.0, device=device, dtype=dtype)
        s_w = torch.exp(self.log_scale_w) if self.learnable_axis_scale else torch.tensor(1.0, device=device, dtype=dtype)
        pos_h = torch.arange(H, device=device, dtype=dtype) * s_h
        pos_w = torch.arange(W, device=device, dtype=dtype) * s_w

        # Build cos/sin with pair interleaving -> broadcast to full D
        # freqs: [H, D/2] and [W, D/2]
        freqs_h = torch.outer(pos_h, inv_freq)
        freqs_w = torch.outer(pos_w, inv_freq)
        cos_h = torch.repeat_interleave(torch.cos(freqs_h), repeats=2, dim=-1)  # [H, D]
        sin_h = torch.repeat_interleave(torch.sin(freqs_h), repeats=2, dim=-1)
        cos_w = torch.repeat_interleave(torch.cos(freqs_w), repeats=2, dim=-1)  # [W, D]
        sin_w = torch.repeat_interleave(torch.sin(freqs_w), repeats=2, dim=-1)

        # Broadcast to [1, H, 1, 1, D] and [1, 1, W, 1, D]
        cos_h = cos_h.view(1, H, 1, 1, D)
        sin_h = sin_h.view(1, H, 1, 1, D)
        cos_w = cos_w.view(1, 1, W, 1, D)
        sin_w = sin_w.view(1, 1, W, 1, D)

        # Sequential rotations on the same pairs: first along H, then along W
        x = x_hl
        x = (x * cos_h) + (self._rotate_half(x) * sin_h)
        x = (x * cos_w) + (self._rotate_half(x) * sin_w)
        return x

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q, kv: [B, H, W, C]
        returns: [B, H, W, C]
        """
        B, Hp, Wp, C = q.shape
        H, W = int(Hp), int(Wp)

        # Pad if spatial dims are smaller than the effective window
        pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            # NCHW for pad
            q_nchw = q.permute(0, 3, 1, 2)
            kv_nchw = kv.permute(0, 3, 1, 2)
            q_nchw = pad(q_nchw, (0, pad_r, 0, pad_b))
            kv_nchw = pad(kv_nchw, (0, pad_r, 0, pad_b))
            q = q_nchw.permute(0, 2, 3, 1).contiguous()
            kv = kv_nchw.permute(0, 2, 3, 1).contiguous()
            _, H, W, _ = q.shape

        # Optional CPE applied independently to q and kv to provide shared absolute anchors
        if self.cpe is not None:
            q = self.cpe(q)
            kv = self.cpe(kv)

        # Linear projections to heads-last
        q_hl = self.q(q).view(B, H, W, self.num_heads, self.head_dim)
        kv_hlh = self.kv(kv).view(B, H, W, 2, self.num_heads, self.head_dim)
        k_hl = kv_hlh[:, :, :, 0, :, :]
        v_hl = kv_hlh[:, :, :, 1, :, :]

        # Interleaved 2D RoPE on q and k (heads-last)
        q_hl = self._rope_2d_interleaved(q_hl)
        k_hl = self._rope_2d_interleaved(k_hl)

        # Fused NA2D (heads-last), scale handled in-kernel
        x = na2d(
            q_hl, k_hl, v_hl,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
            scale=self.scale,
        )  # -> [B, H, W, Heads, D]

        # Merge heads
        x = x.reshape(B, H, W, C)

        # Remove padding if applied
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        # Output projection and regularization
        x = self.proj(x)
        x = self.attn_out_drop(x)
        # Note: In a standard Transformer block, DropPath is applied to the residual branch
        # outside this module; here we allow optional in-module DropPath for convenience.
        x = self.drop_path(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rope_base={self.rope_base}, learnable_axis_scale={self.learnable_axis_scale}, "
            + f"cpe={'on' if self.cpe is not None else 'off'}"
        )
