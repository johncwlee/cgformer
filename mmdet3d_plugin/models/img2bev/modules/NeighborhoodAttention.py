import math
import torch
from torch import nn

try:
    from natten.functional import na2d  # fused heads-last NA2D
except ImportError:
    raise ImportError("Fused NATTEN na2d is required for this refactor.")


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask.div(keep)


class DepthwiseCPE2d(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size >= 1
        self.proj = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=dim, bias=True)

    def forward(self, x_bhwc: torch.Tensor) -> torch.Tensor:
        x = x_bhwc.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class RMSNormLast(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        # x: [..., D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.weight is not None:
            x = x * self.weight
        return x


class NeighborhoodCrossAttention2D(nn.Module):
    """
    Stable fused Neighborhood Attention 2D with:
      - pre-norm on q and kv
      - QK RMS-norm
      - float32 RoPE angles + bounded learnable scales/frequencies
      - optional ConvPosEnc (CPE)
      - no manual input padding; per-dim kernel_size clamps to H and W

    Inputs: q, kv as [B, H, W, C]; output: [B, H, W, C]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.1,    # output-side reg (no in-kernel dropout in na2d)
        proj_drop: float = 0.1,
        drop_path: float = 0.1,
        rope_base: float = 1000.0,
        learnable_axis_scale: bool = True,
        cpe_kernel_size: int | None = 3,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__()
        assert kernel_size > 1 and kernel_size % 2 == 1
        assert dilation is None or dilation >= 1
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        assert (self.head_dim % 2) == 0, "head_dim must be even for RoPE."
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation or 1)
        self.scale = qk_scale or self.head_dim ** -0.5

        # Pre-norms for stability (q and kv may come from different sources)
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)

        # Projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        # Optional local absolute positional prior
        self.cpe = DepthwiseCPE2d(dim, kernel_size=cpe_kernel_size) if (cpe_kernel_size and cpe_kernel_size > 0) else None

        # Mixed 2D RoPE with learnable per-head frequencies; stabilized
        self.rope_base = float(rope_base)
        pair_dim = self.head_dim // 2
        inv_freq_init = (1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))).view(1, pair_dim)
        # raw (unbounded) parameters; apply tanh + limit in forward
        self.theta_x_raw = nn.Parameter(inv_freq_init.repeat(self.num_heads, 1))  # [H, pair_dim]
        self.theta_y_raw = nn.Parameter(inv_freq_init.repeat(self.num_heads, 1))  # [H, pair_dim]
        # Reasonable bound to keep sin/cos arguments in a safe range in reduced precision
        self.register_buffer("theta_limit", torch.tensor(10.0), persistent=False)

        self.learnable_axis_scale = bool(learnable_axis_scale)
        if self.learnable_axis_scale:
            # clamp log-scales in forward to avoid runaway angle growth
            self.log_scale_h = nn.Parameter(torch.zeros(()))
            self.log_scale_w = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_scale_h", None)
            self.register_parameter("log_scale_w", None)

        # QK normalization to tame logits (affine=False keeps it minimal)
        self.qk_norm = RMSNormLast(self.head_dim, eps=1e-6, affine=False)

        # Output path regularization (attention dropout not available in fused NA2D)
        self.attn_out_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.is_causal = bool(is_causal)
        self.rpb = None  # fused kernels remove RPB

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape(x.shape)

    def _rope_2d_mixed(self, x_hl: torch.Tensor) -> torch.Tensor:
        """
        Compute mixed 2D RoPE in float32 for stability, then cast to x.dtype.
        x_hl: [B, H, W, Heads, D]
        """
        B, H, W, Nh, D = x_hl.shape
        device = x_hl.device
        dtype = x_hl.dtype
        pair_dim = D // 2

        # Bound axis scales and convert positions to float32
        if self.learnable_axis_scale:
            log_h = torch.clamp(self.log_scale_h, min=-4.0, max=4.0)
            log_w = torch.clamp(self.log_scale_w, min=-4.0, max=4.0)
            s_h = torch.exp(log_h).to(torch.float32)
            s_w = torch.exp(log_w).to(torch.float32)
        else:
            s_h = torch.tensor(1.0, device=device, dtype=torch.float32)
            s_w = torch.tensor(1.0, device=device, dtype=torch.float32)

        pos_h = torch.arange(H, device=device, dtype=torch.float32) * s_h  # [H]
        pos_w = torch.arange(W, device=device, dtype=torch.float32) * s_w  # [W]

        # Bound per-head frequencies via tanh and a fixed limit
        theta_limit = self.theta_limit.to(device=device, dtype=torch.float32)
        theta_x = torch.tanh(self.theta_x_raw).to(torch.float32) * theta_limit  # [Nh, pair_dim]
        theta_y = torch.tanh(self.theta_y_raw).to(torch.float32) * theta_limit  # [Nh, pair_dim]

        # angle(h,w,head,t) = pos_h[h]*theta_x + pos_w[w]*theta_y  -> [H,W,Nh,pair_dim]
        angle = (
            pos_h.view(H, 1, 1, 1) * theta_x.view(1, 1, Nh, pair_dim) +
            pos_w.view(1, W, 1, 1) * theta_y.view(1, 1, Nh, pair_dim)
        )

        # cos/sin in float32 then expand to D
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1).view(1, H, W, Nh, D)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1).view(1, H, W, Nh, D)

        # Apply rotation and cast back
        x = (x_hl.to(torch.float32) * cos) + (self._rotate_half(x_hl.to(torch.float32)) * sin)
        return x.to(dtype)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q, kv: [B, H, W, C] -> returns [B, H, W, C]
        """
        B, H, W, C = q.shape

        # Optional CPE first (absolute prior), then pre-norm
        if self.cpe is not None:
            q = self.cpe(q)
            kv = self.cpe(kv)
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Projections -> heads-last
        q_hl = self.q(q).view(B, H, W, self.num_heads, self.head_dim).contiguous()
        kv_hlh = self.kv(kv).view(B, H, W, 2, self.num_heads, self.head_dim).contiguous()
        k_hl = kv_hlh[:, :, :, 0, :, :].contiguous()
        v_hl = kv_hlh[:, :, :, 1, :, :].contiguous()

        # RoPE (stable) then QK RMS-norm
        q_hl = self._rope_2d_mixed(q_hl)
        k_hl = self._rope_2d_mixed(k_hl)
        q_hl = self.qk_norm(q_hl)
        k_hl = self.qk_norm(k_hl)

        # Use per-dimension kernel that never exceeds H or W; let NA2D handle borders
        ks_h = min(self.kernel_size, H)
        ks_w = min(self.kernel_size, W)
        ks = (ks_h, ks_w)

        # Fused NA2D
        x = na2d(
            q_hl, k_hl, v_hl,
            kernel_size=ks,
            dilation=self.dilation,
            is_causal=self.is_causal,
            scale=self.scale,
        )  # [B, H, W, Heads, D]

        # Merge heads and output path regularization
        x = x.reshape(B, H, W, C)
        x = self.proj(x)
        x = self.attn_out_drop(x)
        x = self.drop_path(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rope_base={self.rope_base}, mixed_rope='on', "
            + f"axis_scale={self.learnable_axis_scale}, "
            + f"cpe={'on' if self.cpe is not None else 'off'}"
        )
