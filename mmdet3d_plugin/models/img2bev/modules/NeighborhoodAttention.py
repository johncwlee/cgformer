import torch
from torch import nn

# Require fused NA2D (fused neighborhood attention, heads-last)
try:
    from natten.functional import na2d
except Exception as e:
    raise ImportError(
        "Fused NATTEN na2d is required for this module. "
        "Please install a NATTEN build that provides natten.functional.na2d."
    ) from e


class NeighborhoodCrossAttention2D(nn.Module):
    """
    Neighborhood Attention 2D (cross-attention) using fused na2d only.

    Inputs: q, kv as [B, H, W, C]
    Internals: heads-last [B, H, W, Heads, D] for na2d
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        qkv_bias: bool = True,
        qk_scale: float | None = None,  # if None, uses head_dim ** -0.5
        proj_drop: float = 0.0,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__()
        assert kernel_size > 1 and kernel_size % 2 == 1, f"Kernel size must be odd and > 1, got {kernel_size}."
        assert dilation is None or dilation >= 1, f"Dilation must be >= 1, got {dilation}."
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation or 1)
        # fused na2d applies the scale internally
        self.scale = qk_scale or (self.head_dim ** -0.5)
        self.is_causal = bool(is_causal)

        # Linear projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q, kv: [B, H, W, C]
        returns: [B, H, W, C]
        """
        B, H, W, C = q.shape

        # Project and reshape to heads-last
        q_hl = self.q(q).reshape(B, H, W, self.num_heads, self.head_dim)
        kv_hlh = self.kv(kv).reshape(B, H, W, 2, self.num_heads, self.head_dim)
        k_hl = kv_hlh[:, :, :, 0, :, :]
        v_hl = kv_hlh[:, :, :, 1, :, :]

        # Fused NA2D; scaling and softmax occur inside the kernel
        x = na2d(
            q_hl, k_hl, v_hl,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
            scale=self.scale,
        )  # -> [B, H, W, Heads, D]

        # Merge heads and project
        x = x.reshape(B, H, W, C)
        x = self.proj_drop(self.proj(x))
        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"fused='na2d', causal={self.is_causal}"
        )
