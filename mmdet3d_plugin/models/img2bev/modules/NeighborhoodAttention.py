#################################################################################################
# Copyright (c) 2023 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
import torch
from torch import nn
from torch.nn.functional import pad

try:
    # Fused path (preferred)
    from natten.functional import na2d
except ImportError:
    # Legacy unfused path not available; raise early if missing
    raise ImportError("Fused NATTEN na2d is required for this RoPE refactor.")


class NeighborhoodCrossAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True, # kept for API compatibility; no RPB is used
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_base=10000.0,      # RoPE frequency base
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        #* Rotary Positional Embeddings (2D) instead of deprecated rpb
        #* RoPE requires even pairs along each axis; enforce 4-way divisibility (pairs for H and W)
        assert (self.head_dim % 4) == 0, (
            f"head_dim ({self.head_dim}) must be divisible by 4 for axial 2D RoPE."
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        #* RPB removed; keep a placeholder attribute for repr compatibility
        self.rpb = None

        #* RoPE configuration and lazy caches
        #  Keep caches as plain attributes (not registered buffers) to avoid SWA/EMA averaging
        #  attempting to copy/average them across varying spatial sizes.
        self.rope_base = rope_base
        self._cos_h = None
        self._sin_h = None
        self._cos_w = None
        self._sin_w = None
        self._rope_cache_shape = None  # (H, W, dtype, device)

        #* Optional: causal flag for na2d
        self.is_causal = False

    @staticmethod
    def _rope_rotate_half(x):
        # x: (..., d), pair dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # rotate (x_even, x_odd) -> (-x_odd, x_even)
        return torch.stack((-x_odd, x_even), dim=-1).reshape(x.shape)

    def _apply_rope_2d(self, x, sin_h, cos_h, sin_w, cos_w):
        # x: (B, heads, H, W, head_dim)
        head_dim = x.shape[-1]
        half_dim = head_dim // 2
        x_h = x[..., :half_dim]
        x_w = x[..., half_dim:]
        # Apply RoPE along height and width halves separately
        x_h = (x_h * cos_h) + (self._rope_rotate_half(x_h) * sin_h)
        x_w = (x_w * cos_w) + (self._rope_rotate_half(x_w) * sin_w)
        return torch.cat([x_h, x_w], dim=-1)

    def _build_rope_cache(self, H, W, dtype, device):
        # Build 2D RoPE caches with broadcasting-friendly shapes
        d = self.head_dim
        d_half = d // 2
        d_h = d_half
        d_w = d_half
        assert (d_h % 2) == 0 and (d_w % 2) == 0, "Each axial half of head_dim must be even."

        def inv_freq(dim_ax):
            return 1.0 / (self.rope_base ** (torch.arange(0, dim_ax, 2, device=device, dtype=dtype) / dim_ax))

        inv_h = inv_freq(d_h)
        inv_w = inv_freq(d_w)

        pos_h = torch.arange(H, device=device, dtype=dtype)
        pos_w = torch.arange(W, device=device, dtype=dtype)

        freqs_h = torch.outer(pos_h, inv_h)  # (H, d_h/2)
        freqs_w = torch.outer(pos_w, inv_w)  # (W, d_w/2)

        cos_h = torch.repeat_interleave(torch.cos(freqs_h), repeats=2, dim=-1)  # (H, d_h)
        sin_h = torch.repeat_interleave(torch.sin(freqs_h), repeats=2, dim=-1)
        cos_w = torch.repeat_interleave(torch.cos(freqs_w), repeats=2, dim=-1)  # (W, d_w)
        sin_w = torch.repeat_interleave(torch.sin(freqs_w), repeats=2, dim=-1)

        self._cos_h = cos_h.view(1, 1, H, 1, d_h)
        self._sin_h = sin_h.view(1, 1, H, 1, d_h)
        self._cos_w = cos_w.view(1, 1, 1, W, d_w)
        self._sin_w = sin_w.view(1, 1, 1, W, d_w)
        self._rope_cache_shape = (H, W, dtype, device)

    def forward(self, q, kv):
        B, Hp, Wp, C = q.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            # Pad q and kv on spatial dims; convert to NCHW for pad semantics
            q = q.permute(0, 3, 1, 2)
            kv = kv.permute(0, 3, 1, 2)
            q = pad(q, (0, pad_r, 0, pad_b))
            kv = pad(kv, (0, pad_r, 0, pad_b))
            q = q.permute(0, 2, 3, 1)
            kv = kv.permute(0, 2, 3, 1)
            _, H, W, _ = q.shape
        
        # Projections -> heads-first
        q = self.q(q).reshape(B, H, W, 1, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5).squeeze(0)  # [B, Heads, H, W, D]
        kv = self.kv(kv).reshape(B, H, W, 2, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]

        # Scale queries
        q = q * self.scale

        # Build/refresh RoPE caches if shape/dtype/device changed
        cache_needed = (
            self._rope_cache_shape is None
            or self._rope_cache_shape[0] != H
            or self._rope_cache_shape[1] != W
            or self._rope_cache_shape[2] != q.dtype
            or self._rope_cache_shape[3] != q.device
        )
        if cache_needed:
            self._build_rope_cache(H, W, q.dtype, q.device)

        # Apply 2D RoPE to q and k (heads-first)
        q = self._apply_rope_2d(q, self._sin_h, self._cos_h, self._sin_w, self._cos_w)
        k = self._apply_rope_2d(k, self._sin_h, self._cos_h, self._sin_w, self._cos_w)

        # Fused NA expects heads-last: [B, H, W, Heads, D]
        q_hl = q.permute(0, 2, 3, 1, 4).contiguous()
        k_hl = k.permute(0, 2, 3, 1, 4).contiguous()
        v_hl = v.permute(0, 2, 3, 1, 4).contiguous()

        x = na2d(
            q_hl, k_hl, v_hl,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=getattr(self, "is_causal", False),
            scale=None,  # already applied to q
        )
        # x: [B, H, W, Heads, D] -> [B, Heads, H, W, D] -> [B, H, W, C]
        x = x.permute(0, 3, 1, 2, 4)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rope=True"
        )
