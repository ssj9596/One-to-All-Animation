'''
原始repo的实现
'''
import os
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from opensora.model_variants.mochi_layers import (
    FeedForward,
    PatchEmbed,
    RMSNorm,
    TimestepEmbedder,
)

from opensora.model_variants.mochi_utils import (
    AttentionPool,
    modulate,
    pad_and_split_xy,
)

import functools
import math

# Based on Llama3 Implementation.
import torch


def apply_rotary_emb_qk_real(
    xqk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor without complex numbers.

    Args:
        xqk (torch.Tensor): Query and/or Key tensors to apply rotary embeddings. Shape: (B, S, *, num_heads, D)
                            Can be either just query or just key, or both stacked along some batch or * dim.
        freqs_cos (torch.Tensor): Precomputed cosine frequency tensor.
        freqs_sin (torch.Tensor): Precomputed sine frequency tensor.

    Returns:
        torch.Tensor: The input tensor with rotary embeddings applied.
    """
    assert xqk.dtype == torch.bfloat16
    # Split the last dimension into even and odd parts
    xqk_even = xqk[..., 0::2]
    xqk_odd = xqk[..., 1::2]

    # Apply rotation
    cos_part = (xqk_even * freqs_cos - xqk_odd * freqs_sin).type_as(xqk)
    sin_part = (xqk_even * freqs_sin + xqk_odd * freqs_cos).type_as(xqk)

    # Interleave the results back into the original shape
    out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2)
    assert out.dtype == torch.bfloat16
    return out

def centers(start: float, stop, num, dtype=None, device=None):
    """linspace through bin centers.

    Args:
        start (float): Start of the range.
        stop (float): End of the range.
        num (int): Number of points.
        dtype (torch.dtype): Data type of the points.
        device (torch.device): Device of the points.

    Returns:
        centers (Tensor): Centers of the bins. Shape: (num,).
    """
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2

def create_position_matrix(
    T: int,
    pH: int,
    pW: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    target_area: float = 36864,
):
    """
    Args:
        T: int - Temporal dimension
        pH: int - Height dimension after patchify
        pW: int - Width dimension after patchify

    Returns:
        pos: [T * pH * pW, 3] - position matrix
    """
    with torch.no_grad():
        # Create 1D tensors for each dimension
        t = torch.arange(T, dtype=dtype)

        # Positionally interpolate to area 36864.
        # (3072x3072 frame with 16x16 patches = 192x192 latents).
        # This automatically scales rope positions when the resolution changes.
        # We use a large target area so the model is more sensitive
        # to changes in the learned pos_frequencies matrix.
        scale = math.sqrt(target_area / (pW * pH))
        w = centers(-pW * scale / 2, pW * scale / 2, pW)
        h = centers(-pH * scale / 2, pH * scale / 2, pH)

        # Use meshgrid to create 3D grids
        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")

        # Stack and reshape the grids.
        pos = torch.stack([grid_t, grid_h, grid_w], dim=-1)  # [T, pH, pW, 3]
        pos = pos.view(-1, 3)  # [T * pH * pW, 3]
        pos = pos.to(dtype=dtype, device=device)

    return pos


def compute_mixed_rotation(
    freqs: torch.Tensor,
    pos: torch.Tensor,
):
    """
    Project each 3-dim position into per-head, per-head-dim 1D frequencies.

    Args:
        freqs: [3, num_heads, num_freqs] - learned rotation frequency (for t, row, col) for each head position
        pos: [N, 3] - position of each token
        num_heads: int

    Returns:
        freqs_cos: [N, num_heads, num_freqs] - cosine components
        freqs_sin: [N, num_heads, num_freqs] - sine components
    """
    assert freqs.ndim == 3
    freqs_sum = torch.einsum("Nd,dhf->Nhf", pos.to(freqs), freqs)
    freqs_cos = torch.cos(freqs_sum)
    freqs_sin = torch.sin(freqs_sum)
    return freqs_cos, freqs_sin


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    # Convert to fp32 for precision
    x_res = x_res.float()

    # Compute RMS
    mean_square = x_res.pow(2).mean(-1, keepdim=True)
    scale = torch.rsqrt(mean_square + eps)

    # Apply tanh to gate
    tanh_gate = torch.tanh(gate).unsqueeze(1)

    # Normalize and apply gated scaling
    x_normed = x_res * scale * tanh_gate

    # Apply residual connection
    output = x + x_normed.type_as(x)
    return output

def modulated_rmsnorm(x, scale, eps=1e-6):
    dtype = x.dtype
    x = x.float()

    # Compute RMS
    mean_square = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)

    # Normalize and modulate
    x_normed = x * inv_rms
    x_modulated = x_normed * (1 + scale.unsqueeze(1).float())
    return x_modulated.to(dtype)


def compute_packed_indices(
    device: torch.device, text_mask: torch.Tensor, num_latents: int
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        num_latents: Number of latent tokens
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    PATCH_SIZE = 2
    num_visual_tokens = num_latents // (PATCH_SIZE**2)
    assert num_visual_tokens > 0

    mask = F.pad(text_mask, (num_visual_tokens, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()  # up to (B * (N + L),)
    assert valid_token_indices.size(0) >= text_mask.size(0) * num_visual_tokens  # At least (B * N,)
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens.to(device, non_blocking=True),
        "max_seqlen_in_batch_kv": cast(int, max_seqlen_in_batch),
        "valid_token_indices_kv": valid_token_indices.to(device, non_blocking=True),
    }

class AsymmetricAttention(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        update_y: bool = True,
        out_bias: bool = True,
        attention_mode: str = "flash",
        softmax_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.attention_mode = attention_mode
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(f"dim_x={dim_x} should be divisible by num_heads={num_heads}")

        # Input layers.
        self.qkv_bias = qkv_bias
        self.qkv_x = nn.Linear(dim_x, 3 * dim_x, bias=qkv_bias, device=device)
        # Project text features to match visual features (dim_y -> dim_x)
        self.qkv_y = nn.Linear(dim_y, 3 * dim_x, bias=qkv_bias, device=device)

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = RMSNorm(self.head_dim, device=device)
        self.k_norm_x = RMSNorm(self.head_dim, device=device)
        self.q_norm_y = RMSNorm(self.head_dim, device=device)
        self.k_norm_y = RMSNorm(self.head_dim, device=device)

        # Output layers. y features go back down from dim_x -> dim_y.
    
        self.proj_x = nn.Linear(dim_x, dim_x, bias=out_bias, device=device)
        self.proj_y = nn.Linear(dim_x, dim_y, bias=out_bias, device=device) if update_y else nn.Identity()

    def run_qkv_y(self, y):
        local_heads = self.num_heads
        qkv_y = self.qkv_y(y)  # (B, L, 3 * dim)
        qkv_y = qkv_y.view(qkv_y.size(0), qkv_y.size(1), 3, local_heads, self.head_dim)
        q_y, k_y, v_y = qkv_y.unbind(2)

        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)
        return q_y, k_y, v_y

    def prepare_qkv(
        self,
        x: torch.Tensor,  # (B, M, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,
        scale_y: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        valid_token_indices: torch.Tensor,
        max_seqlen_in_batch: int,
    ):
        # Process visual features
        x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size
        qkv_x = self.qkv_x(x)  # (B, M, 3 * dim_x)
        assert qkv_x.dtype == torch.bfloat16

        B, M, _ = qkv_x.size()
        qkv_x = qkv_x.view(B, M, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
        
        # Split qkv_x into q, k, v
        q_x, k_x, v_x = qkv_x.unbind(0)  # (B, N, local_h, head_dim)
        q_x = self.q_norm_x(q_x)
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)

        # Concatenate streams
        B, N, num_heads, head_dim = q_x.size()
        D = num_heads * head_dim

        # Process text features
        if B == 1:
            text_seqlen = max_seqlen_in_batch - N
            if text_seqlen > 0:
                y = y[:, :text_seqlen]  # Remove padding tokens.
                y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
                q_y, k_y, v_y = self.run_qkv_y(y)  # (B, L, local_heads, head_dim)

                q = torch.cat([q_x, q_y], dim=1)
                k = torch.cat([k_x, k_y], dim=1)
                v = torch.cat([v_x, v_y], dim=1)
            else:
                q, k, v = q_x, k_x, v_x
        else:
            y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
            q_y, k_y, v_y = self.run_qkv_y(y)  # (B, L, local_heads, head_dim)

            indices = valid_token_indices[:, None].expand(-1, D)
            q = torch.cat([q_x, q_y], dim=1).view(-1, D).gather(0, indices)  # (total, D)
            k = torch.cat([k_x, k_y], dim=1).view(-1, D).gather(0, indices)  # (total, D)
            v = torch.cat([v_x, v_y], dim=1).view(-1, D).gather(0, indices)  # (total, D)

        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_heads, head_dim)
        v = v.view(-1, num_heads, head_dim)
        return q, k, v

    def sdpa_attention(self, q, k, v):
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return out

    def run_attention(
        self,
        q: torch.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        k: torch.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        v: torch.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        *,
        B: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen_in_batch: Optional[int] = None,
    ):
        local_heads = self.num_heads
        local_dim = local_heads * self.head_dim
        # Check shapes
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        total = q.size(0)
        assert k.size(0) == total and v.size(0) == total
        assert B == 1, \
            f"Non-flash attention mode {self.attention_mode} only supports batch size 1, got {B}"
        q = rearrange(q, "(b s) h d -> b h s d", b=B)
        k = rearrange(k, "(b s) h d -> b h s d", b=B)
        v = rearrange(v, "(b s) h d -> b h s d", b=B)
        out = self.sdpa_attention(q, k, v) 
        out = rearrange(out, "b h s d -> (b s) (h d)")
        return out

    def post_attention(
        self,
        out: torch.Tensor,
        B: int,
        M: int,
        L: int,
        dtype: torch.dtype,
        valid_token_indices: torch.Tensor,
    ):
        """
        Args:
            out: (total <= B * (N + L), local_dim)
            valid_token_indices: (total <= B * (N + L),)
            B: Batch size
            M: Number of visual tokens per context parallel rank
            L: Number of text tokens
            dtype: Data type of the input and output tensors

        Returns:
            x: (B, N, dim_x) tensor of visual tokens where N = M * cp_size
            y: (B, L, dim_y) tensor of text token features
        """
        local_heads = self.num_heads
        local_dim = local_heads * self.head_dim
        N = M

        # Split sequence into visual and text tokens, adding back padding.
        if B == 1:
            out = out.view(B, -1, local_dim)
            if out.size(1) > N:
                x, y = torch.tensor_split(out, (N,), dim=1)  # (B, N, local_dim), (B, <= L, local_dim)
                y = F.pad(y, (0, 0, 0, L - y.size(1)))  # (B, L, local_dim)
            else:
                # Empty prompt.
                x, y = out, out.new_zeros(B, L, local_dim)
        else:
            x, y = pad_and_split_xy(out, valid_token_indices, B, N, L, dtype)
        assert x.size() == (B, M, local_dim)
        assert y.size() == (B, L, local_dim)

        # Communicate across context parallel ranks.
        x = x.view(B, N, local_heads, self.head_dim)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))  # (B, M, dim_x = num_heads * head_dim)
        x = self.proj_x(x)
        y = self.proj_y(y)
        return x, y

    def forward(
        self,
        x: torch.Tensor,  # (B, M, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,  # (B, dim_x), modulation for pre-RMSNorm.
        scale_y: torch.Tensor,  # (B, dim_y), modulation for pre-RMSNorm.
        packed_indices: Dict[str, torch.Tensor] = None,
        **rope_rotation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of asymmetric multi-modal attention.

        Args:
            x: (B, M, dim_x) tensor of visual tokens
            y: (B, L, dim_y) tensor of text token features
            packed_indices: Dict with keys for Flash Attention
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, M, dim_x) tensor of visual tokens after multi-modal attention
            y: (B, L, dim_y) tensor of text token features after multi-modal attention
        """
        B, L, _ = y.shape
        _, M, _ = x.shape

        # Predict a packed QKV tensor from visual and text features.
        q, k, v = self.prepare_qkv(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_rotation.get("rope_cos"),
            rope_sin=rope_rotation.get("rope_sin"),
            valid_token_indices=packed_indices["valid_token_indices_kv"],
            max_seqlen_in_batch=packed_indices["max_seqlen_in_batch_kv"],
        )  # (total <= B * (N + L), 3, local_heads, head_dim)

        # Self-attention is expensive, so don't checkpoint it.
        out = self.run_attention(
            q, k, v, B=B,
            cu_seqlens=packed_indices["cu_seqlens_kv"],
            max_seqlen_in_batch=packed_indices["max_seqlen_in_batch_kv"],
        )

        x, y = self.post_attention(
            out,
            B=B, M=M, L=L,
            dtype=v.dtype,
            valid_token_indices=packed_indices["valid_token_indices_kv"],
        )
        return x, y


class AsymmetricJointBlock(nn.Module):
    def __init__(
        self,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens.
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens.
        update_y: bool = True,  # Whether to update text tokens in this block.
        device: Optional[torch.device] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mod_x = nn.Linear(hidden_size_x, 4 * hidden_size_x, device=device)
        if self.update_y:
            self.mod_y = nn.Linear(hidden_size_x, 4 * hidden_size_y, device=device)
        else:
            self.mod_y = nn.Linear(hidden_size_x, hidden_size_y, device=device)

        # Self-attention:
        self.attn = AsymmetricAttention(
            hidden_size_x,
            hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            device=device,
            **block_kwargs,
        )

        # MLP.
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=256,
            ffn_dim_multiplier=None,
            device=device,
        )

        # MLP for text not needed in last block.
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=256,
                ffn_dim_multiplier=None,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        rope_cos,
        rope_sin,
        packed_indices,
    ):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        N = x.size(1)

        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)

        mod_y = self.mod_y(c)
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y

        # Self-attention block.
        x_attn, y_attn = self.attn(
            x,
            y,
            scale_x=scale_msa_x,
            scale_y=scale_msa_y,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            packed_indices=packed_indices,
        )

        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)

        # MLP block.
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)

        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)  # Sandwich norm
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)  # Sandwich norm
        return y

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, device=device)
        self.mod = nn.Linear(hidden_size, 2 * hidden_size, device=device)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, device=device)

    def forward(self, x, c):
        c = F.silu(c)
        shift, scale = self.mod(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


@maybe_allow_in_graph
class MochiTransformer3DModel(ModelMixin, ConfigMixin):
    """
    Diffusion model with a Transformer backbone.

    Ingests text embeddings instead of a label.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size=2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        pooled_projection_dim: int = 1536,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 4096,
        time_embed_dim: int = 256,
        activation_fn: str = "swiglu",
        max_sequence_length: int = 256,
    ):
        super().__init__()

        hidden_size_x = num_attention_heads * attention_head_dim
        hidden_size_y = pooled_projection_dim
        depth = num_layers
        num_heads = num_attention_heads
        mlp_ratio_x=4.0
        mlp_ratio_y=4.0
        t5_feat_dim = text_embed_dim
        t5_token_length = max_sequence_length
        patch_embed_bias = True
        timestep_mlp_bias = True
        timestep_scale=1000.0
        use_extended_posenc = False
        rope_theta=10000.0
        device = torch.device("cpu") 

        block_kwargs = {
            "qk_norm":True,
            "qkv_bias":False,
            "out_bias":True,
            "attention_mode": "sdpa"
        }

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.head_dim = hidden_size_x // num_heads  # Head dimension and count is determined by visual.
        self.use_extended_posenc = use_extended_posenc
        self.t5_token_length = t5_token_length
        self.t5_feat_dim = t5_feat_dim
        self.rope_theta = rope_theta  # Scaling factor for frequency computation for temporal RoPE.

        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size_x,
            bias=patch_embed_bias,
            device=device,
        )
        # Conditionings
        # Timestep
        self.t_embedder = TimestepEmbedder(hidden_size_x, bias=timestep_mlp_bias, timestep_scale=timestep_scale)

        # Caption Pooling (T5)
        self.t5_y_embedder = AttentionPool(t5_feat_dim, num_heads=8, output_dim=hidden_size_x, device=device)

        # Dense Embedding Projection (T5)
        self.t5_yproj = nn.Linear(t5_feat_dim, hidden_size_y, bias=True, device=device)

        # Initialize pos_frequencies as an empty parameter.
        self.pos_frequencies = nn.Parameter(torch.empty(3, self.num_heads, self.head_dim // 2, device=device))

        # for depth 48:
        #  b =  0: AsymmetricJointBlock, update_y=True
        #  b =  1: AsymmetricJointBlock, update_y=True
        #  ...
        #  b = 46: AsymmetricJointBlock, update_y=True
        #  b = 47: AsymmetricJointBlock, update_y=False. No need to update text features.
        blocks = []
        for b in range(depth):
            # Joint multi-modal block
            update_y = b < depth - 1
            block = AsymmetricJointBlock(
                hidden_size_x,
                hidden_size_y,
                num_heads,
                mlp_ratio_x=mlp_ratio_x,
                mlp_ratio_y=mlp_ratio_y,
                update_y=update_y,
                device=device,
                **block_kwargs,
            )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(hidden_size_x, patch_size, self.out_channels, device=device)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C=12, T, H, W) tensor of visual tokens

        Returns:
            x: (B, C=3072, N) tensor of visual tokens with positional embedding.
        """
        return self.x_embedder(x)  # Convert BcTHW to BCN

    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ):
        """Prepare input and conditioning embeddings."""

        
        # Visual patch embeddings with positional encoding.
        T, H, W = x.shape[-3:]
        pH, pW = H // self.patch_size, W // self.patch_size
        x = self.embed_x(x)  # (B, N, D), where N = T * H * W / patch_size ** 2
        assert x.ndim == 3
        B = x.size(0)

        # Construct position array of size [N, 3].
        # pos[:, 0] is the frame index for each location,
        # pos[:, 1] is the row index for each location, and
        # pos[:, 2] is the column index for each location.
        N = T * pH * pW
        assert x.size(1) == N
        pos = create_position_matrix(T, pH=pH, pW=pW, device=x.device, dtype=torch.float32)  # (N, 3)
        rope_cos, rope_sin = compute_mixed_rotation(
            freqs=self.pos_frequencies, pos=pos
        )  # Each are (N, num_heads, dim // 2)

        # Global vector embedding for conditionings.
        c_t = self.t_embedder(1 - sigma)  # (B, D)

        
        # Pool T5 tokens using attention pooler
        # Note y_feat[1] contains T5 token features.
        assert (
            t5_feat.size(1) == self.t5_token_length
        ), f"Expected L={self.t5_token_length}, got {t5_feat.shape} for y_feat."
        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)  # (B, D)
        assert t5_y_pool.size(0) == B, f"Expected B={B}, got {t5_y_pool.shape} for t5_y_pool."

        c = c_t + t5_y_pool

        y_feat = self.t5_yproj(t5_feat)  # (B, L, t5_feat_dim) --> (B, L, D)

        return x, c, y_feat, rope_cos, rope_sin

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        return_dict: bool = True,
    ):
        """Forward pass of DiT.

        Args:
            x: (B, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
            sigma: (B,) tensor of noise standard deviations
            y_feat: List((B, L, y_feat_dim) tensor of caption token features. For SDXL text encoders: L=77, y_feat_dim=2048)
            y_mask: List((B, L) boolean tensor indicating which tokens are not padding)
            packed_indices: Dict with keys for Flash Attention. Result of compute_packed_indices.
        """
        sigma = timestep / 1000
        sigma = 1 - sigma
        sigma = sigma.to(hidden_states.dtype)
        x = hidden_states
        y_feat = [encoder_hidden_states] 
        y_mask = [encoder_attention_mask]
        B, _, T, H, W = x.shape
        packed_indices = compute_packed_indices(x.device, y_mask[0], T*H*W) #TODO: 是否会影响到梯度计算

        x, c, y_feat, rope_cos, rope_sin = self.prepare(x, sigma, y_feat[0], y_mask[0])

        N = x.size(1)
        M = N

        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x, y_feat = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    c,
                    y_feat,
                    rope_cos,
                    rope_sin,
                    packed_indices,
                    **ckpt_kwargs,
                )
            else:
                x, y_feat = block(
                    x,
                    c,
                    y_feat,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    packed_indices=packed_indices,
                )

        x = self.final_layer(x, c)  # (B, M, patch_size ** 2 * out_channels)

        patch = x.size(2)
        x = rearrange(x, "(G B) M P -> B (G M) P", G=1, P=patch)
        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )
        if not return_dict:
            return (x, )

        return Transformer2DModelOutput(sample=x)
