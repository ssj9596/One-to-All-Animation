from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner
from opensora.controlnet_modules.controlnet import zero_module, MiniHunyuanEncoder

class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        t: Optional[torch.Tensor] = None,
        image_embed = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        image_embed = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)


class HYVideoDiffusionTransformerEnd2End(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        text_states_dim,
        text_states_dim_2,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        rope_theta = 256,
        embedded_guidance_scale = 6.0, #用默认值
        conditioning_embedding_channels: int = None,
    ):  
        device = None
        dtype = None
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.embedded_guidance_scale = embedded_guidance_scale

        self.use_ipadapter = False

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        self.gradient_checkpointing = False

        # pose_control_part
        if conditioning_embedding_channels:
            self.input_hint_block = MiniHunyuanEncoder(
                    in_channels = 3, #一般肯定是3
                    out_channels = conditioning_embedding_channels, #一般情况下，和self.out_channels一致，这里主要是考虑可能用不同的patch_embedding方案
                    block_out_channels=(16, 16, 16, 16),
                    norm_num_groups = 4,
                    layers_per_block=1
                )
            self.controlnet_x_embedder = torch.nn.Linear(self.in_channels, self.hidden_size)
        else:
            self.input_hint_block = None
            self.controlnet_x_embedder = zero_module(torch.nn.Linear(self.in_channels, self.hidden_size))

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def set_up_ipadapter(self, ipadapter_config, weight_dtype):
        # only set once
        if self.use_ipadapter is True:
            return 

        if isinstance(ipadapter_config, str):
            import json
            with open(ipadapter_config, "r") as f:
                ipadapter_config = json.load(f)
        if not ipadapter_config["enable_ipadapter"]:
            # ("[IP-Adapter] IP-Adapter not enabled in config. Skip.")
            return
            
        ipadapter_double_blocks_depth = ipadapter_config["ipadapter_double_blocks_depth"]
        ipadapter_single_blocks_depth = ipadapter_config["ipadapter_single_blocks_depth"]
        total_adapters_needed = ipadapter_double_blocks_depth + ipadapter_single_blocks_depth
        if total_adapters_needed <= 0:
            return

        from .ipadapter import MLPProjModelAdvanced, HYVideoAttnProcessor2_0
        from .ipablocks import MMDoubleStreamBlockIPAdapter, MMSingleStreamBlockIPAdapter
        # MLP (only one)
        self.ipadapter_proj = MLPProjModelAdvanced(
                cross_attention_dim=ipadapter_config["mlp_out_dim"],
                id_embeddings_dim=ipadapter_config["mlp_in_dim"],
                num_tokens=ipadapter_config["mlp_num_tokens"],
            ).to(dtype=weight_dtype)
        

        # ip_adapters (a lot of cross attention)
        interval_double = max(1, int(np.ceil(len(self.double_blocks) / ipadapter_double_blocks_depth
                                                            ))) if ipadapter_double_blocks_depth > 0 else 999
        interval_single = max(1, int(np.ceil(len(self.single_blocks) / ipadapter_single_blocks_depth
                                                            ))) if ipadapter_single_blocks_depth > 0 else 999
        adapter_idx = 0

        # ----  double_blocks ----
        new_double_blocks = []
        for i, block in enumerate(self.double_blocks):
            if ipadapter_double_blocks_depth > 0 and i % interval_double == 0:
                
                wrapped = MMDoubleStreamBlockIPAdapter(
                    original_block=block,
                    ipadapter_config = ipadapter_config,
                ).to(dtype=weight_dtype)
                new_double_blocks.append(wrapped)

                adapter_idx += 1

            else:
                new_double_blocks.append(block)
        self.double_blocks = nn.ModuleList(new_double_blocks)

        assert adapter_idx == ipadapter_double_blocks_depth, (
            f"IP-Adapter for double_blocks mismatch! "
            f"adapter_idx={adapter_idx}, ipadapter_double_blocks_depth={ipadapter_double_blocks_depth}"
        )
        
        # ---- single_blocks ----
        new_single_blocks = []
        for i, block in enumerate(self.single_blocks):
            if ipadapter_single_blocks_depth > 0 and i % interval_single == 0:
                
                wrapped_sblock = MMSingleStreamBlockIPAdapter(
                    original_block=block,
                    ipadapter_config = ipadapter_config,
                ).to(dtype=weight_dtype)
                new_single_blocks.append(wrapped_sblock)
                adapter_idx += 1
            else:
                new_single_blocks.append(block)
        self.single_blocks = nn.ModuleList(new_single_blocks)


        assert adapter_idx == total_adapters_needed, (
            f"IP-Adapter for single_blocks mismatch! after double we used {ipadapter_double_blocks_depth}, "
            f"but total needed is {total_adapters_needed}, "
            f"currently adapter_idx={adapter_idx}"
        )
        self.use_ipadapter = True

    
    def enable_ipadapter_params_grad(self):
        if not self.use_ipadapter:
            print("[IP-Adapter] Not use ipadapter, skip enabling grad.")
            return
        # mlp
        if self.ipadapter_proj is not None:
            for p in self.ipadapter_proj.parameters():
                p.requires_grad = True
        # ip
        for block in self.double_blocks:
            if hasattr(block, 'ip_adapter') and block.ip_adapter is not None:
                for p in block.ip_adapter.parameters():
                    p.requires_grad = True
        for block in self.single_blocks:
            if hasattr(block, 'ip_adapter') and block.ip_adapter is not None:
                for p in block.ip_adapter.parameters():
                    p.requires_grad = True

        print("[IP-Adapter] All IP-Adapter params set requires_grad=True.")
    

    def get_ipadapter_params_count(self) -> float:
        if not self.use_ipadapter:
            print("[IP-Adapter] Not use ipadapter, skip enabling grad.")
            return
        # mlp
        if self.ipadapter_proj is not None:
            counts["mlp"] = sum(p.numel() for p in self.ipadapter_proj.parameters())
        # ip
        for block in self.double_blocks:
            if hasattr(block, 'ip_adapter') and block.ip_adapter is not None:
                counts["ip_adapters"] += sum(p.numel() for p in block.ip_adapter.parameters())
        for block in self.single_blocks:
            if hasattr(block, 'ip_adapter') and block.ip_adapter is not None:
                counts["ip_adapters"] += sum(p.numel() for p in block.ip_adapter.parameters())

        counts["total"] = counts["mlp"] + counts["ip_adapters"]
        return counts
        
    def get_controlnet_and_ipadapter_trainable_params_count(self) -> dict:
        controlnet_params = 0.0
        ipadapter_params = 0.0

        # Count trainable ControlNet parameters
        if self.input_hint_block is not None:
            controlnet_params += sum(p.numel() for p in self.input_hint_block.parameters() if p.requires_grad)
        if self.controlnet_x_embedder is not None:
            controlnet_params += sum(p.numel() for p in self.controlnet_x_embedder.parameters() if p.requires_grad)
        # Count trainable IP-Adapter parameters
        if self.use_ipadapter:
            if self.ipadapter_proj is not None:
                ipadapter_params = sum(p.numel() for p in self.ipadapter_proj.parameters() if p.requires_grad)

            # Iterate through the double_blocks and single_blocks to count trainable IP-Adapter parameters
            for block in self.double_blocks:
                if hasattr(block, 'ip_adapter') and block.ip_adapter is not None:
                    ipadapter_params += sum(p.numel() for p in block.ip_adapter.parameters() if p.requires_grad)

            for block in self.single_blocks:
                if hasattr(block, 'ip_adapter') and block.ip_adapter is not None:
                    ipadapter_params += sum(p.numel() for p in block.ip_adapter.parameters() if p.requires_grad)
        return {
            "controlnet_params": controlnet_params,
            "ipadapter_params": ipadapter_params,
        }


    def forward(self, 
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        embed_cfg_scale,
        controlnet_cond,
        conditioning_scale = 1.0,
        return_dict = True,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        controlnet_blocks_repeat = False,
        image_emb = None, # ipadapter image_emb
        image_latents = None, # image_latents for controlnet
        **kwargs #用于后续传入guidance
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if self.use_ipadapter:
            if image_emb is None:
                ValueError(
                    f"image_emb is None, no input for ip-adapter !!!!"
                )
            image_emb = self.ipadapter_proj(image_emb)


        lora_scale = 1.0 #TODO: 作为参数传入

        x = hidden_states
        t = timestep
        text_states, text_states_2 = encoder_hidden_states
        text_mask, test_mask_2 = encoder_attention_mask

        embedded_guidance_scale = embed_cfg_scale#TODO： maybe problem
        guidance = (
            torch.tensor(
                embedded_guidance_scale,
                dtype=torch.float32,
                device=hidden_states.device,
            ).to(hidden_states.dtype)
            * 1000.0
            if embedded_guidance_scale is not None
            else None
        )

        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(ot, oh, ow)
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        if conditioning_scale > 0 and self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)
            img = img + controlnet_cond
        # Embed image and text.
        img = self.img_in(img)

        # pose control part

            # batch_size, channels, t_pw, height_pw, width_pw = controlnet_cond.shape
            # patch_size_t, patch_size_h, patch_size_w = self.config.patch_size
            # height = height_pw // patch_size_h
            # width = width_pw // patch_size_w
            # t_ = t_pw // patch_size_t
            # controlnet_cond = controlnet_cond.reshape(
            #     batch_size, channels, t_, patch_size_t, height, patch_size_h, width, patch_size_w
            # )
            # controlnet_cond = controlnet_cond.permute(0, 2, 4, 6, 1, 3, 5, 7)
            # controlnet_cond = controlnet_cond.reshape(batch_size, height * width * t_, -1)
            # # add control here
            # img = img + self.controlnet_x_embedder(controlnet_cond) * conditioning_scale

        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        
        # --------------------- Pass through DiT blocks ------------------------
        for index_block, block in enumerate(self.double_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                img, txt = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    t,
                    image_emb,
                    **ckpt_kwargs,
                )
            else:
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    t,
                    image_emb,
                ]

                img, txt = block(*double_block_args)
            
            if controlnet_block_samples is not None:
                interval_control = len(self.double_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    img = (
                        img + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    img = img + controlnet_block_samples[index_block // interval_control]

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        t,
                        image_emb,
                        **ckpt_kwargs,
                    )
                else:

                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        t,
                        image_emb,
                    ]

                    x = block(*single_block_args)
                
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    x[:, :img.shape[1], ...] = (
                        x[:, :img.shape[1], ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        output = img
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884 
        latents_size = [video_length, height, width]

        if isinstance(self.config.patch_size, int):
            assert all(s % self.config.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.config.patch_size for s in latents_size]
        elif isinstance(self.config.patch_size, list):
            assert all(
                s % self.config.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.config.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.config.hidden_size // self.config.heads_num
        rope_dim_list = self.config.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.config.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
