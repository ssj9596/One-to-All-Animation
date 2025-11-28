# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .attention_processor import Attention
from .normalization import FP32LayerNorm
from .attention import FeedForward
from .embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from opensora.controlnet_modules.controlnet import zero_module, MiniHunyuanEncoder
from diffusers.utils import is_torch_version
from .transformer_wan_refextractor_2d_controlnet_last import WanRotaryPosEmbed, WanTimeTextImageEmbedding, WanTransformerBlock
from .utils import ckpt_kwargs
import numpy as np
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class WanControlNet(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        conditioning_embedding_channels: int = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, inner_dim # 这里本来是added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(inner_dim, inner_dim)))

        # 复用miniencoder
        self.input_hint_block = MiniHunyuanEncoder(
            in_channels = 3, #一般肯定是3
            out_channels = conditioning_embedding_channels, #一般情况下，和self.out_channels一致，这里主要是考虑可能用不同的patch_embedding方案
            block_out_channels=(16, 16, 16, 16),
            norm_num_groups = 4,
            layers_per_block=1
        )

        # make sure conditioning_embedding_channels==in_channels // 8
        self.controlnet_x_embedder = torch.nn.Linear(conditioning_embedding_channels*4, inner_dim)
        self.gradient_checkpointing = False
        def _gradient_checkpointing_func(module, *args):
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *args,
                **ckpt_kwargs,
            )

        self._gradient_checkpointing_func_cuda = _gradient_checkpointing_func

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        image_latents = None,
        ref_block_samples = None,
        ref_rotary_emb = None,
        token_replace = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # =============== control part ===============
        controlnet_cond = self.input_hint_block(controlnet_cond)
        #pixel shuffle的方式进行patchify
        batch_size_c, channels_c, num_frames_c, height_c, width_c = controlnet_cond.shape
        height_c = height_c // p_h
        width_c = width_c // p_w
        num_frames_c = num_frames_c // p_t
        controlnet_cond = controlnet_cond.reshape(
            batch_size, channels_c, num_frames_c, p_t, height_c, p_h, width_c, p_w
        )
        controlnet_cond = controlnet_cond.permute(0, 2, 4, 6, 1, 3, 5, 7)
        controlnet_cond = controlnet_cond.reshape(batch_size, num_frames_c * height_c * width_c, -1)


        # ============== add control here ===============
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        if token_replace: 
            t_token_replace = torch.zeros_like(timestep)
            # 5 frames -> VAE -> 2 frames
            replace_token_num = 2 * post_patch_height * post_patch_width
        else:
            t_token_replace = None
            replace_token_num = None


        temb, timestep_proj, temb_token_replace, timestep_proj_token_replace, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, t_token_replace
        )

        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if timestep_proj_token_replace is not None:
            timestep_proj_token_replace = timestep_proj_token_replace.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        # ref_block_samples >= controlnet blocks
        if ref_block_samples is not None:
            indices = np.linspace(0, len(ref_block_samples) - 1, len(self.blocks)).round().astype(int)

        # 4. Transformer blocks
        block_samples = ()
        for index_block, block in enumerate(self.blocks):
            ref_latent = None
            if ref_block_samples is not None:

                ref_latent = ref_block_samples[indices[index_block]]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func_cuda(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, ref_latent, ref_rotary_emb, timestep_proj_token_replace, replace_token_num
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, ref_latent, ref_rotary_emb, timestep_proj_token_replace, replace_token_num)
            
            block_samples = block_samples + (hidden_states, )


        controlnet_block_samples = ()
        assert len(block_samples) == len(self.controlnet_blocks), \
                f"Length mismatch: block_samples has length {len(block_samples)}, controlnet_blocks has length {len(self.controlnet_blocks)}"
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            controlnet_block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (controlnet_block_sample, )
        
        return controlnet_block_samples
        
