# copy from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnets/controlnet_flux.py#L41
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.utils import is_torch_version
from diffusers.loaders import  PeftAdapterMixin
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor2_0
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero, AdaLayerNormZeroSingle
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.controlnets.controlnet import Tuple, zero_module
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXDownBlock3D, CogVideoXCausalConv3d
from .embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps, get_2d_sincos_pos_embed, get_3d_rotary_pos_embed
from .transformer_cogvideo import CogVideoXBlock

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width) #在720x480的grid上crop的左上角和右下角

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t = 1,
    attention_head_dim: int = 64,
    device = None,
    dtype = None,
    base_height: int = 480,
    base_width: int = 720,
):
    grid_height = height // patch_size
    grid_width = width // patch_size

    p = patch_size
    p_t = patch_size_t

    if p_t is None:
        # CogVideoX 1.0 I2V
        base_size_width = base_width // p
        base_size_height = base_height // p

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5 I2V
        base_size_width = base_width // p
        base_size_height = base_height // p
        base_num_frames = (num_frames + p_t - 1) // p_t

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )
    freqs_cos = freqs_cos.to(device=device, dtype=dtype)
    freqs_sin = freqs_sin.to(device=device, dtype=dtype)
    return freqs_cos, freqs_sin

class CogVideoXControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 64,
        vae_channels: int = 16, # 只需要和视频模态交互
        in_channels: int = 64, # contiditon的编码数量
        downscale_coef: int = 8,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        out_proj :bool = True,
        block_out_channels = (16, 32, 32, 64),
        layers_per_block = 3,
        norm_num_groups = 8,
        pad_mode = "first",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
        out_channels = vae_channels
        self.controlnet_encode = CogVideoXControlnetEncoder3D(
            in_channels = in_channels,
            out_channels = out_channels ,
            down_block_types = (
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ),
            block_out_channels = block_out_channels,
            layers_per_block = layers_per_block,
            act_fn = timestep_activation_fn,
            norm_eps = norm_eps,
            norm_num_groups = norm_num_groups,
            dropout = dropout,
            pad_mode = pad_mode,
            temporal_compression_ratio = temporal_compression_ratio,
        )
        self.vae_channels = vae_channels
        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=vae_channels,
            embed_dim=inner_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_projectors = None
        if out_proj is not None:
            self.out_projectors = zero_module(nn.ModuleList(
                [nn.Linear(inner_dim, inner_dim) for _ in range(num_layers)]
            ))
            
        self.gradient_checkpointing = False
        
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def compress_time(self, x, num_frames):
        x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
        batch_size, frames, channels, height, width = x.shape
        x = rearrange(x, 'b f c h w -> (b h w) c f')
        
        if x.shape[-1] % 2 == 1:
            x_first, x_rest = x[..., 0], x[..., 1:]
            if x_rest.shape[-1] > 0:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

            x = torch.cat([x_first[..., None], x_rest], dim=-1)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = rearrange(x, '(b h w) c f -> (b f) c h w', b=batch_size, h=height, w=width)
        return x
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_size, channels, num_frames, height, width = controlnet_cond.shape
        # 0. Controlnet encoder
        controlnet_cond = self.controlnet_encode(controlnet_cond)
        controlnet_cond = rearrange(controlnet_cond, 'b c f h w -> b f c h w')
        hidden_states = hidden_states[:, :, :self.vae_channels, :, :] + controlnet_cond
        image_rotary_emb = prepare_rotary_positional_embeddings(
                height=hidden_states.shape[-2], #cogvideox固定分辨率
                width=hidden_states.shape[-1],
                num_frames=hidden_states.shape[1],
                vae_scale_factor_spatial=8,
                patch_size=2,
                patch_size_t = self.config.patch_size_t if hasattr(self.config, "patch_size_t") else None,
                attention_head_dim=self.config.attention_head_dim,
                base_height=self.config.sample_height,
                base_width=self.config.sample_width,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)


        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        
        controlnet_hidden_states = ()
        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
                
            if self.out_projectors is not None:
                controlnet_hidden_states += (self.out_projectors[i](hidden_states),)
            else:
                controlnet_hidden_states += (hidden_states,)
            
        if not return_dict:
            return (controlnet_hidden_states,)
        return Transformer2DModelOutput(sample=controlnet_hidden_states)

class CogVideoXControlnetEncoder3D(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (16, 32, 32, 64),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
        self.down_blocks = nn.ModuleList([])

        # down blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if down_block_type == "CogVideoXDownBlock3D":
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    compress_time=compress_time,
                )
            else:
                raise ValueError("Invalid `down_block_type` encountered. Must be `CogVideoXDownBlock3D`")

            self.down_blocks.append(down_block)


        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = zero_module(CogVideoXCausalConv3d(
            block_out_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode
        ))

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `CogVideoXEncoder3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. Down
            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block),
                    hidden_states,
                    temb,
                    None,
                    conv_cache.get(conv_cache_key),
                )
        else:
            # 1. Down
            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = down_block(
                    hidden_states, temb, None, conv_cache.get(conv_cache_key)
                )

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)

        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states