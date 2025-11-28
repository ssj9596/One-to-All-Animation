from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import SpatialNorm

from opensora.vae_variants.hunyuan_vae_src.unet_causal_3d_blocks import (
    CausalConv3d,
    get_down_block3d,
)

class ControlNetCausalConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = CausalConv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class MiniHunyuanEncoder(nn.Module):
    '''
    a direct copy of hunyuan encoder
    '''
    def __init__(
        self,
        in_channels = 3,
        out_channels = 3,
        down_block_types = ['DownEncoderBlockCausal3D', 'DownEncoderBlockCausal3D', 'DownEncoderBlockCausal3D', 'DownEncoderBlockCausal3D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block = 2,
        norm_num_groups = 32,
        act_fn: str = "silu",
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = CausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(
                np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_downsample = bool(
                    i < num_spatial_downsample_layers)
                add_time_downsample = bool(i >= (
                    len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block)
            elif time_compression_ratio == 8:
                add_spatial_downsample = bool(
                    i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {time_compression_ratio}")

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2, ) if add_time_downsample else (1, )
            downsample_stride = tuple(
                downsample_stride_T + downsample_stride_HW)
            down_block = get_down_block3d(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(
                    add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)
        
        self.conv_out = zero_module(
            CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)
        )

    def forward(self, sample):
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions"
        sample = self.conv_in(sample)
        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.conv_out(sample)
        return sample


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class InflatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class ResnetBlockInflated(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        use_inflated_groupnorm=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        assert use_inflated_groupnorm != None
        if use_inflated_groupnorm:
            self.norm1 = InflatedGroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )
        else:
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )

        self.conv1 = InflatedConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )

            self.time_emb_proj = torch.nn.Linear(
                temb_channels, time_emb_proj_out_channels
            )
        else:
            self.time_emb_proj = None

        if use_inflated_groupnorm:
            self.norm2 = InflatedGroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )
        else:
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = (
            self.in_channels != self.out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

class DownEncoderBlockInflated(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        add_downsample: bool,
        downsample_stride: tuple = (1, 2, 2),
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        temb_channels: int = None,
    ):
        super().__init__()

        self.resnets = nn.ModuleList(
            [
                ResnetBlockInflated(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    groups=resnet_groups,
                    use_inflated_groupnorm=True,
                )
                for i in range(num_layers)
            ]
        )

        self.downsamplers = nn.ModuleList()
        if add_downsample:
            self.downsamplers.append(
                InflatedConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            self.down_stride = downsample_stride
        else:
            self.down_stride = (1, 1, 1)

    def forward(self, x, temb=None):
        for resnet in self.resnets:
            x = resnet(x, temb)

        for down in self.downsamplers:
            x = down(x)
        return x



class SFT(nn.Module): # 2D SFT
    def __init__(
        self,
        in_channels, 
        out_channels, 
        intermediate_channels=128,
        groups=32,
        eps=1e-6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.norm = InflatedGroupNorm(groups, out_channels, eps, affine=True)

        self.mlp_shared = nn.Sequential(
            InflatedConv3d(
                in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.SiLU(),
        )
        self.mlp_gamma = InflatedConv3d(
            intermediate_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.mlp_beta = InflatedConv3d(
            intermediate_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    @staticmethod
    def _match_time(cond, target_t):
        b, c, tc, h, w = cond.shape
        if tc == target_t:
            return cond
        if tc == 1:
            return cond.repeat(1, 1, target_t, 1, 1)

    def forward(self, hidden_state, condition):
        """
        hidden_state : (B, Cout,  T,  H,  W)
        condition    : (B, Cin, 1, H, W)
        """
        b, _, t, h, w = hidden_state.shape
        hidden_state = self.norm(hidden_state) #2D SFT 2D Norm

        condition = self._match_time(condition, t)

        actv = self.mlp_shared(condition)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return hidden_state * (1 + gamma) + beta


class MiniEncoder2D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: list = (
            "DownEncoderBlockInflated",
            "DownEncoderBlockInflated",
            "DownEncoderBlockInflated",
            "DownEncoderBlockInflated",
        ),
        block_out_channels: list = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()

        # -------------------------------------------------------------------
        # 输入卷积
        # -------------------------------------------------------------------
        self.conv_in = InflatedConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        num_spatial_down_layers = int(np.log2(spatial_compression_ratio))

        for i, block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            # is_final_block = i == len(block_out_channels) - 1

            add_spatial_downsample = bool(i < num_spatial_down_layers)

            downsample_stride = (1, 2, 2) if add_spatial_downsample else (1, 1, 1)

            down_block = DownEncoderBlockInflated(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=add_spatial_downsample,
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.conv_out = zero_module(
            InflatedConv3d(output_channel, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):

        if x.ndim == 4:
            x = x.unsqueeze(2)  # (B,C,1,H,W)

        assert x.ndim == 5 and x.shape[2] == 1, "时间维度必须为 1"
        feats = []

        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)
            feats.append(x) 

        x = self.conv_out(x)
        return x, feats


class Driven_Ref_PoseEncoder(nn.Module):
    def __init__(
        self,
        in_channels = 3,
        out_channels = 3,
        down_block_types = ['DownEncoderBlockCausal3D', 'DownEncoderBlockCausal3D', 'DownEncoderBlockCausal3D', 'DownEncoderBlockCausal3D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block = 2,
        norm_num_groups = 32,
        act_fn: str = "silu",
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = CausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(
                np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_downsample = bool(
                    i < num_spatial_downsample_layers)
                add_time_downsample = bool(i >= (
                    len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block)
            elif time_compression_ratio == 8:
                add_spatial_downsample = bool(
                    i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {time_compression_ratio}")

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2, ) if add_time_downsample else (1, )
            downsample_stride = tuple(
                downsample_stride_T + downsample_stride_HW)
            down_block = get_down_block3d(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(
                    add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)
        
        self.conv_out = zero_module(
            CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)
        )

        self.ref_pose_encoder =  MiniEncoder2D(
            in_channels = in_channels,
            out_channels = out_channels, 
            block_out_channels = block_out_channels,
            norm_num_groups = norm_num_groups,
            layers_per_block = layers_per_block,
            spatial_compression_ratio = spatial_compression_ratio,
        )
        self.sft_layers = nn.ModuleList()
        for i, ch in enumerate(block_out_channels):
            if i == 0:                         # 0 层 (H/2,W/2) 不做 SFT
                self.sft_layers.append(None)
            else:                              # H/4、H/8、H/16 做 SFT
                self.sft_layers.append(
                    SFT(
                        in_channels=ch,
                        out_channels=ch,
                        intermediate_channels=max(8, ch // 2),
                        groups=norm_num_groups,
                    )
                )

    def forward(self, driven_pose, ref_pose):
        # driven_pose b c t h w
        # ref_pose b c 1 h w
        ref_pose_cond, ref_feats = self.ref_pose_encoder(ref_pose) 

        x = self.conv_in(driven_pose)
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x)

            if self.sft_layers[i] is not None:
                cond_feat = ref_feats[i]
                x = self.sft_layers[i](x, cond_feat)

        driven_pose_cond = self.conv_out(x)
        return driven_pose_cond, ref_pose_cond


        





