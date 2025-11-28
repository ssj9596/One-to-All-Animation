import torch
import torch.nn as nn

from opensora.video_vae.modeling_causal_vae import CausalVideoVAE
from .wrapper import VAEWrapper

class PyradmidFlowVAEWrapper(VAEWrapper):
    def __init__(self, vae):
        self.vae = vae
        self.vae.enable_tiling()
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_shift_factor = -0.04
        self.vae_scale_factor = 1 / 1.8726
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986

    def encode(self, x):
        #TODO: check这里的temporal chunk逻辑
        #TODO: 对齐一下是否sample，以及其他逻辑
        latents = self.vae.encode(x, temporal_chunk=True).latent_dist.sample()
        if latents.shape[2] == 1:
            latents = (latents - self.vae_shift_factor) / self.vae_scale_factor
        else:
            latents[:, :, :1] =  (latents[:, :, :1] - self.vae_shift_factor) * self.vae_scale_factor
            latents[:, :, 1:] =  (latents[:, :, 1:] - self.vae_video_shift_factor) * self.vae_video_scale_factor
        return latents

    def decode(self, latents, save_memory=True):
        # TODO: 输出形式待定
        if latents.shape[2] == 1:
            latents = (latents / self.vae_scale_factor) + self.vae_shift_factor
        else:
            latents[:, :, :1] = (latents[:, :, :1] / self.vae_scale_factor) + self.vae_shift_factor
            latents[:, :, 1:] = (latents[:, :, 1:] / self.vae_video_scale_factor) + self.vae_video_shift_factor
        if save_memory:
            # reducing the tile size and temporal chunk window size
            image = self.vae.decode(latents, temporal_chunk=True, window_size=1, tile_sample_min_size=256).sample
        else:
            image = self.vae.decode(latents, temporal_chunk=True, window_size=2, tile_sample_min_size=512).sample
        return image

def get_pyramid_flow_vae_wrapper(model_path, weight_dtype):
    vae = CausalVideoVAE.from_pretrained(model_path, torch_dtype=weight_dtype, interpolate=False).eval()
    return PyradmidFlowVAEWrapper(vae)