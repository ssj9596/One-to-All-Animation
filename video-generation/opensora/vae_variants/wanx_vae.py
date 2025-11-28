from .wrapper import VAEWrapper
import torch

from diffusers.models import AutoencoderKLWan

class WanxVAEWrapper(VAEWrapper):
    def __init__(self, vae):
        self.vae = vae
        # self.vae.enable_tiling()
        self.vae.requires_grad_(False)
        self.vae.eval()
        
    def encode(self, x):
        x = self.vae.encode(x).latent_dist.mode()  # same setting with pipeline
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(x.device, x.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            x.device, x.dtype
        )
        x = (x - latents_mean) * latents_std
        return x

    def decode(self, latents):

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        return self.vae.decode(latents, return_dict=False)[0]

def get_wanx_vae_wrapper(model_path, weight_dtype):
    vae = AutoencoderKLWan.from_pretrained(model_path, torch_dtype=weight_dtype)
    return WanxVAEWrapper(vae)