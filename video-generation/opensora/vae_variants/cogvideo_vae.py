from .wrapper import VAEWrapper

from diffusers.models import AutoencoderKLCogVideoX

class CogVideoVAEWrapper(VAEWrapper):
    def __init__(self, vae):
        self.vae = vae
        self.vae.enable_tiling()
        self.vae.requires_grad_(False)
        self.vae.eval()

    def encode(self, x):
        x = self.vae.encode(x).latent_dist.sample()  #* self.vae.config.scaling_factor 
        if self.vae.config.invert_scale_latents and x.shape[2] == 1:
            x = 1 / self.vae.config.scaling_factor * x
        else:
            x = self.vae.config.scaling_factor * x
        return x

    def decode(self, latents):
        latents = 1 / self.vae.scaling_factor * latents
        return self.vae.decode(latents).sample

def get_cogvideo_vae_wrapper(model_path, weight_dtype):
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=weight_dtype)
    return CogVideoVAEWrapper(vae)