'''
calculate diffusion loss according to different diffusion formula
'''
import torch
from diffusers import DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from opensora.utils.parallel_states import initialize_sequence_parallel_state, \
    destroy_sequence_parallel_group, get_sequence_parallel_state, set_sequence_parallel_state
from opensora.utils.communications import prepare_parallel_data, broadcast
from diffusers.training_utils import compute_snr
from diffusers.utils.torch_utils import randn_tensor
from torch.nn import functional as F
import numpy as np

from opensora.train.edm import EDMPrecond, EDMLoss
from opensora.sample.pyramid_flow_matching_scheduler import AutoRegressivePyramidFlowMatchEulerDiscreteScheduler
from functools import partial
from copy import deepcopy
from einops import rearrange
import random


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u
    
try:
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
except:
    pass

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device = None,
    dtype = None,
    base_height: int = 480,
    base_width: int = 720,
):
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )
    freqs_cos = freqs_cos.to(device=device, dtype=dtype)
    freqs_sin = freqs_sin.to(device=device, dtype=dtype)
    return freqs_cos, freqs_sin

class CustomDiffusion():
    def __init__(self):
        from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
        #TODO：优化了
        self.noise_scheduler = CogVideoXDPMScheduler.from_config("/CogVideoX-5b-I2V/scheduler/scheduler_config.json")

    def get_loss(self, model, model_input, model_kwargs, args, accelerator, **kwargs):
        model.train()
        if args.task == 'i2v':
            model_input, image_latents = model_input
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
        model_input = model_input.permute(0, 2, 1, 3, 4)
        dtype = model_input.dtype
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[1] #注意是B T C H W
        height, width = model_input.shape[3:]
        model_config = model.module.config if hasattr(model, 'module') else model.config
        patch_size_t = model_config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None: #TODO: not sure，对齐训练测试
            additional_frames = patch_size_t - current_step_frame % patch_size_t
            pading_frames = model_input[:, :additional_frames]
            model_input = torch.cat([pading_frames, model_input], dim=1)
            # image_latents = torch.cat([image_latents[:, :additional_frames], image_latents], dim=1)
        padding_shape = (model_input.shape[0], model_input.shape[1]-1, *model_input.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        noise = torch.randn_like(model_input)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        noisy_video_latents = self.noise_scheduler.add_noise(model_input, noise, timesteps)
        model_config = model.module.config if hasattr(model, 'module') else model.config
        ofs_emb = None if model_config.ofs_embed_dim is None else noisy_video_latents.new_full((1,), fill_value=2.0)
        if args.task == 't2v':
            noisy_model_input = noisy_video_latents
        elif args.task == 'i2v':
            noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
        model_output = model(
            hidden_states = noisy_model_input,
            encoder_hidden_states = model_kwargs['encoder_hidden_states'],
            encoder_attention_mask = model_kwargs['encoder_attention_mask'],
            timestep = timesteps,
            return_dict=False,
            ofs = ofs_emb,
        )[0]
        model_pred = self.noise_scheduler.get_velocity(model_output, noisy_video_latents, timesteps) #[:, additional_frames:]
        alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)
        target = model_input #[:, additional_frames:]
        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1)
        loss = loss.mean()
        return loss

    def get_loss_controlnet(self, condition, controlnet, model, model_input, model_kwargs, args, accelerator, **kwargs):
        controlnet.train()
        if args.task == 'i2v':
            model_input, image_latents = model_input
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
        model_input = model_input.permute(0, 2, 1, 3, 4)
        dtype = model_input.dtype
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[1] #注意是B T C H W
        height, width = model_input.shape[3:]
        model_config = model.module.config if hasattr(model, 'module') else model.config
        patch_size_t = model_config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None: #TODO: not sure，对齐训练测试
            additional_frames = patch_size_t - current_step_frame % patch_size_t
            pading_frames = model_input[:, :additional_frames]
            model_input = torch.cat([pading_frames, model_input], dim=1)
            # image_latents = torch.cat([image_latents[:, :additional_frames], image_latents], dim=1)
        padding_shape = (model_input.shape[0], model_input.shape[1]-1, *model_input.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        noise = torch.randn_like(model_input)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        noisy_video_latents = self.noise_scheduler.add_noise(model_input, noise, timesteps)
        model_config = model.module.config if hasattr(model, 'module') else model.config
        ofs_emb = None if model_config.ofs_embed_dim is None else noisy_video_latents.new_full((1,), fill_value=2.0)
        if args.task == 't2v':
            noisy_model_input = noisy_video_latents
        elif args.task == 'i2v':
            noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
        controlnet_states = controlnet(
            hidden_states=noisy_model_input,
            encoder_hidden_states=model_kwargs['encoder_hidden_states'],
            controlnet_cond=condition,
            timestep=timesteps,
            return_dict=False,
        )[0]
        if isinstance(controlnet_states, (tuple, list)):
            controlnet_states = [x.to(dtype=noisy_video_latents.dtype) for x in controlnet_states]
        else:
            controlnet_states = controlnet_states.to(dtype=noisy_video_latents.dtype)
        model_output = model(
            hidden_states = noisy_model_input,
            encoder_hidden_states = model_kwargs['encoder_hidden_states'],
            encoder_attention_mask = model_kwargs['encoder_attention_mask'],
            timestep = timesteps,
            return_dict=False,
            ofs = ofs_emb,
            controlnet_states = controlnet_states,
            controlnet_weights = 1.0
        )[0]
        model_pred = self.noise_scheduler.get_velocity(model_output, noisy_video_latents, timesteps) #[:, additional_frames:]
        alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)
        target = model_input #[:, additional_frames:]
        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1)
        loss = loss.mean()
        return loss
    
    @torch.no_grad()
    def get_valid_loss_controlnet(self, condition, controlnet, model, model_input, model_kwargs, args, **kwargs):
        if args.task == 'i2v':
            model_input, image_latents = model_input
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
        model_input = model_input.permute(0, 2, 1, 3, 4)
        dtype = model_input.dtype
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[1] #注意是B T C H W
        height, width = model_input.shape[3:]
        model_config = model.module.config if hasattr(model, 'module') else model.config
        patch_size_t = model_config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None: #TODO: not sure
            additional_frames = patch_size_t - current_step_frame % patch_size_t
            pading_frames = model_input[:, :additional_frames]
            model_input = torch.cat([pading_frames, model_input], dim=1)
            # image_latents = torch.cat([image_latents[:, :additional_frames], image_latents], dim=1)
        padding_shape = (model_input.shape[0], model_input.shape[1]-1, *model_input.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)
        
        evaluate_steps = kwargs['evaluate_steps'] # valid多少个timesteps的结果
        generator = torch.torch.Generator(device = model_input.device)
        generator.manual_seed(42)
        loss_list = []
        for step_i in range(evaluate_steps):
            generator = torch.torch.Generator(device = model_input.device)
            generator.manual_seed(42+step_i)
            timesteps = torch.ones((bsz,), dtype=int) * (step_i * self.noise_scheduler.config.num_train_timesteps // evaluate_steps)
            timesteps = timesteps.to(model_input.device)
            noise = torch.randn_like(model_input)
            noisy_video_latents = self.noise_scheduler.add_noise(model_input, noise, timesteps)
            ofs_emb = None if model_config.ofs_embed_dim is None else noisy_video_latents.new_full((1,), fill_value=2.0)
            if args.task == 't2v':
                noisy_model_input = noisy_video_latents
            elif args.task == 'i2v':
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
            controlnet_states = controlnet(
                hidden_states=noisy_model_input,
                encoder_hidden_states=model_kwargs['encoder_hidden_states'],
                controlnet_cond=condition,
                timestep=timesteps,
                return_dict=False,
                )[0]
            if isinstance(controlnet_states, (tuple, list)):
                controlnet_states = [x.to(dtype=model_kwargs["weight_dtype"]) for x in controlnet_states]
            else:
                controlnet_states = controlnet_states.to(dtype=model_kwargs["weight_dtype"])
            model_output = model(
                hidden_states = noisy_model_input,
                encoder_hidden_states = model_kwargs['encoder_hidden_states'],
                encoder_attention_mask = model_kwargs['encoder_attention_mask'],
                timestep = timesteps,
                return_dict=False,
                ofs = ofs_emb,
                controlnet_states = controlnet_states,
                controlnet_weights = args.controlnet_weights
            )[0]
            model_pred = self.noise_scheduler.get_velocity(model_output, noisy_video_latents, timesteps) #[:, additional_frames:]
            alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps]
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)
            target = model_input #[:, additional_frames:]
            loss = torch.mean((weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1)
            loss = loss.mean()
            loss_list.append(loss)
        loss_list = torch.stack(loss_list)
        return loss_list

    @torch.no_grad()
    def get_valid_loss(self, model, model_input, model_kwargs, args, **kwargs):
        if args.task == 'i2v':
            model_input, image_latents = model_input
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
        model_input = model_input.permute(0, 2, 1, 3, 4)
        dtype = model_input.dtype
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[1] #注意是B T C H W
        height, width = model_input.shape[3:]
        model_config = model.module.config if hasattr(model, 'module') else model.config
        patch_size_t = model_config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None: #TODO: not sure
            additional_frames = patch_size_t - current_step_frame % patch_size_t
            pading_frames = model_input[:, :additional_frames]
            model_input = torch.cat([pading_frames, model_input], dim=1)
            # image_latents = torch.cat([image_latents[:, :additional_frames], image_latents], dim=1)
        padding_shape = (model_input.shape[0], model_input.shape[1]-1, *model_input.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)
        
        evaluate_steps = kwargs['evaluate_steps'] # valid多少个timesteps的结果
        generator = torch.torch.Generator(device = model_input.device)
        generator.manual_seed(42)
        loss_list = []
        for step_i in range(evaluate_steps):
            generator = torch.torch.Generator(device = model_input.device)
            generator.manual_seed(42+step_i)
            timesteps = torch.ones((bsz,), dtype=int) * (step_i * self.noise_scheduler.config.num_train_timesteps // evaluate_steps)
            timesteps = timesteps.to(model_input.device)
            noise = torch.randn_like(model_input)
            noisy_video_latents = self.noise_scheduler.add_noise(model_input, noise, timesteps)
            ofs_emb = None if model_config.ofs_embed_dim is None else noisy_video_latents.new_full((1,), fill_value=2.0)
            if args.task == 't2v':
                noisy_model_input = noisy_video_latents
            elif args.task == 'i2v':
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
            model_output = model(
                hidden_states = noisy_model_input,
                encoder_hidden_states = model_kwargs['encoder_hidden_states'],
                encoder_attention_mask = model_kwargs['encoder_attention_mask'],
                timestep = timesteps,
                return_dict=False,
                ofs = ofs_emb,
            )[0]
            model_pred = self.noise_scheduler.get_velocity(model_output, noisy_video_latents, timesteps) #[:, additional_frames:]
            alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps]
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)
            target = model_input #[:, additional_frames:]
            loss = torch.mean((weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1)
            loss = loss.mean()
            loss_list.append(loss)
        loss_list = torch.stack(loss_list)
        return loss_list

class CogVideoXDDPM():
    def __init__(self, model_path, ori_height, ori_width, vae_scale_factor_spatial):
        from diffusers import CogVideoXDPMScheduler
        #TODO: create schedulers
        self.noise_scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder = 'scheduler')
        self.ori_height = ori_height
        self.ori_width = ori_width
        self.vae_scale_factor_spatial = vae_scale_factor_spatial

    def get_loss(self, model, model_input, model_kwargs, args, accelerator, **kwargs):
        model.train()
        if args.task == 'i2v':
            model_input, image_latents = model_input
        dtype = model_input.dtype
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[1]
        height, width = model_input.shape[3:]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        noisy_video_latents = self.noise_scheduler.add_noise(model_input, noise, timesteps)
        model_config = model.module.config if hasattr(model, 'module') else model.config
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=self.ori_height, #cogvideox固定分辨率
                width=self.ori_width,
                num_frames=current_step_frame,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                patch_size=model_config.patch_size,
                attention_head_dim=model_config.attention_head_dim,
                device=accelerator.device,
                dtype=dtype
            )
            if model_config.use_rotary_positional_embeddings
            else None
        )
        if args.task == 't2v':
            noisy_model_input = noisy_video_latents
        elif args.task == 'i2v':
            noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
        model_output = model(
            hidden_states = noisy_model_input,
            encoder_hidden_states = model_kwargs['encoder_hidden_states'],
            timestep = timesteps,
            image_rotary_emb = image_rotary_emb,
            return_dict=False,
        )[0]

        model_pred = self.noise_scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

        alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        target = model_input

        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1)
        loss = loss.mean()

        return loss

class DDPM():
    def __init__(self, use_flow=False):
        if not use_flow:
            self.noise_scheduler = DDPMScheduler()
        else:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler()
            self.noise_scheduler_copy = deepcopy(self.noise_scheduler)
            print(f"====> use rectified flow !!!")
            
        self.use_flow = use_flow
    
    # 前向运算计算loss
    def get_loss(self, model, model_input, model_kwargs, args, **kwargs):
        dtype = model_input.dtype
        noise = torch.randn_like(model_input)
        # if args.noise_offset:
        #     # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        #     noise += args.noise_offset * torch.randn((model_input.shape[0], model_input.shape[1], 1, 1, 1),
        #                                              device=model_input.device)
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[2]
        if args.reflow:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        else:
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)

        if current_step_frame != 1 and get_sequence_parallel_state():  # image do not need sp
            broadcast(timesteps)
        if self.use_flow:
            sigmas = self.noise_scheduler_copy.sigmas.to(device=model_input.device, dtype=dtype)
            schedule_timesteps = self.noise_scheduler_copy.timesteps.to(model_input.device)
            timesteps = timesteps.to(model_input.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < model_input.ndim:
                sigma = sigma.unsqueeze(-1)
            sigmas = sigma
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        else:
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        model_pred = model(
            noisy_model_input,
            timesteps,
            **model_kwargs
        )[0]
        if args.reflow:
            # model_pred = model_pred * (-sigmas) + noisy_model_input
            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        if args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=args.prediction_type)
        if hasattr(self.noise_scheduler.config, 'prediction_type'):
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            elif self.noise_scheduler.config.prediction_type == "sample":
                # We set the target to latents here, but the model_pred will return the noise sample prediction.
                target = model_input
                # We will have to subtract the noise residual from the prediction to get the target sample.
                model_pred = model_pred - noise
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        else: # reflow
            # target = model_input
            target = noise - model_input
        mask = model_kwargs.get('attention_mask', None)
        if torch.all(mask.bool()):
            mask = None
        assert mask is None
        b, c, t_shard, _, _ = model_pred.shape
        if mask is not None:
            # 实际上不知道这里是哪里，因此需要所有帧的attention mask是一样的，确保在时序上不会补帧即可
            mask = mask.unsqueeze(1).repeat(1, c, 1, 1, 1).float()[:, :, -t_shard:, :, :]  # b t h w -> b c t h w
            mask = mask.reshape(b, -1)
        if args.reflow:
            # Compute regular loss.
            loss = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
            if mask is not None:
                loss = (loss * mask).sum() / mask.sum()  # mean loss on unpad patches
            else:
                loss = loss.mean()
        else:
            if args.snr_gamma is None:
                # model_pred: b c t h w, attention_mask: b t h w
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.reshape(b, -1)
                if mask is not None:
                    loss = (loss * mask).sum() / mask.sum()  # mean loss on unpad patches
                else:
                    loss = loss.mean()
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(self.noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.reshape(b, -1)
                mse_loss_weights = mse_loss_weights.reshape(b, 1)
                if mask is not None:
                    loss = (loss * mask * mse_loss_weights).sum() / mask.sum()  # mean loss on unpad patches
                else:
                    loss = (loss * mse_loss_weights).mean()
        return loss

class FlowMatching():
    def __init__(self):
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=1.0,
            num_train_timesteps=1000,
            use_dynamic_shifting = False
        )
    # hunyuan
    def get_loss(self, model, model_input, model_kwargs, args, **kwargs):
        model.train()
        if args.task == 'i2v':
            model_input, image_latents = model_input
            padding_shape = (model_input.shape[0], model_input.shape[1], model_input.shape[2]-1, *model_input.shape[3:])
            latent_padding = image_latents.new_zeros(padding_shape)
            image_latents = torch.cat([image_latents, latent_padding], dim=2)
        dtype = model_input.dtype

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[2]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)

        sigmas = self.noise_scheduler.sigmas.to(device=model_input.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(model_input.device)
        timesteps = timesteps.to(model_input.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < model_input.ndim:
            sigma = sigma.unsqueeze(-1)
        sigmas = sigma

        if not self.noise_scheduler.config.invert_sigmas:
            noisy_video_latents = (1.0 - sigmas) * model_input + sigmas * noise
        else:
            noisy_video_latents = sigmas * model_input + (1 - sigmas) * noise
        
        if args.task == 't2v':
            noisy_model_input = noisy_video_latents
        elif args.task == 'i2v':
            noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=1)
        # no image_emb (pay attention)
        model_pred = model(
            hidden_states = noisy_model_input,
            encoder_hidden_states = model_kwargs['encoder_hidden_states'],
            encoder_attention_mask = model_kwargs['encoder_attention_mask'],
            embed_cfg_scale = model_kwargs['batch_embed_scale'],
            timestep = timesteps,
            return_dict=False,
        )[0]

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        if not self.noise_scheduler.config.invert_sigmas:
            target = noise - model_input
        else:
            target = model_input - noise
        loss = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
        loss = loss.mean()
        return loss

    # hunyuan
    def get_loss_controlnet(self, condition, model, model_input, model_kwargs, args, **kwargs):
        model.train()
        if args.task == 'i2v':
            model_input, image_latents = model_input
            if args.repeat:
                image_latents = image_latents.repeat(1, 1, model_input.shape[2], 1, 1)

            else:
                padding_shape = (model_input.shape[0], model_input.shape[1], model_input.shape[2]-1, *model_input.shape[3:])
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding], dim=2)
                
        dtype = model_input.dtype

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[2]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)

        sigmas = self.noise_scheduler.sigmas.to(device=model_input.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(model_input.device)
        timesteps = timesteps.to(model_input.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < model_input.ndim:
            sigma = sigma.unsqueeze(-1)
        sigmas = sigma

        if not self.noise_scheduler.config.invert_sigmas:
            noisy_video_latents = (1.0 - sigmas) * model_input + sigmas * noise
        else:
            noisy_video_latents = sigmas * model_input + (1 - sigmas) * noise
        
        if args.task == 't2v':
            noisy_model_input = noisy_video_latents
        elif args.task == 'i2v':
            noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=1)

        model_pred = model(
            hidden_states = noisy_model_input,
            encoder_hidden_states = model_kwargs['encoder_hidden_states'],
            encoder_attention_mask = model_kwargs['encoder_attention_mask'],
            embed_cfg_scale = model_kwargs['batch_embed_scale'],
            timestep = timesteps,
            controlnet_cond = condition,
            conditioning_scale = model_kwargs.get('conditioning_scale',1),
            controlnet_blocks_repeat = False, #NOTE：pay attention
            return_dict=False,
            image_emb = model_kwargs['image_emb'],
            image_latents = model_kwargs['image_latents']
        )[0]

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        if not self.noise_scheduler.config.invert_sigmas:
            target = noise - model_input
        else:
            target = model_input - noise
        # add_ocr mask here
        b, c, t, h, w = model_input.shape
        # print(b, c, t, h, w)
        ocr_mask = model_kwargs["ocr_mask"]
        ocr_mask = rearrange(ocr_mask, "b c t H W -> (b t) c H W")
        # 确保是0,1不会出现其他的值， 插值到latent shape
        ocr_mask = F.interpolate(ocr_mask, size=(h, w), mode='nearest') 
        ocr_mask = rearrange(ocr_mask, "(b t) c h w -> b c t h w", t=t) # c = 1
        ocr_mask = ocr_mask.expand(-1, c, -1, -1, -1)

        # auto broadcast
        loss = (weighting.float() * (model_pred.float() - target.float()) ** 2 * ocr_mask.float())

        if args.face_mask:
            face_mask = model_kwargs["face_mask"]
            face_mask = rearrange(face_mask, "b c t H W -> (b t) c H W")
            face_mask = F.interpolate(face_mask, size=(h, w), mode='nearest')
            face_mask = rearrange(face_mask, "(b t) c h w -> b c t h w", t=t)
            face_mask = face_mask.expand(-1, c, -1, -1, -1)
     
            loss = loss * ((args.face_weight - 1) * face_mask + 1).float()
        
        loss = loss.reshape(target.shape[0], -1).mean()
        return loss
    
    # 4.14待完善
    def get_loss_wanx(self, condition, model, model_input, model_kwargs, args, **kwargs):
        model.train()

        if args.task == 'i2v':
            model_input, image_latents = model_input
            if args.repeat:
                image_latents = image_latents.repeat(1, 1, model_input.shape[2], 1, 1)
        

        dtype = model_input.dtype
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[2]
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()

        timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)
        sigmas = self.noise_scheduler.sigmas.to(device=model_input.device, dtype=dtype)

        schedule_timesteps = self.noise_scheduler.timesteps.to(model_input.device)
        timesteps = timesteps.to(model_input.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < model_input.ndim:
            sigma = sigma.unsqueeze(-1)
        sigmas = sigma

        if not self.noise_scheduler.config.invert_sigmas:
            noisy_video_latents = (1.0 - sigmas) * model_input + sigmas * noise
        else:
            noisy_video_latents = sigmas * model_input + (1 - sigmas) * noise

        if args.task == 't2v':
            noisy_model_input = noisy_video_latents
        # add i2v
        elif args.task == 'i2v':
            noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=1)

    
        # add token replace
        do_token_replace = False
        loss_mask = torch.ones_like(model_input)
        token_replace_prob = model_kwargs.get('token_replace_prob', 0.0)

        num_frames = model_input.shape[2]
        num_replace_frames = 2
        if random.random() < token_replace_prob and num_frames > num_replace_frames:
            do_token_replace = True
            loss_mask[:, :, :num_replace_frames, :, :] = 0
            noisy_model_input[:, :, :num_replace_frames, :, :] = model_input[:, :, :num_replace_frames, :, :]


        model_pred = model(
            hidden_states = noisy_model_input,
            encoder_hidden_states = model_kwargs['encoder_hidden_states'],
            timestep = timesteps,
            controlnet_cond = condition,
            conditioning_scale = model_kwargs['conditioning_scale'],
            controlnet_blocks_repeat = False, #NOTE：pay attention
            image_pose = model_kwargs.get('image_pose',None),
            return_dict=False,
            image_latents = model_kwargs['image_latents'],
            token_replace = do_token_replace
        )[0]

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        if not self.noise_scheduler.config.invert_sigmas:
            target = noise - model_input
        else:
            target = model_input - noise

        # add_ocr mask here
        b, c, t, h, w = model_input.shape
        ocr_mask = model_kwargs["ocr_mask"]
        ocr_mask = rearrange(ocr_mask, "b c t H W -> (b t) c H W")
        # 确保是0,1不会出现其他的值， 插值到latent shape
        ocr_mask = F.interpolate(ocr_mask, size=(h, w), mode='nearest') 
        ocr_mask = rearrange(ocr_mask, "(b t) c h w -> b c t h w", t=t) # c = 1
        ocr_mask = ocr_mask.expand(-1, c, -1, -1, -1)

        final_mask = loss_mask.float() * ocr_mask.float()

        # auto broadcast
        loss = (weighting.float() * (model_pred.float() - target.float()) ** 2 * final_mask)

        if args.face_mask:
            face_mask = model_kwargs["face_mask"]
            face_mask = rearrange(face_mask, "b c t H W -> (b t) c H W")
            face_mask = F.interpolate(face_mask, size=(h, w), mode='nearest')
            face_mask = rearrange(face_mask, "(b t) c h w -> b c t h w", t=t)
            face_mask = face_mask.expand(-1, c, -1, -1, -1)
     
            loss = loss * ((args.face_weight - 1) * face_mask + 1).float()
        
        loss = loss.reshape(target.shape[0], -1).mean()
        return loss

   
  






class PyramidFlowMatching():
    #TODO: 相对SD3中的flow matching，缺少了两个部分，分别是timesteps的非均匀采样，以及loss的weighting机制
    def __init__(self):
        # self.stages = stages
        # self.scheduler = AutoRegressivePyramidFlowMatchEulerDiscreteScheduler(stages = stages, stage_range = stage_range)
        from opensora.sample.flow_matching_scheduler import PyramidFlowMatchEulerDiscreteScheduler
        self.scheduler = PyramidFlowMatchEulerDiscreteScheduler(stages=3, stage_range=[0, 1/3, 2/3, 1])
        self.stages = 3
    
    # 前向运算计算loss
    # 暂时的训练逻辑：防止batch_size过大，先每个step计算一个stage，然后累积3步
    def get_loss(self, model, model_input, model_kwargs, args, **kwargs):
        dtype = model_input.dtype
        assert 'stage_index' in kwargs, "passing stage index for pyramid training"
        stage_index = kwargs['stage_index'] # need

        # 这里对当前latent下采样到当前的stage进行计算，由于patch_size是2，所以下采样之后的长宽需要能够整除2
        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[2]
        height, width = model_input.shape[3:]
        resize_scale = 2**(self.stages - stage_index - 1)
        resized_height, resized_width = int(height / resize_scale), int(width / resize_scale)        
        resized_height += resized_height % 2
        resized_width += resized_width % 2
        model_input = rearrange(model_input, "b c t h w -> (b t) c h w")
        model_input = F.interpolate(model_input, size = (resized_height, resized_width), mode = 'nearest')
        model_input = rearrange(model_input, "(b t) c h w -> b c t h w", t = current_step_frame)
        height, width = model_input.shape[3:]

        noise = torch.randn_like(model_input)

        import ipdb
        ipdb.set_trace()

        #TODO: 目前是随机采样
        timesteps_indices = torch.randint(0, len(self.scheduler.timesteps_per_stage[stage_index]), (bsz,))
        timesteps = self.scheduler.timesteps_per_stage[stage_index][timesteps_indices].to(model_input.device)
        sigmas = self.scheduler.sigmas_per_stage[stage_index][timesteps_indices].to(model_input.device)
        start_sigma = self.scheduler.start_sigmas[stage_index]
        end_sigma = self.scheduler.end_sigmas[stage_index]
        
        stage_start_point = rearrange(model_input, "b c t h w -> (b t) c h w")
        stage_start_point = F.interpolate(stage_start_point, size = (int(height//2), int(width//2)), mode = 'bilinear')
        stage_start_point = F.interpolate(stage_start_point, size = (height, width), mode = 'nearest')
        stage_start_point = rearrange(stage_start_point, "(b t) c h w -> b c t h w", t = current_step_frame)

        if self.scheduler.invert_sigmas:
            stage_start_point = start_sigma * stage_start_point + (1 - start_sigma) * noise
            stage_end_point = end_sigma * model_input + (1 - end_sigma) * noise
        else:
            stage_start_point = (1 - start_sigma) * stage_start_point + start_sigma * noise
            stage_end_point = (1 - end_sigma) * model_input + end_sigma * noise
    
        while len(sigmas.shape) < model_input.ndim:
            sigmas = sigmas.unsqueeze(-1)
        if self.scheduler.invert_sigmas:
            noisy_model_input = (1.0 - sigmas) * stage_start_point + sigmas * stage_end_point
        else:
            noisy_model_input = sigmas * stage_start_point + (1 - sigmas) * stage_end_point

        model_pred = model(
            noisy_model_input,
            timesteps,
            **model_kwargs
        )[0]

        if args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=args.prediction_type)
        
        if self.scheduler.invert_sigmas:
            target = stage_end_point - stage_start_point
        else:
            target = stage_start_point - stage_end_point

        b, c, t_shard, _, _ = model_pred.shape
       
        # Compute regular loss.
        loss = ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
        loss = loss.mean()
        return loss

class AutoRegressivePyramidFlowMatching():
    #TODO: 相对SD3中的flow matching，缺少了两个部分，分别是timesteps的非均匀采样，以及loss的weighting机制
    def __init__(self, stages, stage_range, process_index, model_name='arpf', max_unit_num=8, frame_per_unit=1, max_history_len=4, extra_sample_steps=1):
        self.stages = stages
        self.scheduler = AutoRegressivePyramidFlowMatchEulerDiscreteScheduler(stages = stages, stage_range = stage_range)
        self.frame_per_unit = frame_per_unit # 每次训练多少个latent
        if model_name == 'arpf':
            self.vae_shift_factor = 0.1490
            self.vae_scale_factor = 1 / 1.8415
        elif model_name == 'flux_arpf':
            self.vae_shift_factor = -0.04
            self.vae_scale_factor = 1 / 1.8726

        # For the video latent
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986

        self.max_history_len = max_history_len
        assert self.max_history_len > self.stages - 1

        self.extra_sample_steps = extra_sample_steps

        self.iters = 0
        self.process_index = process_index
        self.max_unit_num = max_unit_num
        self.sample_ratios = [1,2,1]
    
    @torch.no_grad()
    def get_pyramid_latent(self, x, stage_num, generator=None):
        # TODO: 增加noise
        vae_latent_list = []
        vae_latent_list.append(x) #当前stage
        temp, height, width = x.shape[-3:]
        for _ in range(stage_num):
            height //= 2
            width //= 2
            height += height % 2
            width += width % 2
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = F.interpolate(x, size = (height, width), mode='bilinear')
            x = rearrange(x, '(b t) c h w -> b c t h w', t=temp)
            vae_latent_list.append(x)
        vae_latent_list = list(reversed(vae_latent_list)) # idx和stage_index相对应
        return vae_latent_list
    
    def sample_stage_length(self, num_stages, max_units = None):
        self.video_sync_group = self.max_unit_num
        max_units_in_training = 1 + ((self.max_unit_num - 1) // self.frame_per_unit)
        cur_rank = self.process_index

        self.iters = self.iters + 1
        total_turns =  max_units_in_training // self.video_sync_group #NOTE: video_sync_group
        update_turn = self.iters % total_turns

        # # uniformly sampling each position
        cur_highres_unit = max(int((cur_rank % self.video_sync_group + 1) + update_turn * self.video_sync_group), 1)
        cur_mid_res_unit = max(1 + max_units_in_training - cur_highres_unit, 1)
        cur_low_res_unit = cur_mid_res_unit #平衡下速度的考量

        if max_units is not None:
            cur_highres_unit = min(cur_highres_unit, max_units)
            cur_mid_res_unit = min(cur_mid_res_unit, max_units)
            cur_low_res_unit = min(cur_low_res_unit, max_units)

        length_list = [cur_low_res_unit, cur_mid_res_unit, cur_highres_unit]
        
        assert len(length_list) == num_stages

        return length_list

    @torch.no_grad()
    def add_pyramid_noise(
        self, 
        latents_list,
        sample_ratios=[1, 1, 1],
    ):
        """
        add the noise for each pyramidal stage
            noting that, this method is a general strategy for pyramid-flow, it 
            can be used for both image and video training.
            You can also use this method to train pyramid-flow with full-sequence 
            diffusion in video generation (without using temporal pyramid and autoregressive modeling)

        Params:
            latent_list: [low_res, mid_res, high_res] The vae latents of all stages
            sample_ratios: The proportion of each stage in the training batch
        """
        noise = torch.randn_like(latents_list[-1])
        device = noise.device
        dtype = latents_list[-1].dtype
        t = noise.shape[2]

        stages = self.stages
        tot_samples = noise.shape[0]
        assert tot_samples % (int(sum(sample_ratios))) == 0
        assert stages == len(sample_ratios)
        
        height, width = noise.shape[-2], noise.shape[-1]
        noise_list = [noise]
        cur_noise = noise
        for i_s in range(stages-1):
            height //= 2;width //= 2
            height += height % 2
            width += width % 2
            cur_noise = rearrange(cur_noise, 'b c t h w -> (b t) c h w')
            cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
            cur_noise = rearrange(cur_noise, '(b t) c h w -> b c t h w', t=t)
            noise_list.append(cur_noise)

        noise_list = list(reversed(noise_list))   # make sure from low res to high res
        
        # To calculate the padding batchsize and column size
        batch_size = tot_samples // int(sum(sample_ratios))
        column_size = int(sum(sample_ratios))
        
        column_to_stage = {} #每个数据对应哪个stage
        i_sum = 0
        for i_s, column_num in enumerate(sample_ratios):
            for index in range(i_sum, i_sum + column_num):
                column_to_stage[index] = i_s
            i_sum += column_num

        noisy_latents_list = []
        ratios_list = []
        targets_list = []
        timesteps_list = []
        training_steps = self.scheduler.config.num_train_timesteps

        # from low resolution to high resolution
        for index in range(column_size):
            i_s = column_to_stage[index]
            clean_latent = latents_list[i_s][index::column_size] # 哪些样本在这个column  # [bs, c, t, h, w]
            last_clean_latent = None if i_s == 0 else latents_list[i_s-1][index::column_size] # 否则的话，就是更早一级的latent
            start_sigma = self.scheduler.start_sigmas[i_s]
            end_sigma = self.scheduler.end_sigmas[i_s]
            
            if i_s == 0:
                start_point = noise_list[i_s][index::column_size] #stage 0的，初始的latent就是noise
            else:
                # Get the upsampled latent
                last_clean_latent = rearrange(last_clean_latent, 'b c t h w -> (b t) c h w')
                last_clean_latent = F.interpolate(last_clean_latent, size=(clean_latent.shape[-2], clean_latent.shape[-1]), mode='nearest')
                last_clean_latent = rearrange(last_clean_latent, '(b t) c h w -> b c t h w', t=t)
                start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent
            
            if i_s == stages - 1:
                end_point = clean_latent
            else:
                end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

            # To sample a timestep
            u = compute_density_for_timestep_sampling(
                weighting_scheme='random',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )

            indices = (u * training_steps).long()   # Totally 1000 training steps per stage
            indices = indices.clamp(0, training_steps-1)
            timesteps = self.scheduler.timesteps_per_stage[i_s][indices].to(device=device)
            ratios = self.scheduler.sigmas_per_stage[i_s][indices].to(device=device)

            while len(ratios.shape) < start_point.ndim:
                ratios = ratios.unsqueeze(-1)

            # interpolate the latent
            noisy_latents = ratios * start_point + (1 - ratios) * end_point

            # last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=device) * self.corrupt_ratio

            # [stage1_latent, stage2_latent, ..., stagen_latent], which will be concat after patching
            noisy_latents_list.append([noisy_latents.to(dtype)])
            ratios_list.append(ratios.to(dtype))
            timesteps_list.append(timesteps.to(dtype))
            targets_list.append(start_point - end_point)     # The standard rectified flow matching objective

        return noisy_latents_list, ratios_list, timesteps_list, targets_list

    @torch.no_grad()
    def add_pyramid_noise_with_temporal_pyramid(
        self, 
        latents_list,
        sample_ratios=[1, 1, 1],
    ):
        stages = self.stages
        tot_samples = latents_list[0].shape[0]
        device = latents_list[0].device
        dtype = latents_list[0].dtype
        assert tot_samples % (int(sum(sample_ratios))) == 0
        assert stages == len(sample_ratios)

        noise = torch.randn_like(latents_list[-1])
        t = noise.shape[2]

        max_units = 1 + (t - 1) // self.frame_per_unit
        num_units_per_stage = self.sample_stage_length(stages, max_units=max_units)   # [The unit number of each stage]
        height, width = noise.shape[-2], noise.shape[-1]
        noise_list = [noise]
        cur_noise = noise
        for i_s in range(stages-1):
            height //= 2;width //= 2
            height += height % 2
            width += width % 2 
            cur_noise = rearrange(cur_noise, 'b c t h w -> (b t) c h w')
            cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
            cur_noise = rearrange(cur_noise, '(b t) c h w -> b c t h w', t=t)
            noise_list.append(cur_noise)
        
        noise_list = list(reversed(noise_list))
        batch_size = tot_samples // int(sum(sample_ratios))
        column_size = int(sum(sample_ratios))
        column_to_stage = {}
        i_sum = 0
        for i_s, column_num in enumerate(sample_ratios):
            for index in range(i_sum, i_sum + column_num):
                column_to_stage[index] = i_s
            i_sum += column_num
        
        noisy_latents_list = []
        ratios_list = []
        targets_list = []
        timesteps_list = []
        training_steps = self.scheduler.config.num_train_timesteps

        for index in range(column_size):
            # First prepare the trainable latent construction
            i_s = column_to_stage[index]
            clean_latent = latents_list[i_s][index::column_size]   # [bs, c, t, h, w]
            last_clean_latent = None if i_s == 0 else latents_list[i_s-1][index::column_size]
            start_sigma = self.scheduler.start_sigmas[i_s]
            end_sigma = self.scheduler.end_sigmas[i_s]

            if i_s == 0:
                start_point = noise_list[i_s][index::column_size]
            else:
                # Get the upsampled latent
                last_clean_latent = rearrange(last_clean_latent, 'b c t h w -> (b t) c h w')
                last_clean_latent = F.interpolate(last_clean_latent, size=(clean_latent.shape[-2], clean_latent.shape[-1]), mode='nearest')
                last_clean_latent = rearrange(last_clean_latent, '(b t) c h w -> b c t h w', t=t)
                start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent
            
            if i_s == stages - 1:
                end_point = clean_latent
            else:
                end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

            u = compute_density_for_timestep_sampling(
                weighting_scheme='random',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )

            indices = (u * training_steps).long()   # Totally 1000 training steps per stage
            indices = indices.clamp(0, training_steps-1)
            timesteps = self.scheduler.timesteps_per_stage[i_s][indices].to(device=device)
            ratios = self.scheduler.sigmas_per_stage[i_s][indices].to(device=device)
            noise_ratios = ratios * start_sigma + (1 - ratios) * end_sigma

            while len(ratios.shape) < start_point.ndim:
                ratios = ratios.unsqueeze(-1)

            noisy_latents = ratios * start_point + (1 - ratios) * end_point

            # The flow matching object
            target_latents = start_point - end_point

            num_units = num_units_per_stage[i_s]
            num_units = min(num_units, 1 + (t - 1) // self.frame_per_unit)
            actual_frames = 1 + (num_units - 1) * self.frame_per_unit

            noisy_latents = noisy_latents[:, :, :actual_frames]
            target_latents = target_latents[:, :, :actual_frames]

            clean_latent = clean_latent[:, :, :actual_frames]
            stage_noise = noise_list[i_s][index::column_size][:, :, :actual_frames]

            # only the last latent takes part in training
            noisy_latents = noisy_latents[:, :, -self.frame_per_unit:] 
            target_latents = target_latents[:, :, -self.frame_per_unit:]

            last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=device) * 1/3

            if num_units == 1:
                stage_input = [noisy_latents.to(dtype)]
            else:
                last_cond_latent = clean_latent[:, :, -(2*self.frame_per_unit):-self.frame_per_unit]
                while len(last_cond_noisy_sigma.shape) < last_cond_latent.ndim:
                    last_cond_noisy_sigma = last_cond_noisy_sigma.unsqueeze(-1)
                last_cond_latent = last_cond_noisy_sigma * torch.randn_like(last_cond_latent) + (1 - last_cond_noisy_sigma) * last_cond_latent
                stage_input = [noisy_latents.to(dtype), last_cond_latent.to(dtype)]
                cur_unit_num = 2
                cur_stage = i_s
                while cur_unit_num < num_units:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_num += 1
                    cond_latents = latents_list[cur_stage][index::column_size][:, :, :actual_frames]
                    cond_latents = cond_latents[:, :, -(cur_unit_num * self.frame_per_unit) : -((cur_unit_num - 1) * self.frame_per_unit)]
                    cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(cond_latents.to(dtype))
                if cur_stage == 0 and cur_unit_num < num_units:
                    cond_latents = latents_list[0][index::column_size][:, :, :actual_frames]
                    cond_latents = cond_latents[:, :, :-(cur_unit_num * self.frame_per_unit)]

                    cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                    stage_input.append(cond_latents.to(dtype))

            stage_input = list(reversed(stage_input))
            noisy_latents_list.append(stage_input)
            ratios_list.append(ratios.to(dtype))
            timesteps_list.append(timesteps.to(dtype))
            targets_list.append(target_latents)  
        return noisy_latents_list, ratios_list, timesteps_list, targets_list

    @torch.no_grad()
    def get_vae_latent(self, video, use_temporal_pyramid=True):
        # if video.shape[2] == 1:
        #     # is image
        #     video = (video - self.vae_shift_factor) * self.vae_scale_factor
        # else:
        #     # is video
        #     video[:, :, :1] = (video[:, :, :1] - self.vae_shift_factor) * self.vae_scale_factor
        #     video[:, :, 1:] =  (video[:, :, 1:] - self.vae_video_shift_factor) * self.vae_video_scale_factor
        vae_latent_list = self.get_pyramid_latent(video, self.stages - 1)
        if use_temporal_pyramid:
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.add_pyramid_noise_with_temporal_pyramid(vae_latent_list, self.sample_ratios)
        else:
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.add_pyramid_noise(vae_latent_list, self.sample_ratios)
        return noisy_latents_list, ratios_list, timesteps_list, targets_list
    
    def calculate_loss(self, model_preds_list, targets_list):
        loss_list = []
        for model_pred, target in zip(model_preds_list, targets_list):
            loss_weight = torch.ones_like(target)
            loss = torch.mean(
                (loss_weight.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss_list.append(loss)
        diffusion_loss = torch.cat(loss_list, dim=0).mean()
        return diffusion_loss

    def get_loss(self, model, model_input, model_kwargs, args, accelerator, **kwargs): #代码部分仍需要注意，因为这里会根据rank来进行划分
        video = model_input
        xdim = video.ndim
        device = video.device
        with torch.no_grad(), accelerator.autocast():
            batch_size = len(video)
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = \
                model_kwargs['encoder_hidden_states'], model_kwargs['encoder_attention_mask'], model_kwargs['encoder_pooled_projections']
            noisy_latents_list, ratios_list, timesteps_list, targets_list = self.get_vae_latent(video, use_temporal_pyramid=True)

        timesteps = torch.cat([timestep.unsqueeze(-1) for timestep in timesteps_list], dim=-1)
        timesteps = timesteps.reshape(-1)

        assert timesteps.shape[0] == prompt_embeds.shape[0]

        model_preds_list = model(
            sample=noisy_latents_list,
            timestep_ratio=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled_prompt_embeds,
        )

        return self.calculate_loss(model_preds_list, targets_list)

    # 前向运算计算loss
    def get_loss_2(self, model, model_input, model_kwargs, args, **kwargs): #实际会更慢
        # TODO: 优化自回归训练策略
        # 如果使用不同的自回归长度，无疑会让sample的数量增长很多
        # 修改成不同stages
        dtype = model_input.dtype
        assert 'stage_index' in kwargs, "passing stage index for pyramid training"
        stage_index = kwargs['stage_index'] # need
        bsz = model_input.shape[0]
        temp = model_input.shape[2]
        frame_per_unit = self.frame_per_unit 
        model_input[:, :, 0:1] = (model_input[:, :, 0:1] -  self.vae_shift_factor) * self.vae_scale_factor
        if model_input.shape[2] > 1:
            model_input[:, :, 1:] = (model_input[:, :, 1:] - self.vae_video_shift_factor) * self.vae_video_scale_factor
        clean_latents_list = self.get_pyramid_latent(model_input, self.stages-1)

        #TODO: 目前是随机采样 timesteps用相同的
        #NOTE: 开头和结尾都能够取到
        #TODO: 暂时只能支持frame_per_unit=1的情况
        sample_list = []
        target_list = []
        timesteps_list = []
        # stage_list = np.arange(self.stages) #同时包含所有stages
        
        self.iters += 1
        assert self.stages == 3 
        stage_list = [1] #进行某个特定stage的优化
        if (self.iters + self.process_index) % 2 == 0:
            stage_list.append(0)
        elif (self.iters + self.process_index) % 2 == 1:
            stage_list.append(2)
        else:
            raise ValueError

        # stage_list = [0,1,1,2]
        # stage_2_length = (self.iters + self.process_index) % (self.max_unit_num-1) + 1
        # stage_1_length = self.max_unit_num - 1 - (self.iters + self.process_index) % (self.max_unit_num-1)
        # stage_0_length = stage_2_length #按照这个逻辑的话，stage2的耗时最长，因此最好让stage0和stage1用相同的步数
        
        # 1到6
        #如果每个stage包含多个自回归长度的话，不如从这里入手，更加均匀 应该是0到6
        frame_idx_list = [(self.process_index) % (self.max_unit_num), self.max_unit_num - 1 - (self.process_index) % (self.max_unit_num)]
        print(self.process_index, stage_list, frame_idx_list)

        # frame_idx_list = [stage_0_length, stage_1_length, stage_1_length, stage_2_length]

        for idx, stage_index in enumerate(stage_list):
            #TODO: 按照rank进行分配
            # u = compute_density_for_timestep_sampling(
            #     weighting_scheme='random',
            #     batch_size=bsz,
            #     logit_mean=0.0,
            #     logit_std=1.0,
            #     mode_scale=1.29,
            # )
            # indices = (u * len(self.scheduler.timesteps_per_stage[stage_index])).long()
            # timesteps_indices = indices.clamp(0, len(self.scheduler.timesteps_per_stage[stage_index])-1)

            timesteps_indices = torch.randint(0, len(self.scheduler.timesteps_per_stage[stage_index]), (bsz,))
            timesteps = self.scheduler.timesteps_per_stage[stage_index][timesteps_indices].to(model_input.device, dtype=dtype)
            sigmas = self.scheduler.sigmas_per_stage[stage_index][timesteps_indices].to(model_input.device, dtype=dtype)
            start_sigma = self.scheduler.start_sigmas[stage_index]
            end_sigma = self.scheduler.end_sigmas[stage_index]
            while len(sigmas.shape) < model_input.ndim:
                sigmas = sigmas.unsqueeze(-1)
            # timesteps_list.append(timesteps.unsqueeze(1))

            #可以最多选择两种长度？
            #TODO: 可以修改成根据rank来在不同的steps中进行均衡的策略
            frame_idx_list_stage = [frame_idx_list[idx]]
            # if temp > 2:
            #     frame_idx_list = list(np.random.permutation(np.arange(1, temp)))[:self.extra_sample_steps]

            timesteps_list.append(timesteps.unsqueeze(1).repeat(1, len(frame_idx_list_stage)))

            for frame_idx in frame_idx_list_stage:
                frame_idx = min(model_input.shape[2]-1, frame_idx)
                cur_latent = model_input[:, :, frame_idx:frame_idx+1]
                if frame_idx == 0:
                    stage_input = []
                else:
                    last_cond_latent = clean_latents_list[stage_index][:, :, frame_idx-frame_per_unit:frame_idx]
                    last_cond_noisy_sigma = torch.rand(size=(bsz,), device=model_input.device, dtype=cur_latent.dtype) * 1 / 3
                    while len(last_cond_noisy_sigma.shape) < last_cond_latent.ndim:
                        last_cond_noisy_sigma = last_cond_noisy_sigma.unsqueeze(-1)
                    last_cond_latent = last_cond_noisy_sigma * torch.randn_like(last_cond_latent, device=model_input.device, dtype=cur_latent.dtype) + (1 - last_cond_noisy_sigma) * last_cond_latent

                    stage_input = [last_cond_latent]
                    cur_unit_num = frame_idx
                    cur_stage = stage_index
                    cur_unit_ptx = 1 #这里是diff
                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage-1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, frame_idx-cur_unit_ptx * frame_per_unit:frame_idx - (cur_unit_ptx-1)*frame_per_unit]
                        cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents, device=model_input.device, dtype=cur_latent.dtype)  + (1 - last_cond_noisy_sigma) * cond_latents
                        stage_input.append(cond_latents)
                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[cur_stage][:, :, :frame_idx - cur_unit_ptx * frame_per_unit]
                        if cond_latents.shape[2] > self.max_history_len-len(stage_input):
                            cond_latents = cond_latents[:, :, -(self.max_history_len-len(stage_input)):]
                        cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents, device=model_input.device, dtype=cur_latent.dtype)  + (1 - last_cond_noisy_sigma) * cond_latents
                        stage_input.append(cond_latents)
                    stage_input = list(reversed(stage_input))
                height, width = cur_latent.shape[3:]
                cur_latent = rearrange(cur_latent, "b c t h w -> (b t) c h w")
                noise = torch.randn_like(cur_latent, device = cur_latent.device, dtype=cur_latent.dtype) #noise考虑用不同的
                for _ in range(self.stages - stage_index - 1):
                    height //= 2
                    width //= 2
                    height += height % 2
                    width += width % 2
                    cur_latent = F.interpolate(cur_latent, size = (height, width), mode = 'bilinear')
                    noise = F.interpolate(noise, size = (height, width), mode = 'bilinear') * 2
                cur_latent = rearrange(cur_latent, "(b t) c h w -> b c t h w", t = frame_per_unit)
                noise = rearrange(noise, "(b t) c h w -> b c t h w", t = frame_per_unit)
                height, width = cur_latent.shape[3:]

                stage_start_point = rearrange(cur_latent, "b c t h w -> (b t) c h w")
                stage_start_point = F.interpolate(stage_start_point, size = (int(height//2), int(width//2)), mode = 'bilinear')
                stage_start_point = F.interpolate(stage_start_point, size = (height, width), mode = 'nearest') #TODO: check这里是否修改
                stage_start_point = rearrange(stage_start_point, "(b t) c h w -> b c t h w", t = frame_per_unit)
                stage_start_point = (1 - start_sigma) * stage_start_point + start_sigma * noise
                stage_end_point = (1 - end_sigma) * cur_latent + end_sigma * noise

                noisy_model_input = (1 - sigmas) * stage_end_point + sigmas * stage_start_point
            
                noisy_model_input = stage_input + [noisy_model_input]
                target = stage_start_point - stage_end_point

                sample_list.append(noisy_model_input)
                target_list.append(target)
        timesteps = torch.cat(timesteps_list, dim=1).reshape(len(sample_list)*bsz)
        model_pred = model(
            sample = sample_list, #实际上可以同一个样本构建多个自回归的序列。这样就不会出现问题了
            timestep_ratio = timesteps, # torch.repeat_interleave(timesteps, repeats = len(sample_list), dim=0),
            encoder_hidden_states = torch.repeat_interleave(model_kwargs['encoder_hidden_states'], repeats = len(sample_list), dim=0),
            encoder_attention_mask = torch.repeat_interleave(model_kwargs['encoder_attention_mask'], repeats = len(sample_list), dim=0),
            pooled_projections = torch.repeat_interleave(model_kwargs['encoder_pooled_projections'], repeats = len(sample_list), dim=0)
        )

        mask = model_kwargs.get('attention_mask', None)
        if torch.all(mask.bool()):
            mask = None
        assert mask is None #否则的话，就需要对mask也进行下采样了
        
        # model_pred = torch.stack(model_pred)
        # target = torch.stack(target_list)
        model_pred_ = torch.cat([pred.reshape(bsz, -1) for pred in model_pred], dim=1)
        target_ = torch.cat([gt.reshape(bsz, -1) for gt in target_list], dim=1)
        loss = ((model_pred_.float() - target_.float()) ** 2).reshape(bsz, -1) 
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def get_valid_loss(self, model, model_input, model_kwargs, args, **kwargs):
        #验证的逻辑: 3个stage，所有可能的自回归长度
        dtype = model_input.dtype
        assert 'stage_index' in kwargs, "passing stage index for pyramid training"
        stage_index = kwargs['stage_index'] # need
        assert 'evaluate_steps' in kwargs, "passing how many steps to evaluate"
        evaluate_steps = kwargs['evaluate_steps']

        bsz = model_input.shape[0]
        temp = model_input.shape[2]
        frame_per_unit = self.frame_per_unit 
        # model_input[:, :, 0:1] = (model_input[:, :, 0:1] -  self.vae_shift_factor) * self.vae_scale_factor
        # if model_input.shape[2] > 1:
        #     model_input[:, :, 1:] = (model_input[:, :, 1:] - self.vae_video_shift_factor) * self.vae_video_scale_factor
        generator = torch.torch.Generator(device = model_input.device)
        generator.manual_seed(42)
        clean_latents_list = self.get_pyramid_latent(model_input, self.stages-1, generator=generator)
        past_condition_list = []
        cur_latent_list = []
        for frame_idx in range(temp):
            cur_latent = model_input[:, :, frame_idx:frame_idx+1]
            if frame_idx == 0:
                stage_input = []
            else:
                last_cond_latent = clean_latents_list[stage_index][:, :, frame_idx-frame_per_unit:frame_idx]
                stage_input = [last_cond_latent]
                cur_unit_num = frame_idx
                cur_stage = stage_index
                cur_unit_ptx = 1
                while cur_unit_ptx < cur_unit_num:
                    cur_stage = max(cur_stage-1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_ptx += 1
                    cond_latents = clean_latents_list[cur_stage][:, :, frame_idx-cur_unit_ptx * frame_per_unit:frame_idx - (cur_unit_ptx-1)*frame_per_unit]
                    stage_input.append(cond_latents)
                if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                    cond_latents = clean_latents_list[cur_stage][:, :, :frame_idx - cur_unit_ptx * frame_per_unit]
                    if cond_latents.shape[2] > self.max_history_len-len(stage_input):
                        cond_latents = cond_latents[:, :, -(self.max_history_len-len(stage_input)):]
                    stage_input.append(cond_latents)
                stage_input = list(reversed(stage_input))
            past_condition_list.append(stage_input)
            height, width = cur_latent.shape[3:]
            cur_latent = rearrange(cur_latent, "b c t h w -> (b t) c h w")
            for _ in range(self.stages - stage_index - 1):
                height //= 2
                width //= 2
                height += height % 2
                width += width % 2
                cur_latent = F.interpolate(cur_latent, size = (height, width), mode = 'bilinear')
            cur_latent = rearrange(cur_latent, "(b t) c h w -> b c t h w", t = frame_per_unit)
            cur_latent_list.append(cur_latent)

        loss_list = []
        for step_i in range(evaluate_steps):
            generator = torch.torch.Generator(device = model_input.device)
            generator.manual_seed(42+step_i)
            timesteps_indices = torch.ones((bsz,), dtype=int) * (step_i * len(self.scheduler.timesteps_per_stage[stage_index]) // evaluate_steps)
            timesteps = self.scheduler.timesteps_per_stage[stage_index][timesteps_indices].to(model_input.device, dtype=dtype)
            sigmas = self.scheduler.sigmas_per_stage[stage_index][timesteps_indices].to(model_input.device, dtype=dtype)
            start_sigma = self.scheduler.start_sigmas[stage_index]
            end_sigma = self.scheduler.end_sigmas[stage_index]
            while len(sigmas.shape) < model_input.ndim:
                sigmas = sigmas.unsqueeze(-1)
            
            #TODO: 暂时只能支持frame_per_unit=1的情况
            sample_list = []
            target_list = []
            for frame_idx in range(temp):
                stage_input = past_condition_list[frame_idx]
                cur_latent = cur_latent_list[frame_idx]
                
                height, width = cur_latent.shape[3:]
                # noise = torch.randn_like(cur_latent, generator=generator) #noise考虑用不同的
                noise = randn_tensor(cur_latent.shape, generator=generator, device=cur_latent.device, dtype=cur_latent.dtype)

                stage_start_point = rearrange(cur_latent, "b c t h w -> (b t) c h w")
                stage_start_point = F.interpolate(stage_start_point, size = (int(height//2), int(width//2)), mode = 'bilinear')
                stage_start_point = F.interpolate(stage_start_point, size = (height, width), mode = 'nearest') #TODO: check这里是否修改
                stage_start_point = rearrange(stage_start_point, "(b t) c h w -> b c t h w", t = frame_per_unit)
                stage_start_point = (1 - start_sigma) * stage_start_point + start_sigma * noise
                stage_end_point = (1 - end_sigma) * cur_latent + end_sigma * noise

                noisy_model_input = (1 - sigmas) * stage_end_point + sigmas * stage_start_point
            
                noisy_model_input = stage_input + [noisy_model_input]
                target = stage_start_point - stage_end_point

                sample_list.append(noisy_model_input)
                target_list.append(target)
            model_pred = model(
                sample = sample_list, #实际上可以同一个样本构建多个自回归的序列。这样就不会出现问题了
                timestep_ratio = torch.repeat_interleave(timesteps, repeats = len(sample_list), dim=0),
                encoder_hidden_states = torch.repeat_interleave(model_kwargs['encoder_hidden_states'], repeats = len(sample_list), dim=0),
                encoder_attention_mask = torch.repeat_interleave(model_kwargs['encoder_attention_mask'], repeats = len(sample_list), dim=0),
                pooled_projections = torch.repeat_interleave(model_kwargs['encoder_pooled_projections'], repeats = len(sample_list), dim=0)
            )

            # mask = model_kwargs.get('attention_mask', None)
            # if torch.all(mask.bool()):
            #     mask = None
            # assert mask is None #否则的话，就需要对mask也进行下采样了
            
            model_pred = torch.stack(model_pred)
            target = torch.stack(target_list)
            loss = ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
            # loss = loss.mean()
            loss_list.append(loss)
        loss_list = torch.stack(loss_list)
        return loss_list

class EDM():
    def __init__(self):
        self.edm_loss = EDMLoss()
        print('====> use EDM Precondition !!!')

    def get_loss(self, model, model_input, model_kwargs, args, **kwargs):
        model = partial(model, **model_kwargs)
        loss = self.edm_loss(model, model_input)
        mask = model_kwargs.get('attention_mask', None)
        if torch.all(mask.bool()):
            mask = None
        b, c, t_shard, _, _ = loss.shape
        if mask is not None:
            # 实际上不知道这里是哪里，因此需要所有帧的attention mask是一样的，确保在时序上不会补帧即可
            mask = mask.unsqueeze(1).repeat(1, c, 1, 1, 1).float()[:, :, -t_shard:, :, :]  # b t h w -> b c t h w
            mask = mask.reshape(b, -1)
        loss = loss.reshape(b, -1)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss